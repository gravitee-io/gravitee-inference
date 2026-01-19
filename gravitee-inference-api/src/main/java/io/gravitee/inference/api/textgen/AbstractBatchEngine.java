/*
 * Copyright © 2015 The Gravitee team (http://gravitee.io)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package io.gravitee.inference.api.textgen;

import io.gravitee.inference.api.EngineAdapter;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Consumer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Abstract base class for batch inference engines.
 * Provides thread-safe sequence management, automatic queuing, stop sequence detection,
 * and token streaming. Engine-specific logic is delegated to an {@link EngineAdapter}.
 *
 * <p>This class handles all the complex orchestration:
 * <ul>
 *   <li>Thread-safe sequence addition, removal, and cancellation</li>
 *   <li>Automatic slot allocation and pending queue management</li>
 *   <li>Stop sequence detection with buffering</li>
 *   <li>Performance tracking and metrics collection</li>
 *   <li>Proper resource cleanup and shutdown</li>
 * </ul>
 *
 * <p>Implementers only need to provide an {@link EngineAdapter} that handles
 * the actual engine-specific operations.</p>
 *
 * @param <CONFIG> Engine configuration type
 * @param <REQUEST> Generation request type
 * @param <TOKEN> Token type
 * @param <STATE> Engine-specific sequence state type
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public abstract class AbstractBatchEngine<CONFIG, REQUEST extends GenerationRequest, TOKEN, STATE> implements AutoCloseable {

  private static final Logger LOGGER = LoggerFactory.getLogger(AbstractBatchEngine.class);
  private final BatchEngineConfig engineConfig;
  private final EngineAdapter<CONFIG, REQUEST, TOKEN, STATE> adapter;
  private final Map<Integer, SequenceState<STATE>> sequences = new ConcurrentHashMap<>();
  private final Map<Integer, Integer> externalToInternal = new ConcurrentHashMap<>();
  private final Deque<QueuedSequence<REQUEST>> pending = new ArrayDeque<>();
  private final Deque<Integer> availableSlots;
  private final ReentrantLock lock = new ReentrantLock();
  private final Condition hasWork = lock.newCondition();
  private final AtomicBoolean running = new AtomicBoolean(false);
  private Consumer<InferenceToken<TOKEN>> tokenConsumer;
  private ExecutorService executor;
  private Future<?> workerFuture;

  /**
   * Creates a new batch engine with custom configuration.
   *
   * @param engineConfig The engine configuration
   * @param adapter The engine adapter
   */
  protected AbstractBatchEngine(BatchEngineConfig engineConfig, EngineAdapter<CONFIG, REQUEST, TOKEN, STATE> adapter) {
    this.engineConfig = Objects.requireNonNull(engineConfig, "engineConfig is required");
    this.adapter = Objects.requireNonNull(adapter, "adapter is required");
    this.availableSlots = new ArrayDeque<>(engineConfig.maxConcurrentSequences());
    for (int i = 0; i < engineConfig.maxConcurrentSequences(); i++) {
      availableSlots.addLast(i);
    }
  }

  /**
   * Starts the engine with a token consumer.
   *
   * @param tokenConsumer Callback for receiving generated tokens
   * @throws IllegalStateException if already started
   * @throws NullPointerException if tokenConsumer is null
   */
  public void start(Consumer<InferenceToken<TOKEN>> tokenConsumer) {
    Objects.requireNonNull(tokenConsumer, "tokenConsumer is required");
    this.tokenConsumer = tokenConsumer;
    if (!running.compareAndSet(false, true)) {
      throw new IllegalStateException("Engine is already running");
    }

    executor = Executors.newSingleThreadExecutor(runnable -> {
      Thread thread = new Thread(runnable, "batch-engine-iterator");
      thread.setUncaughtExceptionHandler((t, e) -> {
        running.set(false);
        LOGGER.error("Uncaught exception in batch engine worker thread: {}", e.getMessage());
      });
      return thread;
    });
    workerFuture = executor.submit(this::runLoop);
  }

  /**
   * Adds a sequence to be processed.
   *
   * <p>If no slots are available, the sequence is queued automatically.
   * If the pending queue is full, the sequence is rejected.</p>
   *
   * @param seqId External sequence ID (client-facing)
   * @param request The generation request
   * @throws IllegalStateException if pending queue is full
   */
  public void addSequence(int seqId, REQUEST request) {
    Objects.requireNonNull(request, "request is required");

    lock.lock();
    try {
      // Check for duplicate sequences
      if (externalToInternal.containsKey(seqId) || containsPending(seqId)) {
        return;
      }

      // Validate request
      PromptStats stats = adapter.validateRequest(request);
      if (!stats.fitsInContext()) {
        emitLength(seqId, stats.promptTokens());
        return;
      }

      // Check if queue is full
      if (pending.size() >= engineConfig.queueCapacity()) {
        throw new IllegalStateException("Pending queue is full (capacity: " + engineConfig.queueCapacity() + ")");
      }

      // Queue or start immediately
      if (availableSlots.isEmpty()) {
        pending.addLast(new QueuedSequence<>(seqId, request));
      } else {
        startSequence(seqId, request);
      }
    } finally {
      lock.unlock();
    }
  }

  /**
   * Cancels a sequence.
   *
   * @param seqId External sequence ID
   * @return The final token if the sequence was active, null otherwise
   */
  public InferenceToken<TOKEN> cancelSequence(int seqId) {
    lock.lock();
    try {
      // Remove from pending queue
      if (removePending(seqId)) {
        return null;
      }

      // Cancel active sequence
      Integer internalId = externalToInternal.get(seqId);
      if (internalId == null) {
        return null;
      }

      SequenceState<STATE> state = sequences.get(internalId);
      if (state != null && adapter.getFinishReason(state.engineState).isEmpty()) {
        adapter.removeSequence(internalId);
        return finalizeSequence(state);
      }
      return null;
    } finally {
      lock.unlock();
    }
  }

  /**
   * Main worker loop that processes sequences.
   */
  private void runLoop() {
    while (running.get()) {
      lock.lock();
      try {
        // Wait for work
        while (running.get() && sequences.isEmpty()) {
          hasWork.await();
        }
        if (!running.get()) {
          return;
        }

        // Process next batch
        EngineAdapter.EngineOutput<TOKEN, STATE> output;
        try {
          var optOutput = adapter.processNextBatch();
          if (optOutput.isEmpty()) {
            emitFinals();
            continue;
          }
          output = optOutput.get();
        } catch (Exception e) {
          LOGGER.error("Error processing batch: {}", e.getMessage());
          continue;
        }

        // Handle output
        SequenceState<STATE> state = sequences.get(output.sequenceId());
        if (state != null) {
          processOutput(state, output.token());
        }

        emitFinals();
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        return;
      } catch (Exception e) {
        LOGGER.error("Error processing batch: {}", e.getMessage());
      } finally {
        lock.unlock();
      }
    }
  }

  /**
   * Processes a token output for a sequence.
   */
  private void processOutput(SequenceState<STATE> state, TOKEN token) {
    // Handle null or empty tokens
    if (token == null || (token instanceof String && ((String) token).isEmpty())) {
      return;
    }

    // Update token counts from engine state
    updateTokenCounts(state);

    // Convert token to string for stop detection
    String tokenText = token.toString();
    TokenEmission emission = state.consume(tokenText);

    // Emit token if there's text to emit
    if (!emission.text().isEmpty()) {
      InferenceToken<TOKEN> inferenceToken = buildToken(state, emission.text(), state.index++, false);
      emitToken(inferenceToken);
    }

    // Check if stop sequence matched
    if (emission.stopMatched()) {
      adapter.removeSequence(state.conversationId);
      finalizeSequence(state);
    }
  }

  /**
   * Emits final tokens for completed sequences.
   */
  private void emitFinals() {
    for (var entry : sequences.entrySet()) {
      SequenceState<STATE> state = entry.getValue();
      emitFinalIfNeeded(state);
    }
  }

  /**
   * Emits final token for a sequence if it's finished.
   */
  private void emitFinalIfNeeded(SequenceState<STATE> state) {
    if (state == null || state.finalSent) {
      return;
    }

    var finishReason = adapter.getFinishReason(state.engineState);
    if (finishReason.isEmpty()) {
      return;
    }

    // Flush pending tokens
    String pending = state.flushPending();
    if (!pending.isEmpty()) {
      InferenceToken<TOKEN> chunk = buildToken(state, pending, state.index++, false);
      emitToken(chunk);
    }

    // Emit final token
    InferenceToken<TOKEN> finalToken = finalizeSequence(state);
    if (finalToken != null) {
      emitToken(finalToken);
    }
  }

  /**
   * Finalizes a sequence and cleans up resources.
   */
  private InferenceToken<TOKEN> finalizeSequence(SequenceState<STATE> state) {
    if (state == null || state.finalSent) {
      return null;
    }

    var finishReason = adapter.getFinishReason(state.engineState);
    if (finishReason.isEmpty()) {
      return null;
    }

    state.finalSent = true;

    // Update token counts before building the final token
    updateTokenCounts(state);

    // Remove from adapter
    adapter.removeSequence(state.conversationId);

    // Build final token
    InferenceToken<TOKEN> token = new InferenceToken<>(
      state.externalId,
      null,
      state.index,
      true,
      finishReason.get(),
      state.inputTokens,
      state.outputTokens,
      state.reasoningTokens,
      state.toolTokens,
      adapter.buildPerformance(state.engineState)
    );

    // Cleanup engine state
    try {
      adapter.cleanupSequenceState(state.engineState);
    } catch (Exception e) {
      LOGGER.error("Error cleaning up sequence state: {}", e.getMessage());
    }

    // Remove from tracking
    sequences.remove(state.conversationId);
    externalToInternal.remove(state.externalId);
    availableSlots.addLast(state.conversationId);

    // Start next pending if enabled
    if (engineConfig.enableAutoStart()) {
      startNextPending();
    }

    return token;
  }

  /**
   * Starts a sequence for processing.
   */
  private void startSequence(int seqId, REQUEST request) {
    Integer internalId = availableSlots.pollFirst();
    if (internalId == null) {
      pending.addLast(new QueuedSequence<>(seqId, request));
      return;
    }

    try {
      STATE engineState = adapter.createSequenceState(internalId, request);
      if (engineState == null) {
        availableSlots.addLast(internalId);
        return;
      }

      sequences.put(internalId, new SequenceState<>(internalId, seqId, engineState, request.stop()));
      externalToInternal.put(seqId, internalId);
      hasWork.signalAll();
    } catch (Exception e) {
      LOGGER.error("Error starting sequence: {}", e.getMessage());
      availableSlots.addLast(internalId);
    }
  }

  /**
   * Starts the next pending sequence if a slot is available.
   */
  private void startNextPending() {
    if (pending.isEmpty() || availableSlots.isEmpty()) {
      return;
    }
    QueuedSequence<REQUEST> next = pending.pollFirst();
    if (next != null) {
      startSequence(next.seqId(), next.request());
    }
  }

  /**
   * Removes a sequence from the pending queue.
   */
  private boolean removePending(int seqId) {
    var iterator = pending.iterator();
    while (iterator.hasNext()) {
      if (iterator.next().seqId() == seqId) {
        iterator.remove();
        return true;
      }
    }
    return false;
  }

  /**
   * Checks if a sequence ID is in the pending queue.
   */
  private boolean containsPending(int seqId) {
    return pending.stream().anyMatch(q -> q.seqId() == seqId);
  }

  /**
   * Emits a length-failed token.
   */
  private void emitLength(int seqId, int promptTokens) {
    InferenceToken<TOKEN> token = new InferenceToken<>(seqId, null, 0, true, "length", promptTokens, 0, 0, 0, null);
    emitToken(token);
  }

  /**
   * Builds an inference token.
   */
  @SuppressWarnings("unchecked")
  private InferenceToken<TOKEN> buildToken(SequenceState<STATE> state, String text, int index, boolean isFinal) {
    TOKEN token;
    if (state.tokenType == String.class) {
      token = (TOKEN) text;
    } else {
      // For non-string token types, you'd need to implement custom conversion
      token = null;
    }

    var finishReason = adapter.getFinishReason(state.engineState);
    return new InferenceToken<>(
      state.externalId,
      token,
      index,
      isFinal,
      finishReason.orElse(null),
      state.inputTokens,
      state.outputTokens,
      state.reasoningTokens,
      state.toolTokens,
      isFinal ? adapter.buildPerformance(state.engineState) : null
    );
  }

  /**
   * Safely emits a token to the consumer.
   */
  private void emitToken(InferenceToken<TOKEN> token) {
    Consumer<InferenceToken<TOKEN>> consumer = this.tokenConsumer;
    if (consumer != null) {
      consumer.accept(token);
    }
  }

  /**
   * Updates token counts from the engine state.
   */
  private void updateTokenCounts(SequenceState<STATE> state) {
    var counts = adapter.getTokenCounts(state.engineState);
    state.inputTokens = counts.inputTokens();
    state.outputTokens = counts.outputTokens();
    state.reasoningTokens = counts.reasoningTokens();
    state.toolTokens = counts.toolTokens();
  }

  @Override
  public void close() {
    running.set(false);
    lock.lock();
    try {
      hasWork.signalAll();
    } finally {
      lock.unlock();
    }
    if (executor != null) {
      executor.shutdownNow();
    }
    if (workerFuture != null) {
      workerFuture.cancel(true);
    }
    adapter.shutdown();
  }
}
