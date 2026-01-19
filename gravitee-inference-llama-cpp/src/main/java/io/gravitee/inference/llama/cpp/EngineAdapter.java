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
package io.gravitee.inference.llama.cpp;

import io.gravitee.inference.api.memory.InsufficientVramException;
import io.gravitee.inference.api.memory.MemoryCheckPolicy;
import io.gravitee.inference.api.memory.MemoryEstimate;
import io.gravitee.inference.api.textgen.InferencePerformance;
import io.gravitee.inference.api.textgen.PromptStats;
import io.gravitee.llama.cpp.BatchIterator;
import io.gravitee.llama.cpp.ConversationState;
import io.gravitee.llama.cpp.FinishReason;
import io.gravitee.llama.cpp.LlamaOutput;
import io.gravitee.llama.cpp.MtmdMedia;
import java.util.Optional;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Engine adapter for llama.cpp backend.
 * Handles all llama.cpp-specific operations while the AbstractBatchEngine
 * manages sequence lifecycle, queuing, and token emission.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public class EngineAdapter
  implements io.gravitee.inference.api.EngineAdapter<ModelConfig, Request, String, ConversationState> {

  private static final Logger LOGGER = LoggerFactory.getLogger(EngineAdapter.class);

  private final Model model;
  private final BatchIterator iterator;

  public EngineAdapter(ModelConfig config) {
    runMemoryCheck(config);
    this.model = new Model(config);
    this.iterator = model.newBatchIterator();
  }

  private static void runMemoryCheck(ModelConfig config) {
    MemoryCheckPolicy policy = config.memoryCheckPolicy();
    if (policy == null || policy == MemoryCheckPolicy.DISABLED) {
      LOGGER.info("Memory pre-flight check is disabled");
      return;
    }
    String modelName = config.modelPath().getFileName().toString();
    LOGGER.info("Running memory pre-flight check for {} (policy={})", modelName, policy);
    MemoryEstimate estimate = LlamaMemoryEstimator.estimate(
      config.modelPath(),
      config.mmprojPath(),
      config.loraPath(),
      config.nGpuLayers(),
      config.nCtx(),
      config.nSeqMax(),
      config.rpcServers(),
      config.logLevel()
    );
    if (estimate.isUnknown()) {
      LOGGER.info("Memory pre-flight: estimate unavailable — skipping check");
      return;
    }
    LOGGER.info("Memory pre-flight: {}", estimate.toHumanReadable());
    if (estimate.willFit()) {
      return;
    }
    if (policy == MemoryCheckPolicy.FAIL) {
      throw new InsufficientVramException(modelName, estimate);
    }
    LOGGER.warn("Memory pre-flight: model may not fit — {}", estimate.suggestion());
  }

  @Override
  public ConversationState createSequenceState(int internalId, Request request) {
    ConversationState state = model.newConversation(internalId, request);
    // Add the state to the iterator so it can be processed
    iterator.addState(state);
    return state;
  }

  @Override
  public PromptStats validateRequest(Request request) {
    var stats = model.promptStats(request);
    return new PromptStats(
      stats.promptTokens(),
      stats.contextTokens(),
      request.maxTokens() != null ? request.maxTokens() : 0
    );
  }

  @Override
  public Optional<EngineOutput<String, ConversationState>> processNextBatch() {
    if (!iterator.hasNext()) {
      return Optional.empty();
    }

    LlamaOutput output = iterator.next();
    return Optional.of(new EngineOutput<>(output.sequenceId(), output.text()));
  }

  @Override
  public void removeSequence(int internalId) {
    iterator.removeState(internalId);
  }

  @Override
  public Optional<String> getFinishReason(ConversationState state) {
    if (state == null) return Optional.empty();

    // Only report the finish reason when the model has truly stopped generating.
    // finishReason may be set as a marker (e.g. TOOL_CALL after the first
    // </tool_call>) while the model is still producing tokens for additional
    // tool calls. The `finished` flag is set by shouldContinue() when EOG
    // or LENGTH is hit.
    if (!state.isFinished()) return Optional.empty();

    FinishReason finishReason = state.getFinishReason();
    return finishReason == null ? Optional.empty() : Optional.of(mapFinishReason(finishReason));
  }

  @Override
  public TokenCountInfo getTokenCounts(ConversationState state) {
    if (state == null) {
      return new TokenCountInfo(0, 0, 0, 0);
    }
    return new TokenCountInfo(
      state.getInputTokens(),
      state.getAnswerTokens(),
      state.getReasoningTokens(),
      state.getToolsTokens()
    );
  }

  @Override
  public InferencePerformance buildPerformance(ConversationState state) {
    if (state == null) {
      return null;
    }
    var arena = state.getArena();
    var contextPerf = state.getContext().getPerformance(arena);
    var samplerPerf = state.getSampler().getPerformance(arena);

    // Direct primitive conversions - eliminates temporary Double object allocation
    // and reduces bytecode overhead vs Double.valueOf(x).longValue()
    return new InferencePerformance(
      (long) contextPerf.startTimeMs(),
      (long) contextPerf.loadTimeMs(),
      (long) contextPerf.promptEvalTimeMs(),
      (long) contextPerf.evalTimeMs(),
      contextPerf.promptTokensEvaluated(),
      contextPerf.tokensGenerated(),
      contextPerf.tokensReused(),
      (long) samplerPerf.samplingTimeMs(),
      samplerPerf.sampleCount()
    );
  }

  @Override
  public void cleanupSequenceState(ConversationState state) {
    if (state != null) {
      // Free native media bitmaps to prevent memory leaks and stale encoder state
      for (MtmdMedia media : state.getMedia()) {
        if (!media.isFree()) {
          media.free();
        }
      }
      state.getMedia().clear();

      // Free the sampler
      if (state.getSampler() != null) {
        state.getSampler().free();
      }
    }
  }

  @Override
  public void shutdown() {
    iterator.stop();
    iterator.free();
    model.close();
  }

  private String mapFinishReason(FinishReason finishReason) {
    if (finishReason == null) {
      return null;
    }
    return switch (finishReason) {
      case LENGTH -> "length";
      case TOOL_CALL -> "tool_calls";
      case STOP, EOS -> "stop";
    };
  }
}
