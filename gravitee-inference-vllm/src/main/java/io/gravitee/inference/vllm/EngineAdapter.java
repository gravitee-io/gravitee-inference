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
package io.gravitee.inference.vllm;

import io.gravitee.inference.api.memory.InsufficientVramException;
import io.gravitee.inference.api.memory.MemoryCheckPolicy;
import io.gravitee.inference.api.memory.MemoryEstimate;
import io.gravitee.inference.api.textgen.AudioContent;
import io.gravitee.inference.api.textgen.Content;
import io.gravitee.inference.api.textgen.ImageContent;
import io.gravitee.inference.api.textgen.InferencePerformance;
import io.gravitee.inference.api.textgen.PromptStats;
import io.gravitee.inference.api.textgen.Role;
import io.gravitee.vllm.engine.CompletionOutput;
import io.gravitee.vllm.engine.LoraRequest;
import io.gravitee.vllm.engine.MultiModalData;
import io.gravitee.vllm.engine.RequestOutput;
import io.gravitee.vllm.engine.SamplingParams;
import io.gravitee.vllm.engine.VllmEngine;
import io.gravitee.vllm.engine.VllmEngineBuilder;
import io.gravitee.vllm.iterator.VllmIterator;
import io.gravitee.vllm.iterator.VllmOutput;
import io.gravitee.vllm.runtime.PythonRuntime;
import io.gravitee.vllm.state.ConversationState;
import io.gravitee.vllm.template.ChatMessage;
import io.gravitee.vllm.template.ChatTemplate;
import io.gravitee.vllm.template.Tool;
import io.gravitee.vllm.template.ToolFunction;
import java.util.Base64;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicReference;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Engine adapter for vLLM backend.
 * Bridges the Gravitee {@link io.gravitee.inference.api.EngineAdapter} abstraction
 * with the vLLM4J {@link VllmEngine} and {@link VllmIterator}.
 *
 * <p>Unlike llama.cpp which uses a native batch iterator, vLLM manages its own
 * continuous batching via the Python engine. This adapter drives the VllmIterator
 * which calls {@code engine.step()} and extracts per-token deltas.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public class EngineAdapter
  implements io.gravitee.inference.api.EngineAdapter<VllmConfig, VllmRequest, String, EngineAdapter.VllmSequenceState> {

  private static final Logger LOGGER = LoggerFactory.getLogger(EngineAdapter.class);

  private final VllmEngine engine;
  private final VllmIterator iterator;
  private final ChatTemplate chatTemplate;

  /** Tracks per-sequence state keyed by internal ID. */
  private final Map<Integer, VllmSequenceState> states = new ConcurrentHashMap<>();

  /** Buffer for the latest output from the iterator. */
  private final AtomicReference<VllmOutput> currentOutput = new AtomicReference<>();

  public EngineAdapter(VllmConfig config) {
    VllmEngineBuilder builder = VllmEngine.builder().model(config.model()).dtype(config.dtype());

    if (config.maxModelLen() > 0) builder.maxModelLen(config.maxModelLen());
    if (config.maxNumSeqs() > 0) builder.maxNumSeqs(config.maxNumSeqs());
    if (config.gpuMemoryUtilization() > 0) builder.gpuMemoryUtilization(config.gpuMemoryUtilization());
    if (config.maxNumBatchedTokens() > 0) builder.maxNumBatchedTokens(config.maxNumBatchedTokens());
    if (config.enforceEager()) builder.enforceEager(true);
    if (config.trustRemoteCode()) builder.trustRemoteCode(true);
    if (config.quantization() != null) builder.quantization(config.quantization());
    if (config.swapSpace() > 0) builder.swapSpace(config.swapSpace());
    if (config.seed() != null) builder.seed(config.seed());
    if (config.enablePrefixCaching()) builder.enablePrefixCaching(true);
    builder.enableChunkedPrefill(config.enableChunkedPrefill());
    if (config.kvCacheDtype() != null) builder.kvCacheDtype(config.kvCacheDtype());
    if (config.enableLora()) {
      builder.enableLora(true);
      if (config.maxLoras() > 0) builder.maxLoras(config.maxLoras());
      if (config.maxLoraRank() > 0) builder.maxLoraRank(config.maxLoraRank());
    }
    if (config.venvPath() != null) builder.venvPath(config.venvPath());
    if (config.enableSleepMode() != null) builder.enableSleepMode(config.enableSleepMode());

    // Distributed inference — read from system properties set by systemd
    String tp = System.getProperty("vllm.tensor_parallel_size");
    if (tp != null) builder.tensorParallelSize(Integer.parseInt(tp));
    String pp = System.getProperty("vllm.pipeline_parallel_size");
    if (pp != null) builder.pipelineParallelSize(Integer.parseInt(pp));
    String executor = System.getProperty("vllm.distributed_executor_backend");
    if (executor != null) builder.distributedExecutorBackend(executor);

    if (config.hfToken() != null && !config.hfToken().isBlank()) {
      PythonRuntime.setEnv("HF_TOKEN", config.hfToken());
      PythonRuntime.setEnv("HUGGING_FACE_HUB_TOKEN", config.hfToken());
    }

    // Initialize CPython runtime before the memory check so that
    // GpuMemoryQuery can safely acquire the GIL via PyGILState_Ensure.
    // Without this, calling PyGILState_Ensure before Py_InitializeEx
    // dereferences a NULL PyThreadState and crashes with SIGSEGV.
    builder.initRuntime();

    runMemoryCheck(config);
    this.engine = builder.build();
    this.iterator = new VllmIterator(engine);
    this.chatTemplate = new ChatTemplate(engine);
  }

  private static void runMemoryCheck(VllmConfig config) {
    MemoryCheckPolicy policy = config.memoryCheckPolicy();
    if (policy == null || policy == MemoryCheckPolicy.DISABLED) {
      LOGGER.debug("Memory pre-flight check disabled for model {}", config.model());
      return;
    }
    MemoryEstimate estimate = VllmMemoryEstimator.estimate(
      config.totalParams(),
      config.bytesPerParam(),
      config.numHiddenLayers(),
      config.numKvHeads(),
      config.headDim(),
      resolveContextLength(config),
      config.maxNumSeqs(),
      config.gpuMemoryUtilization(),
      config.multimodal()
    );
    if (estimate.isUnknown()) {
      LOGGER.warn(
        "Memory pre-flight for model {}: could not query CUDA memory — skipping check. " +
          "Ensure CUDA is available and totalParams/bytesPerParam are provided in the configuration.",
        config.model()
      );
      return;
    }
    if (estimate.willFit()) {
      LOGGER.info("Memory pre-flight for model {}: {}", config.model(), estimate.toHumanReadable());
      return;
    }
    if (policy == MemoryCheckPolicy.FAIL) {
      throw new InsufficientVramException(config.model(), estimate);
    }
    LOGGER.warn("Memory pre-flight for model {}: {}", config.model(), estimate.toHumanReadable());
  }

  /**
   * Resolves the context length to use for KV-cache sizing in the pre-flight
   * VRAM estimate.
   *
   * <p>Priority:
   * <ol>
   *   <li>User-configured {@code maxModelLen} (explicit override).</li>
   *   <li>{@code max_position_embeddings} from the model's {@code config.json}
   *       — the maximum sequence length the model's positional encoding supports.
   *       This is what vLLM uses as default context length when no override is
   *       provided.</li>
   *   <li>Fallback to {@code 4096} if neither is available.</li>
   * </ol>
   */
  private static int resolveContextLength(VllmConfig config) {
    if (config.maxModelLen() > 0) {
      return config.maxModelLen();
    }
    if (config.maxPositionEmbeddings() > 0) {
      return config.maxPositionEmbeddings();
    }
    return 4096;
  }

  @Override
  public VllmSequenceState createSequenceState(int internalId, VllmRequest request) throws Exception {
    // Render prompt using chat template if messages are present
    String prompt = request.prompt();
    MultiModalData multiModalData = null;

    if (request.hasMessages() && request.messages() != null) {
      // Collect multimodal data (images, audio) from all messages
      multiModalData = extractMultiModalData(request.messages());

      List<ChatMessage> vllmMessages = request
        .messages()
        .stream()
        .map(msg -> {
          String content = msg.content();
          // For multimodal messages, build content parts list for the Jinja2 template
          if (msg.hasMedia()) {
            List<Map<String, Object>> contentParts = buildContentParts(msg);
            return ChatMessage.userWithParts(content, contentParts);
          }
          return switch (msg.role()) {
            case SYSTEM -> ChatMessage.system(content);
            case ASSISTANT -> ChatMessage.assistant(content);
            default -> ChatMessage.user(content);
          };
        })
        .toList();

      // Convert tools from OpenAI format (Map) to vLLM4J Tool DSL
      List<Tool> vllmTools = convertTools(request.tools());
      if (vllmTools != null && !vllmTools.isEmpty()) {
        prompt = chatTemplate.render(vllmMessages, vllmTools, true);
      } else {
        prompt = chatTemplate.render(vllmMessages, true);
      }
    }

    if (prompt == null || prompt.isBlank()) {
      LOGGER.error("Cannot create sequence state: prompt is empty for internalId {}", internalId);
      return null;
    }

    // Build SamplingParams — the engine's arena outlives every request
    SamplingParams sp = new SamplingParams(engine.arena());
    if (request.temperature() != null) sp.temperature(request.temperature());
    if (request.maxTokens() != null) sp.maxTokens(request.maxTokens());
    if (request.topP() != null) sp.topP(request.topP());
    if (request.presencePenalty() != null) sp.presencePenalty(request.presencePenalty());
    if (request.frequencyPenalty() != null) sp.frequencyPenalty(request.frequencyPenalty());
    if (request.seed() != null) sp.seed(request.seed().longValue());
    if (request.stop() != null && !request.stop().isEmpty()) sp.stop(request.stop());

    // Always create a ConversationState so token counts (prompt, answer,
    // reasoning, tools) are tracked even without reasoning/tool tags.
    ConversationState conversationState = new ConversationState();
    if (request.reasoningTags() != null && request.reasoningTags().isConfigured()) {
      conversationState.reasoning(request.reasoningTags().openToken(), request.reasoningTags().closeToken());
    }
    if (request.toolTags() != null && request.toolTags().isConfigured()) {
      conversationState.toolCall(request.toolTags().openToken(), request.toolTags().closeToken());
    }

    String requestId = "seq-" + internalId;

    // Build optional LoRA request
    LoraRequest loraReq = null;
    if (request.hasLora()) {
      loraReq = new LoraRequest(
        request.loraName() != null ? request.loraName() : "lora-" + internalId,
        internalId + 1, // loraIntId must be >= 1
        request.loraPath()
      );
    }

    // Build the vLLM4J request with full constructor (supports multimodal + LoRA)
    var vllmRequest = new io.gravitee.vllm.engine.VllmRequest(requestId, prompt, sp, multiModalData, 0, loraReq);

    // Submit to iterator with conversation state for token tracking
    iterator.addRequest(vllmRequest, conversationState);

    VllmSequenceState state = new VllmSequenceState(requestId, sp, conversationState, System.currentTimeMillis());
    states.put(internalId, state);
    return state;
  }

  @Override
  public PromptStats validateRequest(VllmRequest request) {
    // vLLM handles prompt validation internally via the Python engine.
    // We provide a permissive estimate here. The engine will reject
    // requests that exceed the model's context length.
    int estimatedPromptTokens = 0;
    if (request.prompt() != null) {
      estimatedPromptTokens = request.prompt().length() / 4; // rough estimate
    }
    int maxTokens = request.maxTokens() != null ? request.maxTokens() : 0;
    // Use a large context window estimate since vLLM manages this internally
    return new PromptStats(estimatedPromptTokens, Integer.MAX_VALUE, maxTokens);
  }

  @Override
  public Optional<EngineOutput<String, VllmSequenceState>> processNextBatch() throws Exception {
    if (!iterator.hasNext()) {
      return Optional.empty();
    }

    VllmOutput output = iterator.next();
    currentOutput.set(output);

    // Find the internal ID for this request
    for (var entry : states.entrySet()) {
      if (entry.getValue().requestId.equals(output.requestId())) {
        VllmSequenceState state = entry.getValue();
        state.lastOutput = output;
        state.totalTokensGenerated++;
        if (output.finished()) {
          state.finishReason = output.finishReason();
          state.finishedTimeMs = System.currentTimeMillis();
        }
        return Optional.of(new EngineOutput<>(entry.getKey(), output.delta()));
      }
    }

    return Optional.empty();
  }

  @Override
  public void removeSequence(int internalId) {
    VllmSequenceState state = states.get(internalId);
    if (state != null) {
      try {
        iterator.abortRequest(state.requestId);
      } catch (Exception e) {
        LOGGER.debug("Error aborting request {}: {}", state.requestId, e.getMessage());
      }
    }
  }

  @Override
  public Optional<String> getFinishReason(VllmSequenceState state) {
    if (state == null) return Optional.empty();
    if (state.finishReason == null) return Optional.empty();

    // Prefer the Java-side ConversationState finish reason when it detected
    // tool calls. Python vLLM has no concept of <tool_call> tags and reports
    // "stop", but the Java FSM correctly identified TOOL_CALL boundaries.
    if (state.conversationState != null && state.conversationState.finishReason() != null) {
      return Optional.of(state.conversationState.finishReason().label());
    }
    return Optional.of(state.finishReason);
  }

  @Override
  public TokenCountInfo getTokenCounts(VllmSequenceState state) {
    if (state == null) {
      return new TokenCountInfo(0, 0, 0, 0);
    }
    if (state.conversationState != null) {
      return new TokenCountInfo(
        state.conversationState.inputTokens(),
        state.conversationState.answerTokens(),
        state.conversationState.reasoningTokens(),
        state.conversationState.toolsTokens()
      );
    }
    return new TokenCountInfo(0, state.totalTokensGenerated, 0, 0);
  }

  @Override
  public InferencePerformance buildPerformance(VllmSequenceState state) {
    if (state == null) {
      return null;
    }
    long startTime = state.startTimeMs;
    long totalTime = (state.finishedTimeMs > 0 ? state.finishedTimeMs : System.currentTimeMillis()) - startTime;
    return new InferencePerformance(
      startTime,
      0,
      0,
      totalTime,
      0,
      state.totalTokensGenerated,
      0,
      0,
      state.totalTokensGenerated
    );
  }

  @Override
  public void cleanupSequenceState(VllmSequenceState state) {
    if (state != null && state.samplingParams != null) {
      try {
        state.samplingParams.close();
      } catch (Exception e) {
        LOGGER.debug("Error closing sampling params: {}", e.getMessage());
      }
    }
    // Note: we do NOT call engine.freeCache() here. vLLM manages its own
    // KV cache internally — calling torch.cuda.synchronize() + empty_cache()
    // after every request destroys pipeline overlap, forces expensive
    // cudaMalloc round-trips, and can race with vLLM's background engine_core
    // loop that manages block allocation/deallocation asynchronously.
  }

  /**
   * Performs aggressive memory maintenance suitable for periodic scheduling.
   *
   * <p>Call this method periodically (e.g., every 60-300 seconds) to trigger
   * heavier-weight cleanup including multiple garbage collection passes.
   * Useful when operating the gateway in low-memory environments or when
   * gradual memory growth is observed despite per-sequence flushing.
   *
   * <p>Does NOT restart the engine or release model weights, only cleans up
   * temporary allocations and breaks circular references in the Python runtime.
   *
   * <p>Best-effort — silently ignores errors.
   */
  public void performMemoryMaintenance() {
    try {
      engine.reset();
      LOGGER.debug("Performed aggressive GPU memory maintenance");
    } catch (Exception e) {
      LOGGER.warn("Error during GPU memory maintenance: {}", e.getMessage());
    }
  }

  @Override
  public void shutdown() {
    try {
      iterator.stop();
    } catch (Exception e) {
      LOGGER.debug("Error stopping iterator: {}", e.getMessage());
    }
    try {
      engine.close();
    } catch (Exception e) {
      LOGGER.debug("Error closing engine: {}", e.getMessage());
    }
  }

  /**
   * Builds OpenAI-format content parts list for a multimodal message.
   *
   * <p>Converts the Gravitee API {@link io.gravitee.inference.api.textgen.ChatMessage}
   * into the format that VLM Jinja2 templates expect:
   * <pre>{@code
   * [{"type": "text", "text": "Describe this image"},
   *  {"type": "image"}]
   * }</pre>
   *
   * <p>The image/audio binary data is passed separately via {@link MultiModalData},
   * not embedded in the content parts. The template just needs to know that an
   * image is present (via {@code {"type": "image"}}) to insert placeholder tokens.
   *
   * @param msg a message with media content
   * @return list of content part maps in OpenAI format
   */
  private static List<Map<String, Object>> buildContentParts(io.gravitee.inference.api.textgen.ChatMessage msg) {
    List<Map<String, Object>> parts = new java.util.ArrayList<>();

    // Add text part if present
    if (msg.hasText()) {
      parts.add(Map.of("type", "text", "text", msg.content()));
    }

    // Add media placeholders — the actual bytes are in MultiModalData
    for (Content content : msg.media()) {
      if (content instanceof ImageContent) {
        parts.add(Map.of("type", "image"));
      } else if (content instanceof AudioContent) {
        parts.add(Map.of("type", "audio"));
      }
    }

    return parts;
  }

  /**
   * Extracts multimodal data (images, audio) from chat messages.
   *
   * <p>Iterates over all messages, collecting any {@link ImageContent} or
   * {@link AudioContent} media items. The base64-encoded data is decoded
   * to raw bytes and added to a {@link MultiModalData} object.
   *
   * <p>For VLMs (e.g. Qwen2.5-VL, LLaVA), the chat template handles
   * inserting the appropriate placeholder tokens ({@code <image>}, etc.)
   * into the rendered prompt. This method only handles the binary data.
   *
   * @param messages the parsed chat messages with optional media
   * @return a populated {@link MultiModalData}, or {@code null} if no media found
   */
  private static MultiModalData extractMultiModalData(List<io.gravitee.inference.api.textgen.ChatMessage> messages) {
    MultiModalData mmData = null;

    for (var msg : messages) {
      if (!msg.hasMedia()) continue;

      for (Content content : msg.media()) {
        if (content instanceof ImageContent img) {
          try {
            byte[] imageBytes = Base64.getDecoder().decode(img.data());
            if (mmData == null) mmData = new MultiModalData();
            mmData.addImage(imageBytes);
          } catch (IllegalArgumentException e) {
            LOGGER.warn("Failed to decode base64 image data: {}", e.getMessage());
          }
        } else if (content instanceof AudioContent audio) {
          try {
            byte[] audioBytes = Base64.getDecoder().decode(audio.data());
            if (mmData == null) mmData = new MultiModalData();
            mmData.addAudio(audioBytes);
          } catch (IllegalArgumentException e) {
            LOGGER.warn("Failed to decode base64 audio data: {}", e.getMessage());
          }
        }
      }
    }

    return mmData;
  }

  /**
   * Converts OpenAI-format tools (List of Maps) to vLLM4J {@link Tool} DSL objects.
   *
   * <p>Each tool map follows the OpenAI format:
   * <pre>{@code
   * {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
   * }</pre>
   *
   * @param tools the raw OpenAI tools from the request payload, or null
   * @return list of vLLM4J Tool objects, or null if input is null/empty
   */
  @SuppressWarnings("unchecked")
  private static List<Tool> convertTools(List<Map<String, Object>> tools) {
    if (tools == null || tools.isEmpty()) {
      return null;
    }
    List<Tool> result = new java.util.ArrayList<>();
    for (Map<String, Object> toolMap : tools) {
      Object functionObj = toolMap.get("function");
      if (!(functionObj instanceof Map<?, ?> functionMap)) {
        continue;
      }
      String name = functionMap.get("name") instanceof String s ? s : null;
      String description = functionMap.get("description") instanceof String s ? s : null;
      Map<String, Object> parameters = functionMap.get("parameters") instanceof Map<?, ?> p ? (Map<String, Object>) p : null;
      if (name != null) {
        result.add(Tool.function(name, description != null ? description : "", parameters));
      }
    }
    return result.isEmpty() ? null : result;
  }

  /**
   * Per-sequence state for vLLM.
   * Tracks the request ID, sampling params, conversation state for token classification,
   * and timing information for performance metrics.
   */
  public static class VllmSequenceState {

    final String requestId;
    final SamplingParams samplingParams;
    final ConversationState conversationState;
    final long startTimeMs;
    VllmOutput lastOutput;
    String finishReason;
    long finishedTimeMs;
    int totalTokensGenerated;

    VllmSequenceState(
      String requestId,
      SamplingParams samplingParams,
      ConversationState conversationState,
      long startTimeMs
    ) {
      this.requestId = requestId;
      this.samplingParams = samplingParams;
      this.conversationState = conversationState;
      this.startTimeMs = startTimeMs;
    }
  }
}
