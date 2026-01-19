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

import io.gravitee.llama.cpp.*;
import io.gravitee.llama.cpp.nativelib.LlamaLibLoader;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.Objects;
import javax.sound.sampled.UnsupportedAudioFileException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class Model implements AutoCloseable {

  private static final Logger LOGGER = LoggerFactory.getLogger(Model.class);

  static {
    LlamaBackend.init();
  }

  private static void loadAllBackends() {
    String location = LlamaLibLoader.load();
    try (Arena a = Arena.ofConfined()) {
      LlamaRuntime.ggml_backend_load_all_from_path(a, location);
    }
  }

  private final Arena arena;
  private final LlamaModel model;
  private final LlamaContext context;
  private final LlamaVocab vocab;
  private final LlamaTokenizer tokenizer;
  private final LlamaLogger logger;
  private final MtmdContext mtmdContext;

  public Model(ModelConfig config) {
    this.arena = Arena.ofAuto();

    this.logger = config.logLevel() == null ? null : new LlamaLogger(arena);
    if (logger != null) {
      logger.setLogging(config.logLevel());
      if (config.logLevel() == LlamaLogLevel.NONE) {
        logger.setLogging(config.logLevel(), s -> {});
      } else {
        logger.setLogging(config.logLevel());
      }
    }

    // When using RPC, skip ggml_backend_load_all_from_path — loading all backends
    // (CPU, Metal, RPC plugin) from the library directory can interfere with
    // explicit RPC server registration via ggml_backend_rpc_add_server.
    // This matches the behavior of Main.java in llamaj.cpp.
    if (!config.hasRpcServers()) {
      loadAllBackends();
    }

    var modelParams = new LlamaModelParams(arena)
      .useMlock(config.useMlock())
      .useMmap(config.useMmap())
      .nGpuLayers(config.nGpuLayers());

    if (config.splitMode() != null) {
      modelParams.splitMode(config.splitMode());
    }
    if (config.mainGpu() >= 0) {
      modelParams.mainGpu(config.mainGpu());
    }

    // Register RPC servers for distributed inference offloading
    if (config.hasRpcServers()) {
      LOGGER.info("Registering {} RPC server(s): {}", config.rpcServers().size(), config.rpcServers());
      modelParams.rpcServers(arena, config.rpcServers().toArray(String[]::new));
      var rpcDevices = BackendRegistry.getRpcDeviceHandles();
      LOGGER.info("RPC device handles obtained: {}", rpcDevices.size());
      if (!rpcDevices.isEmpty()) {
        modelParams.devices(arena, rpcDevices);
      }
    }

    this.model = new LlamaModel(arena, config.modelPath(), modelParams);
    if (config.loraPath() != null) {
      this.model.initLoraAdapter(arena, config.loraPath());
    }

    var contextParams = new LlamaContextParams(arena)
      .nCtx(config.nCtx())
      .nBatch(config.nBatch())
      .nUBatch(config.nUBatch())
      .nSeqMax(config.nSeqMax())
      .nThreads(config.nThreads())
      .nThreadsBatch(config.nThreadsBatch());

    if (config.poolingType() != null) {
      contextParams.poolingType(config.poolingType());
    }
    if (config.attentionType() != null) {
      contextParams.attentionType(config.attentionType());
    }
    if (config.flashAttnType() != null) {
      contextParams.flashAttnType(config.flashAttnType());
    }
    contextParams.offloadKQV(config.offloadKQV());
    contextParams.noPerf(config.noPerf());

    this.context = new LlamaContext(arena, model, contextParams);
    this.vocab = new LlamaVocab(model);
    this.tokenizer = new LlamaTokenizer(vocab, context);

    // Initialize multimodal context if mmproj file is provided
    if (config.mmprojPath() != null) {
      var mtmdParams = new MtmdContextParams(arena)
        .useGpu(config.nGpuLayers() > 0)
        .nThreads(config.nThreads())
        .printTimings(false)
        .mediaMarker("<__media__>");
      if (config.flashAttnType() != null) {
        mtmdParams.flashAttnType(config.flashAttnType());
      }
      this.mtmdContext = new MtmdContext(arena, this.model, config.mmprojPath(), mtmdParams);
    } else {
      this.mtmdContext = null;
    }
  }

  public BatchIterator newBatchIterator() {
    return new BatchIterator(arena, context, mtmdContext);
  }

  /**
   * Returns true if this model has a multimodal projection loaded,
   * meaning it can process image and/or audio inputs.
   */
  public boolean isMultimodal() {
    return mtmdContext != null;
  }

  /**
   * Returns the MtmdContext for multimodal operations, or null if not multimodal.
   */
  public MtmdContext getMtmdContext() {
    return mtmdContext;
  }

  public ConversationState newConversation(int seqId, Request request) {
    Objects.requireNonNull(request, "request is required");
    String prompt = promptFor(request);
    PromptStats stats = promptStats(prompt);
    var sampler = samplerFor(request);
    int promptTokens = stats.promptTokens();
    int contextTokens = stats.contextTokens();
    if (promptTokens >= contextTokens) {
      throw new LlamaException("Prompt tokens (" + promptTokens + ") exceed or match context size (" + contextTokens + ").");
    }
    int availableForCompletion = stats.availableForCompletion();
    int maxTokens = request.maxTokens() != null ? request.maxTokens() : availableForCompletion;
    if (maxTokens > availableForCompletion) {
      maxTokens = availableForCompletion;
    }

    var state = ConversationState.create(arena, context, tokenizer, sampler, seqId);
    state.setMaxTokens(maxTokens);

    if (request.stop() != null && !request.stop().isEmpty()) {
      state.setStopStrings(request.stop());
    }

    if (request.reasoningTags() != null && request.reasoningTags().isConfigured()) {
      state.setReasoning(request.reasoningTags().openToken(), request.reasoningTags().closeToken());
    }
    if (request.toolTags() != null && request.toolTags().isConfigured()) {
      state.setToolCall(request.toolTags().openToken(), request.toolTags().closeToken());
    }

    state.initialize(prompt);

    // Set media on the conversation state if the model is multimodal and request has media
    if (mtmdContext != null && request.hasMessages()) {
      MediaInfo mediaInfo = processMediaContent(request.messages());
      if (!mediaInfo.media().isEmpty()) {
        state.setMedia(mediaInfo.media());
      }
    }

    return state;
  }

  public LlamaSampler samplerFor(Request request) {
    float temperature = request.temperature() != null ? request.temperature() : 0.7f;
    float topP = request.topP() != null ? request.topP() : 0.9f;
    float presencePenalty = request.presencePenalty() != null ? request.presencePenalty() : 0.0f;
    float frequencyPenalty = request.frequencyPenalty() != null ? request.frequencyPenalty() : 0.0f;
    int seed = request.seed() != null ? request.seed() : 42;

    var sampler = new LlamaSampler(arena);
    if (temperature <= 0f) {
      return sampler.greedy().seed(seed);
    }
    return sampler
      .temperature(temperature)
      .topP(topP, 64)
      .penalties(context.nCtx(), 1.0f, frequencyPenalty, presencePenalty)
      .seed(seed);
  }

  public String promptFor(Request request) {
    if (request.hasMessages()) {
      return buildChatPrompt(request.messages());
    }
    return Objects.requireNonNullElse(request.prompt(), "");
  }

  public PromptStats promptStats(Request request) {
    Objects.requireNonNull(request, "request is required");
    return promptStats(promptFor(request));
  }

  private PromptStats promptStats(String prompt) {
    int promptTokens = countPromptTokens(prompt);
    int contextTokens = context.nCtx();
    return new PromptStats(promptTokens, contextTokens);
  }

  private String buildChatPrompt(List<io.gravitee.inference.api.textgen.ChatMessage> messages) {
    try (Arena promptArena = Arena.ofConfined()) {
      List<LlamaChatMessage> llamaMessages = messages
        .stream()
        .map(message -> {
          String content = message.content() != null ? message.content() : "";
          if (mtmdContext != null && message.hasMedia()) {
            // Use unified media processor to get both markers and validate media
            MediaInfo mediaInfo = processMediaContent(List.of(message));
            if (!mediaInfo.media().isEmpty()) {
              content = mediaInfo.promptSuffix() + content;
            }
          }
          return new LlamaChatMessage(promptArena, toRole(message.role()), content);
        })
        .toList();
      return new LlamaTemplate(model).applyTemplate(
        promptArena,
        new LlamaChatMessages(promptArena, llamaMessages),
        context.nCtx()
      );
    }
  }

  /**
   * Builds a list of MtmdMedia from chat messages containing image/audio content.
   * Media is extracted in message order, matching the media markers inserted in buildChatPrompt.
   *
   * <p>Uses {@link Base64#getMimeDecoder()} instead of {@link Base64#getDecoder()} to tolerate
   * whitespace and line breaks in base64 data, matching the permissive behavior of the reference
   * llama.cpp server's custom base64 decoder.</p>
   *
   * <p><strong>Resource Management:</strong> This method acquires native memory resources via
   * {@link MtmdImage#fromBytesNative} and {@link MtmdAudio#fromBytes}. On exception, all
   * successfully created media objects are automatically freed to prevent memory leaks.</p>
   *
   * @deprecated Use {@link #processMediaContent(List)} instead - provides unified processing
   *             with better performance and consistent error handling.
   */
  @Deprecated(since = "1.0", forRemoval = false)
  private List<MtmdMedia> buildMedia(List<io.gravitee.inference.api.textgen.ChatMessage> messages) {
    return processMediaContent(messages).media();
  }

  /**
   * Safely frees all media resources in the list.
   * Continues cleanup even if individual free() calls fail.
   *
   * @param mediaList the list of media resources to free
   */
  private void cleanupMedia(List<MtmdMedia> mediaList) {
    for (MtmdMedia media : mediaList) {
      try {
        if (media != null && !media.isFree()) {
          media.free();
        }
      } catch (Exception cleanupException) {
        // Log but continue cleanup to prevent cascading failures
        // Note: Logger is not available in Model class, so this is silent for now
        // Production code should inject a logger or use System.err as fallback
      }
    }
  }

  /**
   * Processes media content from chat messages in a single pass.
   * This replaces the previous dual-loop pattern where media was checked twice
   * (once for prompt markers, once for building media list).
   *
   * <p>Returns both the prompt suffix with media markers and the processed media objects.
   * On exception, all successfully created media objects are automatically freed.</p>
   *
   * @param messages the chat messages containing media content
   * @return MediaInfo with prompt suffix and media list
   */
  private MediaInfo processMediaContent(List<io.gravitee.inference.api.textgen.ChatMessage> messages) {
    StringBuilder promptBuilder = new StringBuilder();
    List<MtmdMedia> mediaList = new ArrayList<>();

    try {
      for (var message : messages) {
        if (!message.hasMedia()) continue;

        // Single loop with unified type checking - eliminates redundant instanceof calls
        for (var content : message.media()) {
          try {
            if (content instanceof io.gravitee.inference.api.textgen.ImageContent img) {
              promptBuilder.append("<__media__>\n");
              byte[] imageBytes = Base64.getMimeDecoder().decode(img.data());
              mediaList.add(MtmdImage.fromBytesNative(arena, mtmdContext, imageBytes));
            } else if (content instanceof io.gravitee.inference.api.textgen.AudioContent audio) {
              promptBuilder.append("<__media__>\n");
              byte[] audioBytes = Base64.getMimeDecoder().decode(audio.data());
              int sampleRate = mtmdContext.getAudioSampleRate();
              mediaList.add(MtmdAudio.fromBytes(arena, audioBytes, sampleRate > 0 ? sampleRate : 16000));
            }
          } catch (IOException | UnsupportedAudioFileException e) {
            throw new LlamaException("Failed to load media: " + e.getMessage());
          }
        }
      }
    } catch (Exception e) {
      // Cleanup on failure
      cleanupMedia(mediaList);
      throw e;
    }

    return new MediaInfo(promptBuilder.toString(), mediaList);
  }

  private int countPromptTokens(String prompt) {
    try (Arena promptArena = Arena.ofConfined()) {
      return tokenizer.tokenize(promptArena, prompt).size();
    }
  }

  public record PromptStats(int promptTokens, int contextTokens) {
    public int availableForCompletion() {
      return Math.max(0, contextTokens - promptTokens);
    }
  }

  /**
   * Encapsulates the result of media content processing.
   * Contains both the prompt suffix with media markers and the processed media list.
   */
  private record MediaInfo(String promptSuffix, List<MtmdMedia> media) {}

  private Role toRole(io.gravitee.inference.api.textgen.Role role) {
    return switch (role) {
      case ASSISTANT -> Role.ASSISTANT;
      case SYSTEM -> Role.SYSTEM;
      case USER -> Role.USER;
    };
  }

  @Override
  public void close() {
    if (mtmdContext != null) {
      mtmdContext.free();
    }
    context.free();
    model.free();
    if (arena.scope().isAlive()) {
      arena.close();
    }
  }
}
