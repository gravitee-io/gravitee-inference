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

import io.gravitee.inference.api.memory.MemoryCheckPolicy;
import io.gravitee.llama.cpp.AttentionType;
import io.gravitee.llama.cpp.FlashAttentionType;
import io.gravitee.llama.cpp.LlamaLogLevel;
import io.gravitee.llama.cpp.PoolingType;
import io.gravitee.llama.cpp.SplitMode;
import java.nio.file.Path;
import java.util.List;

/**
 * Complete configuration for a llama.cpp model.
 *
 * <p>Prefer constructing instances via the fluent {@link Builder}:
 * <pre>{@code
 * ModelConfig cfg = ModelConfig.builder(Path.of("model.gguf"))
 *     .nGpuLayers(999)
 *     .poolingType(PoolingType.MEAN)
 *     .build();
 * }</pre>
 *
 * <p>The only mandatory field is {@code modelPath}; every other field has a
 * sensible default that matches standard llama.cpp behaviour.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public record ModelConfig(
  Path modelPath,
  int nCtx,
  int nBatch,
  int nUBatch,
  int nSeqMax,
  int nThreads,
  int nThreadsBatch,
  int nGpuLayers,
  boolean useMlock,
  boolean useMmap,
  SplitMode splitMode,
  int mainGpu,
  PoolingType poolingType,
  AttentionType attentionType,
  FlashAttentionType flashAttnType,
  boolean offloadKQV,
  boolean noPerf,
  LlamaLogLevel logLevel,
  Path loraPath,
  Path mmprojPath,
  List<String> rpcServers,
  MemoryCheckPolicy memoryCheckPolicy
) {
  /**
   * Returns true if this model configuration includes a multimodal projection file,
   * indicating the model supports vision and/or audio input.
   */
  public boolean isMultimodal() {
    return mmprojPath != null;
  }

  /**
   * Returns true if this model configuration includes RPC server endpoints
   * for distributed inference offloading.
   */
  public boolean hasRpcServers() {
    return rpcServers != null && !rpcServers.isEmpty();
  }

  /**
   * Returns a new {@link Builder} pre-set with all defaults.
   * Only {@code modelPath} must be supplied before calling {@link Builder#build()}.
   *
   * @param modelPath path to the GGUF model file
   */
  public static Builder builder(Path modelPath) {
    return new Builder(modelPath);
  }

  /**
   * Fluent builder for {@link ModelConfig}.
   *
   * <p>Default values:
   * <ul>
   *   <li>{@code nCtx} = 0 (defer to llama.cpp / model metadata)</li>
   *   <li>{@code nBatch} = 0 (defer to llama.cpp default)</li>
   *   <li>{@code nUBatch} = 0 (defer to llama.cpp default)</li>
   *   <li>{@code nSeqMax} = 0 (defer to llama.cpp default)</li>
   *   <li>{@code nThreads} / {@code nThreadsBatch} = {@code Runtime.availableProcessors()}</li>
   *   <li>{@code nGpuLayers} = 0 (CPU-only; set to 999 to offload everything to GPU)</li>
   *   <li>{@code useMlock} = false</li>
   *   <li>{@code useMmap} = true</li>
   *   <li>{@code splitMode} = {@link SplitMode#LAYER}</li>
   *   <li>{@code mainGpu} = 0</li>
   *   <li>{@code poolingType} = {@link PoolingType#UNSPECIFIED} (auto-detect in llama.cpp)</li>
   *   <li>{@code attentionType} = {@link AttentionType#UNSPECIFIED} (auto-detect)</li>
   *   <li>{@code flashAttnType} = {@link FlashAttentionType#AUTO}</li>
   *   <li>{@code offloadKQV} = true</li>
   *   <li>{@code noPerf} = false</li>
   *   <li>{@code logLevel} = {@link LlamaLogLevel#WARN}</li>
   *   <li>{@code loraPath} = null</li>
   *   <li>{@code mmprojPath} = null</li>
   *   <li>{@code rpcServers} = empty list</li>
   *   <li>{@code memoryCheckPolicy} = {@link MemoryCheckPolicy#WARN}</li>
   * </ul>
   */
  public static final class Builder {

    private final Path modelPath;
    private int nCtx = 0;
    private int nBatch = 0;
    private int nUBatch = 0;
    private int nSeqMax = 0;
    private int nThreads = Runtime.getRuntime().availableProcessors();
    private int nThreadsBatch = Runtime.getRuntime().availableProcessors();
    private int nGpuLayers = 0;
    private boolean useMlock = false;
    private boolean useMmap = true;
    private SplitMode splitMode = SplitMode.LAYER;
    private int mainGpu = 0;
    private PoolingType poolingType = PoolingType.UNSPECIFIED;
    private AttentionType attentionType = AttentionType.UNSPECIFIED;
    private FlashAttentionType flashAttnType = FlashAttentionType.AUTO;
    private boolean offloadKQV = true;
    private boolean noPerf = false;
    private LlamaLogLevel logLevel = LlamaLogLevel.WARN;
    private Path loraPath = null;
    private Path mmprojPath = null;
    private List<String> rpcServers = List.of();
    private MemoryCheckPolicy memoryCheckPolicy = MemoryCheckPolicy.WARN;

    private Builder(Path modelPath) {
      if (modelPath == null) throw new IllegalArgumentException("modelPath must not be null");
      this.modelPath = modelPath;
    }

    public Builder nCtx(int nCtx) {
      this.nCtx = nCtx;
      return this;
    }

    public Builder nBatch(int nBatch) {
      this.nBatch = nBatch;
      return this;
    }

    public Builder nUBatch(int nUBatch) {
      this.nUBatch = nUBatch;
      return this;
    }

    public Builder nSeqMax(int nSeqMax) {
      this.nSeqMax = nSeqMax;
      return this;
    }

    public Builder nThreads(int nThreads) {
      this.nThreads = nThreads;
      return this;
    }

    public Builder nThreadsBatch(int nThreadsBatch) {
      this.nThreadsBatch = nThreadsBatch;
      return this;
    }

    /** Number of model layers to offload to GPU. Use {@code 999} to offload all layers. */
    public Builder nGpuLayers(int nGpuLayers) {
      this.nGpuLayers = nGpuLayers;
      return this;
    }

    public Builder useMlock(boolean useMlock) {
      this.useMlock = useMlock;
      return this;
    }

    public Builder useMmap(boolean useMmap) {
      this.useMmap = useMmap;
      return this;
    }

    public Builder splitMode(SplitMode splitMode) {
      this.splitMode = splitMode;
      return this;
    }

    public Builder mainGpu(int mainGpu) {
      this.mainGpu = mainGpu;
      return this;
    }

    public Builder poolingType(PoolingType poolingType) {
      this.poolingType = poolingType;
      return this;
    }

    public Builder attentionType(AttentionType attentionType) {
      this.attentionType = attentionType;
      return this;
    }

    public Builder flashAttnType(FlashAttentionType flashAttnType) {
      this.flashAttnType = flashAttnType;
      return this;
    }

    public Builder offloadKQV(boolean offloadKQV) {
      this.offloadKQV = offloadKQV;
      return this;
    }

    public Builder noPerf(boolean noPerf) {
      this.noPerf = noPerf;
      return this;
    }

    public Builder logLevel(LlamaLogLevel logLevel) {
      this.logLevel = logLevel;
      return this;
    }

    public Builder loraPath(Path loraPath) {
      this.loraPath = loraPath;
      return this;
    }

    public Builder mmprojPath(Path mmprojPath) {
      this.mmprojPath = mmprojPath;
      return this;
    }

    public Builder rpcServers(List<String> rpcServers) {
      this.rpcServers = rpcServers != null ? rpcServers : List.of();
      return this;
    }

    public Builder memoryCheckPolicy(MemoryCheckPolicy memoryCheckPolicy) {
      this.memoryCheckPolicy = memoryCheckPolicy;
      return this;
    }

    public ModelConfig build() {
      return new ModelConfig(
        modelPath,
        nCtx,
        nBatch,
        nUBatch,
        nSeqMax,
        nThreads,
        nThreadsBatch,
        nGpuLayers,
        useMlock,
        useMmap,
        splitMode,
        mainGpu,
        poolingType,
        attentionType,
        flashAttnType,
        offloadKQV,
        noPerf,
        logLevel,
        loraPath,
        mmprojPath,
        rpcServers,
        memoryCheckPolicy
      );
    }
  }
}
