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
}
