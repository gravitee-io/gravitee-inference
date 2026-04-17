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

import io.gravitee.inference.api.textgen.AbstractBatchEngine;
import io.gravitee.inference.api.textgen.BatchEngineConfig;
import io.gravitee.inference.api.textgen.InferenceToken;
import java.util.function.Consumer;

/**
 * Batch inference engine for llama.cpp models.
 * Delegates all complex orchestration to AbstractBatchEngine,
 * focusing only on llama.cpp-specific configuration.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public class BatchEngine extends AbstractBatchEngine<ModelConfig, Request, String, io.gravitee.llama.cpp.ConversationState> {

  private final EngineAdapter engineAdapter;

  /**
   * Creates a new llama.cpp batch engine with default configuration.
   *
   * @param config The model configuration
   */
  public BatchEngine(ModelConfig config) {
    this(BatchEngineConfig.of(config.nSeqMax()), config);
  }

  /**
   * Creates a new llama.cpp batch engine with custom engine configuration.
   *
   * @param engineConfig The engine configuration (slots, queue capacity, etc.)
   * @param modelConfig The model configuration
   */
  public BatchEngine(BatchEngineConfig engineConfig, ModelConfig modelConfig) {
    this(engineConfig, new EngineAdapter(modelConfig));
  }

  private BatchEngine(BatchEngineConfig engineConfig, EngineAdapter adapter) {
    super(engineConfig, adapter);
    this.engineAdapter = adapter;
  }

  /** Returns the raw chat template string from the GGUF model. */
  public String chatTemplateString() {
    return engineAdapter.model().chatTemplateString();
  }

  public String bosToken() {
    return engineAdapter.model().bosToken();
  }

  public String eosToken() {
    return engineAdapter.model().eosToken();
  }

  /**
   * Starts the engine and begins processing sequences.
   *
   * @param tokenConsumer Callback for receiving generated tokens
   */
  @Override
  public void start(Consumer<InferenceToken<String>> tokenConsumer) {
    super.start(tokenConsumer);
  }
}
