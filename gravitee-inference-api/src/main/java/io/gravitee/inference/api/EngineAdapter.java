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
package io.gravitee.inference.api;

import io.gravitee.inference.api.textgen.InferencePerformance;
import io.gravitee.inference.api.textgen.PromptStats;
import java.util.Optional;

/**
 * Adapter interface for engine-specific operations.
 * Implementations handle the actual interaction with inference backends.
 *
 * <p>This interface uses the Template Method pattern - the abstract batch engine
 * handles all sequence management, queuing, thread safety, and token emission,
 * while implementations focus only on engine-specific logic.</p>
 *
 * @param <CONFIG> Engine configuration type
 * @param <REQUEST> Generation request type
 * @param <TOKEN> Token type
 * @param <STATE> Engine-specific sequence state type
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public interface EngineAdapter<CONFIG, REQUEST, TOKEN, STATE> {
  /**
   * Creates a new sequence state for the given internal ID and request.
   * This is called when a sequence is ready to start processing.
   *
   * @param internalId The internal sequence ID (slot index)
   * @param request The generation request
   * @return A new sequence state, or null if the request is invalid
   * @throws Exception if the state cannot be created
   */
  STATE createSequenceState(int internalId, REQUEST request) throws Exception;

  /**
   * Validates a generation request and calculates statistics.
   * Called before queuing to ensure the request is valid.
   *
   * @param request The generation request
   * @return Statistics about the prompt
   */
  PromptStats validateRequest(REQUEST request);

  /**
   * Processes the next batch of tokens for all active sequences.
   * This is called repeatedly by the worker thread until sequences complete.
   *
   * @return An optional output, or empty if no sequences are active
   * @throws Exception if processing fails
   */
  Optional<EngineOutput<TOKEN, STATE>> processNextBatch() throws Exception;

  /**
   * Removes a sequence from the batch processor.
   * Called when a sequence completes or is cancelled.
   *
   * @param internalId The internal sequence ID
   */
  void removeSequence(int internalId);

  /**
   * Checks if a sequence has finished.
   *
   * @param state The sequence state
   * @return Optional finish reason if finished, empty otherwise
   */
  Optional<String> getFinishReason(STATE state);

  /**
   * Gets token counts from a sequence state.
   *
   * @param state The sequence state
   * @return Token count information
   */
  TokenCountInfo getTokenCounts(STATE state);

  /**
   * Builds performance metrics for a completed sequence.
   *
   * @param state The sequence state
   * @return Performance metrics, or null if not available
   */
  InferencePerformance buildPerformance(STATE state);

  /**
   * Releases resources associated with a sequence state.
   *
   * @param state The sequence state to cleanup
   */
  void cleanupSequenceState(STATE state);

  /**
   * Stops the batch processor and releases all resources.
   * Called during engine shutdown.
   */
  void shutdown();

  /**
   * Represents the output from processing a batch.
   *
   * @param sequenceId The internal sequence ID that produced this output
   * @param token The generated token
   */
  record EngineOutput<TOKEN, STATE>(int sequenceId, TOKEN token) {}

  /**
   * Token count information for a sequence.
   *
   * @param inputTokens Number of input/prompt tokens
   * @param outputTokens Number of output/generated tokens
   * @param reasoningTokens Number of reasoning tokens (0 if not supported)
   * @param toolTokens Number of tool call tokens (0 if not supported)
   */
  record TokenCountInfo(int inputTokens, int outputTokens, int reasoningTokens, int toolTokens) {}
}
