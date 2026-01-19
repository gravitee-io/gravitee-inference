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

/**
 * Represents a single token emitted by an inference engine.
 * Designed to be engine-agnostic while providing all necessary metadata.
 *
 * @param <T> The token type (e.g., String for text tokens)
 * @param seqId The external sequence identifier
 * @param token The token content
 * @param index The token index in the sequence
 * @param isFinal Whether this is the final token for the sequence
 * @param finishReason The reason the sequence finished (if final)
 * @param promptTokens Number of prompt tokens processed
 * @param completionTokens Number of completion tokens generated
 * @param reasoningTokens Number of reasoning tokens (if supported)
 * @param toolTokens Number of tool call tokens (if supported)
 * @param performance Optional performance metrics
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public record InferenceToken<T>(
  int seqId,
  T token,
  int index,
  boolean isFinal,
  String finishReason,
  int promptTokens,
  int completionTokens,
  int reasoningTokens,
  int toolTokens,
  InferencePerformance performance
) {
  public InferenceToken {
    if (seqId < 0) {
      throw new IllegalArgumentException("seqId must be non-negative");
    }
    if (index < 0) {
      throw new IllegalArgumentException("index must be non-negative");
    }
    if (isFinal && finishReason == null) {
      throw new IllegalArgumentException("finishReason is required when isFinal is true");
    }
  }
}
