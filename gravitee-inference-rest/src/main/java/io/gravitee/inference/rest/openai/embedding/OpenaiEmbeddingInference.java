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
package io.gravitee.inference.rest.openai.embedding;

import io.gravitee.inference.api.embedding.EmbeddingTokenCount;
import io.gravitee.inference.rest.openai.GraviteeInferenceOpenaiException;
import io.gravitee.inference.rest.openai.OpenaiRestInference;
import io.gravitee.inference.rest.openai.embedding.model.EmbeddingRequest;
import io.gravitee.inference.rest.openai.embedding.model.EmbeddingResponse;
import io.reactivex.rxjava3.core.Maybe;
import io.reactivex.rxjava3.core.Single;
import io.vertx.core.json.Json;
import io.vertx.rxjava3.core.Vertx;
import io.vertx.rxjava3.core.buffer.Buffer;
import io.vertx.rxjava3.ext.web.client.HttpRequest;
import io.vertx.rxjava3.ext.web.client.HttpResponse;
import java.util.Objects;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class OpenaiEmbeddingInference extends OpenaiRestInference<OpenAIEmbeddingConfig, EmbeddingTokenCount> {

  static final String EMBEDDINGS_ENDPOINT = "/embeddings";
  static final String MEDIA_TYPE = "application/json";
  static final String CONTENT_TYPE = "Content-Type";
  static final String OPEN_AI_ORGANIZATION = "OpenAI-Organization";
  static final String OPEN_AI_PROJECT = "OpenAI-Project";
  private static final Logger LOGGER = LoggerFactory.getLogger(OpenaiEmbeddingInference.class);

  public OpenaiEmbeddingInference(OpenAIEmbeddingConfig config, Vertx vertx) {
    super(config, vertx);
  }

  @Override
  protected Maybe<EmbeddingTokenCount> parseResponse(Buffer responseJson) {
    LOGGER.debug("Parsing response from OpenAI embedding inference");
    return Maybe
      .just(responseJson)
      .flatMap(this::validateBuffer)
      .map(response -> Json.decodeValue(responseJson.toString(), EmbeddingResponse.class))
      .flatMap(this::validateResponse)
      .flatMap(this::extractEmbeddingTokenCount)
      .doOnSuccess(embeddingTokenCount -> {
        LOGGER.debug(
          "Total token processed: {}; Embedding dimension: {}",
          embeddingTokenCount.tokenCount(),
          embeddingTokenCount.embedding().length
        );
      });
  }

  private Maybe<Buffer> validateBuffer(Buffer buffer) {
    if (buffer == null || buffer.length() == 0) {
      return Maybe.error(new GraviteeInferenceOpenaiException("Response buffer is null or empty"));
    }
    return Maybe.just(buffer);
  }

  private Maybe<EmbeddingResponse> validateResponse(EmbeddingResponse response) {
    if (response == null || response.data() == null) {
      return Maybe.error(new GraviteeInferenceOpenaiException("Invalid embedding response structure"));
    }
    return Maybe.just(response);
  }

  private Maybe<EmbeddingTokenCount> extractEmbeddingTokenCount(EmbeddingResponse response) {
    var embedding = response.data().getFirst().embedding();
    int totalTokens = response.usage().total_tokens();
    return Maybe.just(new EmbeddingTokenCount(embedding, totalTokens));
  }

  @Override
  protected Single<HttpResponse<Buffer>> executeHttpRequest(Buffer requestJson) {
    LOGGER.debug("Executing OpenAI embedding inference");

    HttpRequest<Buffer> request = webClient
      .postAbs(getAbsoluteURI())
      .bearerTokenAuthentication(config.getApiKey())
      .putHeader(CONTENT_TYPE, MEDIA_TYPE);

    if (config.getOrganizationId() != null) {
      request.putHeader(OPEN_AI_ORGANIZATION, config.getOrganizationId());
    }

    if (config.getProjectId() != null) {
      request.putHeader(OPEN_AI_PROJECT, config.getProjectId());
    }

    return request.rxSendBuffer(requestJson);
  }

  private String getAbsoluteURI() {
    return config.getUri().toASCIIString() + EMBEDDINGS_ENDPOINT;
  }

  @Override
  protected Single<Buffer> prepareRequest(String input) {
    LOGGER.debug("Preparing OpenAI embedding inference");

    Objects.requireNonNull(input, "Input cannot be null");

    try {
      return Single.just(
        Buffer.buffer(
          Json.encode(
            new EmbeddingRequest(config.getModel(), input, config.getDimensions(), config.getEncodingFormat().getFormat())
          )
        )
      );
    } catch (Exception e) {
      return Single.error(new GraviteeInferenceOpenaiException("Failed to prepare OpenAI request" + e));
    }
  }
}
