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
package io.gravitee.inference.rest.http.embedding;

import com.jayway.jsonpath.DocumentContext;
import com.jayway.jsonpath.JsonPath;
import io.gravitee.inference.api.embedding.EmbeddingTokenCount;
import io.gravitee.inference.rest.http.CustomHttpInference;
import io.reactivex.rxjava3.core.Maybe;
import io.reactivex.rxjava3.core.Single;
import io.vertx.rxjava3.core.Vertx;
import io.vertx.rxjava3.core.buffer.Buffer;
import io.vertx.rxjava3.ext.web.client.HttpRequest;
import io.vertx.rxjava3.ext.web.client.HttpResponse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class CustomHttpEmbeddingInference
  extends CustomHttpInference<CustomHttpEmbeddingConfig, String, EmbeddingTokenCount> {

  private static final Logger LOGGER = LoggerFactory.getLogger(CustomHttpEmbeddingInference.class);
  public static final String MEDIA_TYPE_JSON = "application/json";
  public static final String CONTENT_TYPE = "Content-Type";

  public CustomHttpEmbeddingInference(CustomHttpEmbeddingConfig config, Vertx vertx) {
    super(config, vertx);
  }

  @Override
  protected Single<Buffer> prepareRequest(String input) {
    String inputLocation = getInputLocation();
    String requestBodyTemplate = getRequestBodyTemplate();

    LOGGER.debug("Preparing request with input location: {} with template {}", inputLocation, requestBodyTemplate);

    return Single
      .fromCallable(() -> JsonPath.parse(requestBodyTemplate))
      .map(ctx -> ctx.set(inputLocation, input))
      .map(DocumentContext::jsonString)
      .doOnSuccess(json -> LOGGER.debug("Prepared request body: {}", json))
      .map(Buffer::buffer);
  }

  @Override
  protected Single<HttpResponse<Buffer>> executeHttpRequest(Buffer requestJson) {
    HttpRequest<Buffer> request = webClient
      .requestAbs(config.getMethod(), config.getUri().toASCIIString())
      .followRedirects(true);
    if (config.getHeaders() != null) {
      config.getHeaders().forEach(request::putHeader);
    }

    request.putHeader(CONTENT_TYPE, MEDIA_TYPE_JSON);

    LOGGER.debug("Executing HTTP request:\n{} {}\n\n{}\n", config.getMethod(), config.getUri(), request.headers());

    return request.rxSendBuffer(requestJson);
  }

  @Override
  protected Maybe<EmbeddingTokenCount> parseResponse(Buffer responseJson) {
    return Maybe
      .fromCallable(() -> {
        String outputLocation = getOutputLocation();
        String responseBody = responseJson.toString();

        LOGGER.debug("Extracting response from location: {}", outputLocation);
        LOGGER.debug("Extracting response from input: {}", responseBody.substring(0, Math.min(60, responseBody.length())));

        DocumentContext responseContext = JsonPath.parse(responseBody);
        return responseContext.read(outputLocation, float[].class);
      })
      .map(embedding -> new EmbeddingTokenCount(embedding, -1));
  }

  private String getRequestBodyTemplate() {
    return config.getRequestBodyTemplate();
  }

  private String getInputLocation() {
    return config.getInputLocation();
  }

  private String getOutputLocation() {
    return config.getOutputEmbeddingLocation();
  }
}
