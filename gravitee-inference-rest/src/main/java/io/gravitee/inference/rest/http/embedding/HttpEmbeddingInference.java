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
import com.jayway.jsonpath.JsonPathException;
import io.gravitee.inference.api.embedding.EmbeddingTokenCount;
import io.gravitee.inference.rest.http.GraviteeInferenceHttpException;
import io.gravitee.inference.rest.http.HttpInference;
import io.reactivex.rxjava3.core.Maybe;
import io.reactivex.rxjava3.core.Single;
import io.vertx.rxjava3.core.Vertx;
import io.vertx.rxjava3.core.buffer.Buffer;
import io.vertx.rxjava3.ext.web.client.HttpRequest;
import io.vertx.rxjava3.ext.web.client.HttpResponse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class HttpEmbeddingInference extends HttpInference<HttpEmbeddingConfig, String, EmbeddingTokenCount> {

  private static final Logger LOGGER = LoggerFactory.getLogger(HttpEmbeddingInference.class);
  public static final String MEDIA_TYPE_JSON = "application/json";
  public static final String CONTENT_TYPE = "Content-Type";

  public HttpEmbeddingInference(HttpEmbeddingConfig config, Vertx vertx) {
    super(config, vertx);
  }

  @Override
  protected Single<Buffer> prepareRequest(String input) {
    if (input == null || input.trim().isEmpty()) {
      return Single.error(new IllegalArgumentException("Input cannot be null or empty"));
    }

    String inputLocation = getInputLocation();
    String requestBodyTemplate = getRequestBodyTemplate();

    if (requestBodyTemplate == null) {
      return Single.error(new IllegalStateException("Request body template is not configured"));
    }

    LOGGER.debug("Preparing request with input location: {} with template {}", inputLocation, requestBodyTemplate);
    try {
      return Single
        .fromCallable(() -> {
          try {
            return JsonPath.parse(requestBodyTemplate);
          } catch (JsonPathException e) {
            throw new GraviteeInferenceHttpException("Invalid JSON template: " + e.getMessage());
          }
        })
        .map(ctx -> {
          try {
            return ctx.set(inputLocation, input);
          } catch (JsonPathException e) {
            throw new GraviteeInferenceHttpException(
              String.format("Cannot set input at location '%s': %s", inputLocation, e.getMessage())
            );
          }
        })
        .map(DocumentContext::jsonString)
        .doOnSuccess(json -> LOGGER.debug("Prepared request body: {}", json))
        .map(Buffer::buffer)
        .doOnError(error ->
          LOGGER.error(
            "Failed to prepare request for input: {}",
            input.length() > 100 ? input.substring(0, 100) + "..." : input,
            error
          )
        );
    } catch (Exception e) {
      LOGGER.error("An error occurs while preparing request", e);
      return Single.error(new GraviteeInferenceHttpException("An error occurs while preparing request"));
    }
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

    return request
      .rxSendBuffer(requestJson)
      .doOnError(error -> LOGGER.error("HTTP request failed for URI: {}", config.getUri()));
  }

  @Override
  protected Maybe<EmbeddingTokenCount> parseResponse(Buffer responseJson) {
    return Maybe
      .fromCallable(() -> {
        String outputLocation = getOutputLocation();

        if (outputLocation == null || outputLocation.trim().isEmpty()) {
          throw new IllegalStateException("Output location is not configured");
        }

        String responseBody = responseJson.toString();

        LOGGER.debug("Extracting response from location: {}", outputLocation);
        LOGGER.debug("Extracting response from input: {}", responseBody.substring(0, Math.min(60, responseBody.length())));

        try {
          DocumentContext responseContext = JsonPath.parse(responseBody);
          float[] embedding = responseContext.read(outputLocation, float[].class);

          if (embedding == null || embedding.length == 0) {
            throw new GraviteeInferenceHttpException(
              String.format("No embedding data found at location: %s", outputLocation)
            );
          }

          return new EmbeddingTokenCount(embedding, -1);
        } catch (JsonPathException e) {
          throw new GraviteeInferenceHttpException(
            String.format("Failed to extract embedding from location '%s': %s", outputLocation, e.getMessage())
          );
        } catch (ClassCastException e) {
          throw new GraviteeInferenceHttpException(
            String.format("Data at location '%s' is not a valid float array: %s", outputLocation, e.getMessage())
          );
        }
      })
      .doOnError(error -> LOGGER.error("Failed to parse response", error));
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
