package io.gravitee.inference.rest.customHttp.embedding;

import com.jayway.jsonpath.DocumentContext;
import com.jayway.jsonpath.JsonPath;
import io.gravitee.inference.api.embedding.EmbeddingTokenCount;
import io.gravitee.inference.rest.customHttp.CustomHttpInference;
import io.reactivex.rxjava3.core.Maybe;
import io.reactivex.rxjava3.core.Single;
import io.vertx.rxjava3.core.Vertx;
import io.vertx.rxjava3.core.buffer.Buffer;
import io.vertx.rxjava3.ext.web.client.HttpRequest;
import io.vertx.rxjava3.ext.web.client.HttpResponse;

import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class CustomHttpEmbeddingInference
        extends CustomHttpInference<CustomHttpEmbeddingConfig, String, EmbeddingTokenCount> {

  private static final Logger LOGGER = LoggerFactory.getLogger(CustomHttpEmbeddingInference.class);

  private static final String DEFAULT_INPUT_LOCATION = "$";
  private static final String DEFAULT_OUTPUT_EMBEDDING_LOCATION = "$";

  public CustomHttpEmbeddingInference(CustomHttpEmbeddingConfig config, Vertx vertx) {
    super(config, vertx);
  }

  @Override
  protected Single<Buffer> prepareRequest(String input) {
    try {
      String inputLocation = getInputLocation();
      String requestBodyTemplate = getRequestBodyTemplate();

      LOGGER.debug("Preparing request with input location: {} with template {}", inputLocation, requestBodyTemplate);

      if ("$".equals(inputLocation)) {
        return Single.just(Buffer.buffer(input));
      }

      DocumentContext templateContext = JsonPath.parse(requestBodyTemplate);
      DocumentContext resultContext = templateContext.set(inputLocation, input);

      String requestBody = resultContext.jsonString();

      LOGGER.debug("Prepared request body: {}", requestBody);

      return Single.just(Buffer.buffer(requestBody));
    } catch (Exception e) {
      LOGGER.error("Error preparing request for input location: {}", config.getInputLocation(), e);
      return Single.error(new RuntimeException("Failed to prepare request", e));
    }
  }

  @Override
  protected Single<HttpResponse<Buffer>> executeHttpRequest(Buffer requestJson) {
    HttpRequest<Buffer> request = webClient.requestAbs(config.getMethod(), config.getUri().toString()).followRedirects(true);
    if (config.getHeaders() != null) {
      config.getHeaders().forEach(request::putHeader);
    }

    if (config.getContentType() != null) {
      request.putHeader("Content-Type", config.getContentType());
    }

    LOGGER.debug("Executing HTTP request:\n{} {}\n\n{}\n", config.getMethod(), config.getUri(), request.headers());

    return request.rxSendBuffer(requestJson);
  }

  @Override
  protected Maybe<EmbeddingTokenCount> parseResponse(Buffer responseJson) {
    try {
      String outputLocation = getOutputLocation();

      LOGGER.debug("Extracting response from location: {}", outputLocation);
      LOGGER.debug(
              "Extracting response from input: {}",
              responseJson.toString().substring(0, Math.min(60, responseJson.toString().length()))
      );

      DocumentContext responseContext = JsonPath.parse(responseJson.toString());
      List<Number> embeddingResponse = responseContext.read(outputLocation);
      float[] embedding = new float[embeddingResponse.size()];

      for (int i = 0; i < embeddingResponse.size(); i++) {
        Number value = embeddingResponse.get(i);
        embedding[i] = value.floatValue();
      }

      return Maybe.just(new EmbeddingTokenCount(embedding, -1));
    } catch (Exception e) {
      LOGGER.error("Error extracting response from location: {}", getOutputLocation(), e);
      throw new RuntimeException("Failed to extract response", e);
    }
  }

  private String getRequestBodyTemplate() {
    return config.getRequestBodyTemplate();
  }

  private String getInputLocation() {
    return config.getInputLocation() != null ? config.getInputLocation() : DEFAULT_INPUT_LOCATION;
  }

  private String getOutputLocation() {
    return config.getOutputEmbeddingLocation() != null
            ? config.getOutputEmbeddingLocation()
            : DEFAULT_OUTPUT_EMBEDDING_LOCATION;
  }
}
