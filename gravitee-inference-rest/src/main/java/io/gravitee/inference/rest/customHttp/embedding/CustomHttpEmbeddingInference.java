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
    return Maybe
      .fromCallable(() -> {
        String outputLocation = getOutputLocation();

        LOGGER.debug("Extracting response from location: {}", outputLocation);
        LOGGER.debug(
          "Extracting response from input: {}",
          responseJson.toString().substring(0, Math.min(60, responseJson.toString().length()))
        );

        DocumentContext responseContext = JsonPath.parse(responseJson.toString());
        return responseContext.read(outputLocation, float[].class);
      })
      .map(embedding -> {
        return new EmbeddingTokenCount(embedding, -1);
      });
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
