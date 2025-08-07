package io.gravitee.inference.rest.customHttp.embedding;

import io.gravitee.inference.rest.customHttp.CustomHttpConfig;
import io.vertx.core.http.HttpMethod;
import java.net.URI;
import java.util.Map;
import java.util.Objects;

public class CustomHttpEmbeddingConfig extends CustomHttpConfig {

  private final String inputLocation;
  private final String outputEmbeddingLocation;

  public CustomHttpEmbeddingConfig(
    URI uri,
    HttpMethod method,
    Map<String, String> headers,
    String contentType,
    String requestBodyTemplate,
    String inputLocation,
    String outputEmbeddingLocation
  ) {
    super(uri, method, headers, contentType, requestBodyTemplate);
    Objects.requireNonNull(inputLocation, "Input location cannot be null");
    Objects.requireNonNull(outputEmbeddingLocation, "Output embedding location cannot be null");

    this.inputLocation = inputLocation;
    this.outputEmbeddingLocation = outputEmbeddingLocation;
  }

  public String getInputLocation() {
    return inputLocation;
  }

  public String getOutputEmbeddingLocation() {
    return outputEmbeddingLocation;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (!(o instanceof CustomHttpEmbeddingConfig that)) return false;
    if (!super.equals(o)) return false;
    return (
      Objects.equals(inputLocation, that.inputLocation) &&
      Objects.equals(outputEmbeddingLocation, that.outputEmbeddingLocation)
    );
  }

  @Override
  public int hashCode() {
    return Objects.hash(super.hashCode(), inputLocation, outputEmbeddingLocation);
  }
}
