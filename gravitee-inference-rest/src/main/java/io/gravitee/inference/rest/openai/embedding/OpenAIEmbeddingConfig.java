package io.gravitee.inference.rest.openai.embedding;

import io.gravitee.inference.rest.openai.OpenaiConfig;

import java.net.URI;


public class OpenAIEmbeddingConfig extends OpenaiConfig {

  public Integer dimensions;
  public EncodingFormat encodingFormat = EncodingFormat.FLOAT;

  public OpenAIEmbeddingConfig(URI uri, String apiKey, String organizationId, String projectId, String model, Integer dimensions, EncodingFormat encodingFormat) {
    super(uri, apiKey, organizationId, projectId, model);

    this.dimensions = validateDimensions(dimensions);
    this.encodingFormat = encodingFormat != null ? encodingFormat : EncodingFormat.FLOAT;
  }

  public OpenAIEmbeddingConfig(URI uri, String apiKey, String organizationId, String projectId, String model, Integer dimensions) {
    this(uri, apiKey, organizationId, projectId, model, dimensions, EncodingFormat.FLOAT);
  }

  public OpenAIEmbeddingConfig(URI uri, String apiKey, String organizationId, String projectId, String model) {
    this(uri, apiKey, organizationId, projectId, model, null, EncodingFormat.FLOAT);
  }

  public OpenAIEmbeddingConfig(URI uri, String apiKey, String organizationId, String model) {
    this(uri, apiKey, organizationId, null, model, null, EncodingFormat.FLOAT);
  }

  public OpenAIEmbeddingConfig(URI uri, String apiKey, String model) {
    this(uri, apiKey, null, null, model, null, EncodingFormat.FLOAT);
  }

  private Integer validateDimensions(Integer dimensions) {
    if (dimensions != null && dimensions <= 0) {
      throw new IllegalArgumentException("Dimensions must be positive when specified");
    }
    return dimensions;
  }
}
