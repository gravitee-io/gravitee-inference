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

import io.gravitee.inference.rest.openai.OpenaiConfig;
import java.net.URI;
import java.util.Objects;

public class OpenAIEmbeddingConfig extends OpenaiConfig {

  private Integer dimensions;
  private EncodingFormat encodingFormat = EncodingFormat.FLOAT;

  public OpenAIEmbeddingConfig(
    URI uri,
    String apiKey,
    String organizationId,
    String projectId,
    String model,
    Integer dimensions,
    EncodingFormat encodingFormat
  ) {
    super(uri, apiKey, organizationId, projectId, model);
    this.dimensions = validateDimensions(dimensions);
    this.encodingFormat = encodingFormat != null ? encodingFormat : EncodingFormat.FLOAT;
  }

  public OpenAIEmbeddingConfig(
    URI uri,
    String apiKey,
    String organizationId,
    String projectId,
    String model,
    Integer dimensions
  ) {
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

  public Integer getDimensions() {
    return dimensions;
  }

  public EncodingFormat getEncodingFormat() {
    return encodingFormat;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (!(o instanceof OpenAIEmbeddingConfig that)) return false;
    if (!super.equals(o)) return false;
    return Objects.equals(dimensions, that.dimensions) && encodingFormat == that.encodingFormat;
  }

  @Override
  public int hashCode() {
    return Objects.hash(super.hashCode(), dimensions, encodingFormat);
  }

  @Override
  public String toString() {
    return (
      "OpenAIEmbeddingConfig{" +
      "uri=" +
      getUri() +
      ", apiKey='[PROTECTED]'" +
      ", organizationId='" +
      getOrganizationId() +
      '\'' +
      ", projectId='" +
      getProjectId() +
      '\'' +
      ", model='" +
      getModel() +
      '\'' +
      ", dimensions=" +
      dimensions +
      ", encodingFormat=" +
      encodingFormat +
      '}'
    );
  }
}
