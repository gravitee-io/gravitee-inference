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

import io.gravitee.inference.rest.http.CustomHttpConfig;
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
    String requestBodyTemplate,
    String inputLocation,
    String outputEmbeddingLocation
  ) {
    super(uri, method, headers, requestBodyTemplate);
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
