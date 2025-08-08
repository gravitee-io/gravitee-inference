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
package io.gravitee.inference.rest.http;

import io.gravitee.inference.rest.RestConfig;
import io.vertx.core.http.HttpMethod;
import java.net.URI;
import java.util.Map;
import java.util.Objects;

public class CustomHttpConfig extends RestConfig {

  private final HttpMethod method;
  private final Map<String, String> headers;
  private final String requestBodyTemplate;

  public CustomHttpConfig(URI uri, HttpMethod method, Map<String, String> headers, String requestBodyTemplate) {
    super(uri);
    this.method = method;
    this.headers = headers;

    this.requestBodyTemplate = requestBodyTemplate;
  }

  public HttpMethod getMethod() {
    return method;
  }

  public Map<String, String> getHeaders() {
    return headers;
  }

  public String getRequestBodyTemplate() {
    return requestBodyTemplate;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (!(o instanceof CustomHttpConfig that)) return false;
    if (!super.equals(o)) return false;
    return (
      Objects.equals(method, that.method) &&
      Objects.equals(headers, that.headers) &&
      Objects.equals(requestBodyTemplate, that.requestBodyTemplate)
    );
  }

  @Override
  public int hashCode() {
    return Objects.hash(super.hashCode(), method, headers, requestBodyTemplate);
  }
}
