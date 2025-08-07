package io.gravitee.inference.rest.customHttp;

import io.gravitee.inference.rest.RestConfig;
import io.gravitee.inference.rest.openai.OpenaiConfig;
import io.vertx.core.http.HttpMethod;
import java.net.URI;
import java.util.Map;
import java.util.Objects;

public class CustomHttpConfig extends RestConfig {

  private final HttpMethod method;
  private final Map<String, String> headers;
  private final String contentType;
  private final String requestBodyTemplate;

  public CustomHttpConfig(
    URI uri,
    HttpMethod method,
    Map<String, String> headers,
    String contentType,
    String requestBodyTemplate
  ) {
    super(uri);
    this.method = method;
    this.headers = headers;
    this.contentType = contentType;

    this.requestBodyTemplate = requestBodyTemplate;
  }

  public HttpMethod getMethod() {
    return method;
  }

  public Map<String, String> getHeaders() {
    return headers;
  }

  public String getContentType() {
    return contentType;
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
      Objects.equals(contentType, that.contentType) &&
      Objects.equals(requestBodyTemplate, that.requestBodyTemplate)
    );
  }

  @Override
  public int hashCode() {
    return Objects.hash(super.hashCode(), method, headers, contentType, requestBodyTemplate);
  }
}
