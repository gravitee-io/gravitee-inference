package io.gravitee.inference.rest.customHttp.classifier;

import io.gravitee.inference.api.classifier.ClassifierResult;
import io.gravitee.inference.rest.customHttp.CustomHttpConfig;
import io.gravitee.inference.rest.customHttp.CustomHttpInference;
import io.gravitee.inference.rest.customHttp.embedding.CustomHttpEmbeddingConfig;
import io.reactivex.rxjava3.core.Maybe;
import io.reactivex.rxjava3.core.Single;
import io.vertx.core.http.HttpMethod;
import io.vertx.rxjava3.core.Vertx;
import io.vertx.rxjava3.core.buffer.Buffer;
import io.vertx.rxjava3.ext.web.client.HttpResponse;

import java.net.URI;
import java.util.Map;

public class CustomHttpClassifierConfig extends CustomHttpConfig {

  // private final String inputLocation;

  // private final String outputLabelLocation;
  // private final String outputScoreLocation;


  public CustomHttpClassifierConfig(URI uri, HttpMethod method, Map<String, String> headers, String requestBodyTemplate, String contentType) {
    super(uri, method, headers, requestBodyTemplate, contentType);
  }
}
