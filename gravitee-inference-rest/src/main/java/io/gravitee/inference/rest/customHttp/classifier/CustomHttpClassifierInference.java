package io.gravitee.inference.rest.customHttp.classifier;

import io.gravitee.inference.api.classifier.ClassifierResult;
import io.gravitee.inference.rest.customHttp.CustomHttpConfig;
import io.gravitee.inference.rest.customHttp.CustomHttpInference;
import io.gravitee.inference.rest.customHttp.embedding.CustomHttpEmbeddingConfig;
import io.reactivex.rxjava3.core.Maybe;
import io.reactivex.rxjava3.core.Single;
import io.vertx.rxjava3.core.Vertx;
import io.vertx.rxjava3.core.buffer.Buffer;
import io.vertx.rxjava3.ext.web.client.HttpResponse;

import java.net.URI;

public class CustomHttpClassifierInference extends CustomHttpInference<CustomHttpEmbeddingConfig, String, ClassifierResult> {

  public CustomHttpClassifierInference(CustomHttpEmbeddingConfig config, Vertx vertx) {
    super(config, vertx);
  }

  @Override
  protected Maybe<ClassifierResult> parseResponse(Buffer responseJson) {
    return null;
  }

  @Override
  protected Single<HttpResponse<Buffer>> executeHttpRequest(Buffer requestJson) {
    return null;
  }

  @Override
  protected Single<Buffer> prepareRequest(String input) {
    return null;
  }
}
