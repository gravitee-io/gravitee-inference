package io.gravitee.inference.rest.http;

import io.gravitee.inference.rest.RestInference;
import io.vertx.rxjava3.core.Vertx;

public abstract class CustomHttpInference<C extends CustomHttpConfig, I, O> extends RestInference<C, I, O> {

  public CustomHttpInference(C config, Vertx vertx) {
    super(config, vertx);
  }
}
