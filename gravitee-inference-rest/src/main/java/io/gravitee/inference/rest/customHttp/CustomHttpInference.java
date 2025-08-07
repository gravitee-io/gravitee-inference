package io.gravitee.inference.rest.customHttp;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jayway.jsonpath.Configuration;
import com.jayway.jsonpath.DocumentContext;
import com.jayway.jsonpath.JsonPath;
import io.gravitee.inference.rest.RestInference;
import io.vertx.rxjava3.core.Vertx;
import io.vertx.rxjava3.core.buffer.Buffer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class CustomHttpInference<C extends CustomHttpConfig, I, O> extends RestInference<C, I, O> {

  private static Logger LOGGER = LoggerFactory.getLogger(CustomHttpInference.class);

  public CustomHttpInference(C config, Vertx vertx) {
    super(config, vertx);
  }
}
