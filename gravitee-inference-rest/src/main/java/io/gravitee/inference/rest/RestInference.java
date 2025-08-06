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
package io.gravitee.inference.rest;

import io.gravitee.inference.api.InferenceModel;
import io.reactivex.rxjava3.core.Maybe;
import io.reactivex.rxjava3.core.Single;
import io.vertx.rxjava3.core.Vertx;
import io.vertx.rxjava3.core.buffer.Buffer;
import io.vertx.rxjava3.ext.web.client.HttpResponse;
import io.vertx.rxjava3.ext.web.client.WebClient;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class RestInference<C extends RestConfig, I, O> extends InferenceModel<C, I, Maybe<O>> {

  private static final int HTTP_CODE_OK = 200;
  private static final int HTTP_CODE_REDIRECTION = 300;
  private static final Logger LOGGER = LoggerFactory.getLogger(RestInference.class);
  protected final Vertx vertx;
  protected final WebClient webClient;

  protected RestInference(C config, Vertx vertx) {
    super(config);
    this.vertx = vertx;
    this.webClient = WebClient.create(this.vertx);
  }

  @Override
  public Maybe<O> infer(I input) {
    LOGGER.debug("Requesting inference model for \"{}\"", input);

    return prepareRequest(input)
      .flatMap(this::executeHttpRequest)
      .flatMapMaybe(response -> {
        if (response.statusCode() < HTTP_CODE_OK || response.statusCode() >= HTTP_CODE_REDIRECTION) {
          return Maybe.error(new RuntimeException("HTTP request failed" + response.statusCode() + response.bodyAsString()));
        }
        return parseResponse(response.body());
      });
  }

  protected abstract Single<Buffer> prepareRequest(I input);

  protected abstract Maybe<O> parseResponse(Buffer responseJson);

  protected abstract Single<HttpResponse<Buffer>> executeHttpRequest(Buffer requestJson);

  @Override
  public void close() {
    this.webClient.close();
  }
}
