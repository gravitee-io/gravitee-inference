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

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

import io.gravitee.inference.rest.http.embedding.CustomHttpEmbeddingConfig;
import io.gravitee.inference.rest.http.embedding.CustomHttpEmbeddingInference;
import io.reactivex.rxjava3.core.Single;
import io.vertx.core.http.HttpMethod;
import io.vertx.core.json.Json;
import io.vertx.rxjava3.core.Vertx;
import io.vertx.rxjava3.core.buffer.Buffer;
import io.vertx.rxjava3.ext.web.client.HttpRequest;
import io.vertx.rxjava3.ext.web.client.HttpResponse;
import io.vertx.rxjava3.ext.web.client.WebClient;
import java.net.URI;
import java.util.Map;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvFileSource;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

@ExtendWith(MockitoExtension.class)
public class CustomHttpEmbeddingInferenceTest {

  @Mock
  private WebClient mockWebClient;

  @Mock
  private HttpRequest<Buffer> mockHttpRequest;

  @Mock
  private HttpResponse<Buffer> mockHttpResponse;

  private Vertx vertx;

  @BeforeEach
  void setUp() {
    vertx = Vertx.vertx();
  }

  @ParameterizedTest(name = "Test {6}")
  @CsvFileSource(resources = "/embedding-test-data.csv", numLinesToSkip = 1)
  void testEmbeddingInferenceWithJsonPath(
    String inputLocation,
    String requestBodyTemplate,
    String jsonResponse,
    String outputJsonPath,
    int expectedEmbeddingLength,
    String expectedRequestBody,
    String description
  ) {
    CustomHttpEmbeddingConfig config = new CustomHttpEmbeddingConfig(
      URI.create("http://localhost:8000/embed/"),
      HttpMethod.POST,
      null,
      requestBodyTemplate,
      inputLocation,
      outputJsonPath
    );

    CustomHttpEmbeddingInference inference = new CustomHttpEmbeddingInference(config, vertx);

    ArgumentCaptor<Buffer> requestBodyCaptor = ArgumentCaptor.forClass(Buffer.class);

    when(mockWebClient.requestAbs(config.getMethod(), config.getUri().toString())).thenReturn(mockHttpRequest);
    when(mockHttpRequest.followRedirects(true)).thenReturn(mockHttpRequest);
    when(mockHttpRequest.putHeader(anyString(), anyString())).thenReturn(mockHttpRequest);
    when(mockHttpRequest.rxSendBuffer(requestBodyCaptor.capture())).thenReturn(Single.just(mockHttpResponse));
    when(mockHttpResponse.statusCode()).thenReturn(200);
    when(mockHttpResponse.body()).thenReturn(Buffer.buffer(jsonResponse));

    try {
      var webClientField = inference.getClass().getSuperclass().getSuperclass().getDeclaredField("webClient");
      webClientField.setAccessible(true);
      webClientField.set(inference, mockWebClient);
    } catch (Exception e) {
      throw new RuntimeException("Failed to inject mock WebClient", e);
    }

    String inputText = "Test input text for " + description;

    var testObserver = inference.infer(inputText).test();

    testObserver
      .assertComplete()
      .assertNoErrors()
      .assertValue(embeddingTokenCount -> expectedEmbeddingLength == embeddingTokenCount.embedding().length);

    Buffer capturedRequestBody = requestBodyCaptor.getValue();
    String actualRequestBody = capturedRequestBody.toString();

    assertEquals(
      Json.decodeValue(expectedRequestBody),
      Json.decodeValue(actualRequestBody),
      "Request body should match the expected fulfilled template for: " + description
    );

    verify(mockHttpRequest).rxSendBuffer(any(Buffer.class));
  }
}
