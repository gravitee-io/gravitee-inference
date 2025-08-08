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

import io.gravitee.inference.rest.http.GraviteeInferenceHttpException;
import io.gravitee.inference.rest.http.embedding.HttpEmbeddingConfig;
import io.gravitee.inference.rest.http.embedding.HttpEmbeddingInference;
import io.reactivex.rxjava3.core.Single;
import io.vertx.core.http.HttpMethod;
import io.vertx.core.json.Json;
import io.vertx.rxjava3.core.Vertx;
import io.vertx.rxjava3.core.buffer.Buffer;
import io.vertx.rxjava3.ext.web.client.HttpRequest;
import io.vertx.rxjava3.ext.web.client.HttpResponse;
import io.vertx.rxjava3.ext.web.client.WebClient;
import java.net.URI;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvFileSource;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

@ExtendWith(MockitoExtension.class)
public class HttpEmbeddingInferenceTest {

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
    HttpEmbeddingConfig config = new HttpEmbeddingConfig(
      URI.create("http://localhost:8000/embed/"),
      HttpMethod.POST,
      null,
      requestBodyTemplate,
      inputLocation,
      outputJsonPath
    );

    HttpEmbeddingInference inference = new HttpEmbeddingInference(config, vertx);

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

  @Test
  void testInferWithNullInput() {
    HttpEmbeddingConfig config = createValidConfig();
    HttpEmbeddingInference inference = new HttpEmbeddingInference(config, vertx);

    var testObserver = inference.infer(null).test();

    testObserver.assertNotComplete().assertError(IllegalArgumentException.class);
  }

  @Test
  void testInferWithEmptyInput() {
    HttpEmbeddingConfig config = createValidConfig();
    HttpEmbeddingInference inference = new HttpEmbeddingInference(config, vertx);

    var testObserver = inference.infer("").test();

    testObserver.assertNotComplete().assertError(IllegalArgumentException.class);
  }

  @Test
  void testInferWithWhitespaceOnlyInput() {
    HttpEmbeddingConfig config = createValidConfig();
    HttpEmbeddingInference inference = new HttpEmbeddingInference(config, vertx);

    var testObserver = inference.infer("   \t\n   ").test();

    testObserver.assertNotComplete().assertError(IllegalArgumentException.class);
  }

  @Test
  void testInferWithMissingRequestBodyTemplate() {
    HttpEmbeddingConfig config = new HttpEmbeddingConfig(
      URI.create("http://localhost:8000/embed/"),
      HttpMethod.POST,
      null,
      null,
      "$.text",
      "$.embedding"
    );
    HttpEmbeddingInference inference = new HttpEmbeddingInference(config, vertx);

    var testObserver = inference.infer("test input").test();

    testObserver.assertNotComplete().assertError(IllegalStateException.class);
  }

  @Test
  void testInferWithNullOutputLocation() {
    try {
      HttpEmbeddingConfig config = new HttpEmbeddingConfig(
        URI.create("http://localhost:8000/embed/"),
        HttpMethod.POST,
        null,
        "{\"text\": \"<DATA>\"}",
        "$.text",
        null
      );
    } catch (NullPointerException e) {
      assertEquals("Output embedding location cannot be null", e.getMessage());
    }
  }

  @Test
  void testInferWithEmptyOutputLocation() {
    HttpEmbeddingConfig config = new HttpEmbeddingConfig(
      URI.create("http://localhost:8000/embed/"),
      HttpMethod.POST,
      null,
      "{\"text\": \"<DATA>\"}",
      "$.text",
      ""
    );
    HttpEmbeddingInference inference = new HttpEmbeddingInference(config, vertx);

    setupMockWebClientForSuccessfulRequest(inference, "{\"embedding\": [0.1, 0.2, 0.3]}");

    var testObserver = inference.infer("test input").test();

    testObserver.assertNotComplete().assertError(IllegalStateException.class);
  }

  @Test
  void testInferWithInvalidJsonTemplate() {
    HttpEmbeddingConfig config = new HttpEmbeddingConfig(
      URI.create("http://localhost:8000/embed/"),
      HttpMethod.POST,
      null,
      "{invalid json}",
      "$.text",
      "$.embedding"
    );
    HttpEmbeddingInference inference = new HttpEmbeddingInference(config, vertx);

    var testObserver = inference.infer("test input").test();

    testObserver.assertNotComplete().assertError(GraviteeInferenceHttpException.class);
  }

  @Test
  void testInferWithInvalidInputLocation() {
    HttpEmbeddingConfig config = new HttpEmbeddingConfig(
      URI.create("http://localhost:8000/embed/"),
      HttpMethod.POST,
      null,
      "{\"text\": \"<DATA>\"}",
      "$.invalid[path",
      "$.embedding"
    );
    HttpEmbeddingInference inference = new HttpEmbeddingInference(config, vertx);

    var testObserver = inference.infer("test input").test();

    testObserver.assertNotComplete().assertError(GraviteeInferenceHttpException.class);
  }

  @Test
  void testInferWithNonExistentInputLocation() {
    HttpEmbeddingConfig config = new HttpEmbeddingConfig(
      URI.create("http://localhost:8000/embed/"),
      HttpMethod.POST,
      null,
      "{\"text\": \"<DATA>\"}",
      "$.nonexistent.field",
      "$.embedding"
    );
    HttpEmbeddingInference inference = new HttpEmbeddingInference(config, vertx);

    var testObserver = inference.infer("test input").test();

    testObserver.assertNotComplete().assertError(GraviteeInferenceHttpException.class);
  }

  @Test
  void testInferWithHttpErrorStatusCode() {
    HttpEmbeddingConfig config = createValidConfig();
    HttpEmbeddingInference inference = new HttpEmbeddingInference(config, vertx);

    when(mockWebClient.requestAbs(any(), anyString())).thenReturn(mockHttpRequest);
    when(mockHttpRequest.followRedirects(true)).thenReturn(mockHttpRequest);
    when(mockHttpRequest.putHeader(anyString(), anyString())).thenReturn(mockHttpRequest);
    when(mockHttpRequest.rxSendBuffer(any(Buffer.class))).thenReturn(Single.just(mockHttpResponse));
    when(mockHttpResponse.statusCode()).thenReturn(500);
    when(mockHttpResponse.bodyAsString()).thenReturn("Internal Server Error");

    try {
      var webClientField = inference.getClass().getSuperclass().getSuperclass().getDeclaredField("webClient");
      webClientField.setAccessible(true);
      webClientField.set(inference, mockWebClient);
    } catch (Exception e) {
      throw new RuntimeException("Failed to inject mock WebClient", e);
    }

    var testObserver = inference.infer("test input").test();

    testObserver.assertNotComplete().assertError(RuntimeException.class);
  }

  @Test
  void testInferWithHttpConnectionFailure() {
    HttpEmbeddingConfig config = createValidConfig();
    HttpEmbeddingInference inference = new HttpEmbeddingInference(config, vertx);

    when(mockWebClient.requestAbs(any(), anyString())).thenReturn(mockHttpRequest);
    when(mockHttpRequest.followRedirects(true)).thenReturn(mockHttpRequest);
    when(mockHttpRequest.putHeader(anyString(), anyString())).thenReturn(mockHttpRequest);
    when(mockHttpRequest.rxSendBuffer(any(Buffer.class)))
      .thenReturn(Single.error(new RuntimeException("Connection failed")));

    try {
      var webClientField = inference.getClass().getSuperclass().getSuperclass().getDeclaredField("webClient");
      webClientField.setAccessible(true);
      webClientField.set(inference, mockWebClient);
    } catch (Exception e) {
      throw new RuntimeException("Failed to inject mock WebClient", e);
    }

    var testObserver = inference.infer("test input").test();

    testObserver.assertNotComplete().assertError(RuntimeException.class);
  }

  @Test
  void testInferWithInvalidJsonResponse() {
    HttpEmbeddingConfig config = createValidConfig();
    HttpEmbeddingInference inference = new HttpEmbeddingInference(config, vertx);

    setupMockWebClientForSuccessfulRequest(inference, "{invalid json response}");

    var testObserver = inference.infer("test input").test();

    testObserver.assertNotComplete().assertError(GraviteeInferenceHttpException.class);
  }

  @Test
  void testInferWithMissingEmbeddingInResponse() {
    HttpEmbeddingConfig config = createValidConfig();
    HttpEmbeddingInference inference = new HttpEmbeddingInference(config, vertx);

    setupMockWebClientForSuccessfulRequest(inference, "{\"result\": \"success\"}");

    var testObserver = inference.infer("test input").test();

    testObserver.assertNotComplete().assertError(GraviteeInferenceHttpException.class);
  }

  @Test
  void testInferWithEmptyEmbeddingArray() {
    HttpEmbeddingConfig config = createValidConfig();
    HttpEmbeddingInference inference = new HttpEmbeddingInference(config, vertx);

    setupMockWebClientForSuccessfulRequest(inference, "{\"embedding\": []}");

    var testObserver = inference.infer("test input").test();

    testObserver.assertNotComplete().assertError(GraviteeInferenceHttpException.class);
  }

  @Test
  void testInferWithNullEmbeddingInResponse() {
    HttpEmbeddingConfig config = createValidConfig();
    HttpEmbeddingInference inference = new HttpEmbeddingInference(config, vertx);

    setupMockWebClientForSuccessfulRequest(inference, "{\"embedding\": null}");

    var testObserver = inference.infer("test input").test();

    testObserver.assertNotComplete().assertError(GraviteeInferenceHttpException.class);
  }

  @Test
  void testInferWithInvalidEmbeddingFormat() {
    HttpEmbeddingConfig config = createValidConfig();
    HttpEmbeddingInference inference = new HttpEmbeddingInference(config, vertx);

    setupMockWebClientForSuccessfulRequest(inference, "{\"embedding\": \"not an array\"}");

    var testObserver = inference.infer("test input").test();

    testObserver.assertNotComplete().assertError(GraviteeInferenceHttpException.class);
  }

  @Test
  void testInferWithInvalidOutputJsonPath() {
    HttpEmbeddingConfig config = new HttpEmbeddingConfig(
      URI.create("http://localhost:8000/embed/"),
      HttpMethod.POST,
      null,
      "{\"text\": \"<DATA>\"}",
      "$.text",
      "$.invalid[path"
    );
    HttpEmbeddingInference inference = new HttpEmbeddingInference(config, vertx);

    setupMockWebClientForSuccessfulRequest(inference, "{\"embedding\": [0.1, 0.2, 0.3]}");

    var testObserver = inference.infer("test input").test();

    testObserver.assertNotComplete().assertError(GraviteeInferenceHttpException.class);
  }

  @Test
  void testInferWithNonExistentOutputPath() {
    HttpEmbeddingConfig config = new HttpEmbeddingConfig(
      URI.create("http://localhost:8000/embed/"),
      HttpMethod.POST,
      null,
      "{\"text\": \"<DATA>\"}",
      "$.text",
      "$.nonexistent.embedding"
    );
    HttpEmbeddingInference inference = new HttpEmbeddingInference(config, vertx);

    setupMockWebClientForSuccessfulRequest(inference, "{\"embedding\": [0.1, 0.2, 0.3]}");

    var testObserver = inference.infer("test input").test();

    testObserver.assertNotComplete().assertError(GraviteeInferenceHttpException.class);
  }

  private HttpEmbeddingConfig createValidConfig() {
    return new HttpEmbeddingConfig(
      URI.create("http://localhost:8000/embed/"),
      HttpMethod.POST,
      null,
      "{\"text\": \"<DATA>\"}",
      "$.text",
      "$.embedding"
    );
  }

  private void setupMockWebClientForSuccessfulRequest(HttpEmbeddingInference inference, String jsonResponse) {
    when(mockWebClient.requestAbs(any(), anyString())).thenReturn(mockHttpRequest);
    when(mockHttpRequest.followRedirects(true)).thenReturn(mockHttpRequest);
    when(mockHttpRequest.putHeader(anyString(), anyString())).thenReturn(mockHttpRequest);
    when(mockHttpRequest.rxSendBuffer(any(Buffer.class))).thenReturn(Single.just(mockHttpResponse));
    when(mockHttpResponse.statusCode()).thenReturn(200);
    when(mockHttpResponse.body()).thenReturn(Buffer.buffer(jsonResponse));

    try {
      var webClientField = inference.getClass().getSuperclass().getSuperclass().getDeclaredField("webClient");
      webClientField.setAccessible(true);
      webClientField.set(inference, mockWebClient);
    } catch (Exception e) {
      throw new RuntimeException("Failed to inject mock WebClient", e);
    }
  }
}
