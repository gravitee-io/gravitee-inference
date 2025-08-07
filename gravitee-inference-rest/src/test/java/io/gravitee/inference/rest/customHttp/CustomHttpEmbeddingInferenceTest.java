package io.gravitee.inference.rest.customHttp;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

import io.gravitee.inference.rest.customHttp.embedding.CustomHttpEmbeddingConfig;
import io.gravitee.inference.rest.customHttp.embedding.CustomHttpEmbeddingInference;
import io.reactivex.rxjava3.core.Single;
import io.vertx.core.http.HttpMethod;
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

  @ParameterizedTest
  @CsvFileSource(resources = "/embedding-test-data.csv", numLinesToSkip = 1)
  void testEmbeddingInferenceWithJsonPath(
          String inputLocation,
          String requestBodyTemplate,
          String jsonResponse,
          String outputJsonPath,
          int expectedEmbeddingLength,
          String description
  ) {
    CustomHttpEmbeddingConfig config = new CustomHttpEmbeddingConfig(
            URI.create("http://localhost:8000/embed/"),
            HttpMethod.POST,
            Map.of("Content-Type", "application/json"),
            "application/json",
            requestBodyTemplate,
            inputLocation,
            outputJsonPath
    );

    CustomHttpEmbeddingInference inference = new CustomHttpEmbeddingInference(config, vertx);

    when(mockWebClient.requestAbs(config.getMethod(), config.getUri().toString())).thenReturn(mockHttpRequest);
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

    String inputText = "Test input text for " + description;

    var testObserver = inference.infer(inputText).test();

    testObserver
            .assertComplete()
            .assertNoErrors()
            .assertValue(embeddingTokenCount -> expectedEmbeddingLength == embeddingTokenCount.embedding().length);

    verify(mockHttpRequest, times(2)).putHeader("Content-Type", "application/json");
    verify(mockHttpRequest).rxSendBuffer(any(Buffer.class));
  }
}
