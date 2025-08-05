package io.gravitee.inference.rest.openai;

import io.gravitee.inference.rest.openai.embedding.OpenAIEmbeddingConfig;
import io.gravitee.inference.rest.openai.embedding.OpenaiEmbeddingInference;
import io.reactivex.rxjava3.core.Single;
import io.vertx.core.json.JsonArray;
import io.vertx.core.json.JsonObject;
import io.vertx.rxjava3.core.Vertx;
import io.vertx.rxjava3.core.buffer.Buffer;
import io.vertx.rxjava3.ext.web.client.HttpRequest;
import io.vertx.rxjava3.ext.web.client.HttpResponse;
import io.vertx.rxjava3.ext.web.client.WebClient;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.mockito.junit.jupiter.MockitoExtension;

import java.net.URI;
import java.net.URISyntaxException;

import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;


@ExtendWith(MockitoExtension.class)
public class OpenaiEmbeddingInferenceTest {
  public static final String TEST_URL = "http://somewhere.at.gravitee";
  public static final String TEST_MODEL_NAME = "all-minilm";
  public static final String TEST_APIKEY = "TEST_KEY";

  public static final String OPENAI_API_ENTRYPOINT = "/v1";

  @Mock
  private Vertx vertx;

  @Mock
  private WebClient webClient;

  @Mock
  private HttpRequest<Buffer> httpRequest;

  @Mock
  private HttpResponse<Buffer> httpResponse;

  public static final String THE_BIG_BROWN_FOX_JUMPED_OVER_THE_LAZY_DOG = "The big brown fox jumped over the lazy dog";

  @BeforeEach
  public void setUp() {
    when(webClient.postAbs(anyString())).thenReturn(httpRequest);
    when(httpRequest.bearerTokenAuthentication(anyString())).thenReturn(httpRequest);
    when(httpRequest.putHeader(anyString(), anyString())).thenReturn(httpRequest);
    lenient().when(httpRequest.rxSendBuffer(any(Buffer.class))).thenReturn(Single.just(httpResponse));
  }

  @AfterEach
  public void afterEach() {
  }

  @Test
  public void must_return_embedding() throws URISyntaxException {
    var config = getTestConfig();

    JsonObject mockResponse = new JsonObject()
            .put("object", "list")
            .put("data", new JsonArray()
                    .add(new JsonObject()
                            .put("object", "embedding")
                            .put("index", 0)
                            .put("embedding", new JsonArray()
                                    .add(0.1f).add(0.2f).add(0.3f) // Sample embedding values
                            )
                    )
            )
            .put("model", TEST_MODEL_NAME)
            .put("usage", new JsonObject()
                    .put("prompt_tokens", 10)
                    .put("total_tokens", 10)
            );

    when(httpResponse.statusCode()).thenReturn(200);
    when(httpResponse.body()).thenReturn(Buffer.buffer(mockResponse.encode()));

    try (MockedStatic<WebClient> webClientMock = Mockito.mockStatic(WebClient.class)) {
      webClientMock.when(() -> WebClient.create(any(Vertx.class))).thenReturn(webClient);

      var client = new OpenaiEmbeddingInference(config, vertx);
      var testObserver = client.infer(THE_BIG_BROWN_FOX_JUMPED_OVER_THE_LAZY_DOG).test();

      testObserver
              .assertComplete()
              .assertNoErrors()
              .assertValue(embeddingTokenCount -> {
                var correct_length = embeddingTokenCount.embedding().length == 3;
                var correct_token_number = embeddingTokenCount.tokenCount() == 10;

                return correct_length && correct_token_number;
              })
              .assertValueCount(1);

      verify(webClient).postAbs(getTestURI() + "/embeddings");
      verify(httpRequest).bearerTokenAuthentication(TEST_APIKEY);
      verify(httpRequest).putHeader("Content-Type", "application/json");
    }
  }

  @Test
  public void must_handle_http_error() throws URISyntaxException {
    var config = getTestConfig();

    when(httpResponse.statusCode()).thenReturn(401);
    when(httpResponse.bodyAsString()).thenReturn("Unauthorized");

    try (MockedStatic<WebClient> webClientMock = Mockito.mockStatic(WebClient.class)) {
      webClientMock.when(() -> WebClient.create(any(Vertx.class))).thenReturn(webClient);

      var client = new OpenaiEmbeddingInference(config, vertx);
      var testObserver = client.infer(THE_BIG_BROWN_FOX_JUMPED_OVER_THE_LAZY_DOG).test();

      testObserver
              .assertNotComplete()
              .assertError(throwable -> throwable.getMessage().contains("401") ||
                      throwable.getMessage().contains("Unauthorized"));
    }
  }

  @Test
  public void must_handle_network_failure() throws URISyntaxException {
    var config = getTestConfig();

    RuntimeException networkError = new RuntimeException("Connection refused");
    when(httpRequest.rxSendBuffer(any(Buffer.class)))
            .thenReturn(Single.error(networkError));

    try (MockedStatic<WebClient> webClientMock = Mockito.mockStatic(WebClient.class)) {
      webClientMock.when(() -> WebClient.create(any(Vertx.class))).thenReturn(webClient);

      var client = new OpenaiEmbeddingInference(config, vertx);
      var testObserver = client.infer(THE_BIG_BROWN_FOX_JUMPED_OVER_THE_LAZY_DOG).test();

      testObserver
              .assertNotComplete()
              .assertError(RuntimeException.class)
              .assertError(throwable -> throwable.getMessage().contains("Connection refused"));
    }
  }

  private static OpenAIEmbeddingConfig getTestConfig() throws URISyntaxException {
    return new OpenAIEmbeddingConfig(
            new URI(getTestURI()),
            TEST_APIKEY,
            TEST_MODEL_NAME
    );
  }

  private static String getTestURI() {
    return TEST_URL + OPENAI_API_ENTRYPOINT;
  }
}
