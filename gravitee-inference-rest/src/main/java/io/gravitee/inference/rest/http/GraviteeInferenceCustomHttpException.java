package io.gravitee.inference.rest.http;

public class GraviteeInferenceCustomHttpException extends RuntimeException {

  public GraviteeInferenceCustomHttpException(String message) {
    super(message);
  }
}
