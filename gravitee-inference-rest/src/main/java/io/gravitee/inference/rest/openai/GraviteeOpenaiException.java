package io.gravitee.inference.rest.openai;

import io.gravitee.inference.rest.GraviteeRestException;

public class GraviteeOpenaiException extends GraviteeRestException {
  public GraviteeOpenaiException(String message) {
    super(message);
  }
}
