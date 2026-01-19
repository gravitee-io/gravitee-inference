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
package io.gravitee.inference.api.textgen;

/**
 * Media types for image and audio content.
 * Follows IANA media type registry conventions.
 */
public enum MediaType {
  // Image types
  IMAGE_JPEG("image/jpeg"),
  IMAGE_PNG("image/png"),
  IMAGE_GIF("image/gif"),
  IMAGE_WEBP("image/webp"),
  IMAGE_BMP("image/bmp"),
  IMAGE_TIFF("image/tiff"),

  // Audio types
  AUDIO_WAV("audio/wav"),
  AUDIO_MP3("audio/mpeg"),
  AUDIO_OGG("audio/ogg"),
  AUDIO_FLAC("audio/flac"),
  AUDIO_AAC("audio/aac"),
  AUDIO_M4A("audio/mp4"),

  // Generic binary
  APPLICATION_OCTET_STREAM("application/octet-stream");

  private final String value;

  MediaType(String value) {
    this.value = value;
  }

  public String value() {
    return value;
  }

  public static MediaType fromString(String value) {
    for (MediaType type : MediaType.values()) {
      if (type.value.equals(value)) {
        return type;
      }
    }
    return APPLICATION_OCTET_STREAM;
  }
}
