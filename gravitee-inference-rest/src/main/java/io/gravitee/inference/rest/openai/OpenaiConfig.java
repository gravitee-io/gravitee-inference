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
package io.gravitee.inference.rest.openai;

import io.gravitee.inference.rest.RestConfig;
import java.net.URI;

public class OpenaiConfig extends RestConfig {

  public String apiKey;
  public String organizationId;
  public String projectId;

  public String model;

  public OpenaiConfig(URI uri, String apiKey, String organizationId, String projectId, String model) {
    super(uri);
    this.apiKey = validateAndGetApiKey(apiKey);
    this.model = validateAndGetModel(model);
    this.organizationId = organizationId;
    this.projectId = projectId;
  }

  public OpenaiConfig(URI uri, String apiKey, String model) {
    this(uri, apiKey, null, null, model);
  }

  private String validateAndGetApiKey(String apiKey) {
    if (apiKey == null || apiKey.trim().isEmpty()) {
      throw new GraviteeOpenaiException("API key cannot be null or blank");
    }
    return apiKey.trim();
  }

  private String validateAndGetModel(String model) {
    if (model == null || model.trim().isEmpty()) {
      throw new GraviteeOpenaiException("Model cannot be null or blank");
    }
    return model.trim();
  }
}
