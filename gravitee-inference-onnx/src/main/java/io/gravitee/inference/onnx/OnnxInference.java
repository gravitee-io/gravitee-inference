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
package io.gravitee.inference.onnx;

import static ai.onnxruntime.OrtEnvironment.getAvailableProviders;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtProvider;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.SessionOptions;
import io.gravitee.inference.api.InferenceModel;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public abstract class OnnxInference<C extends OnnxConfig<?>, I, O> extends InferenceModel<C, I, O> {

  protected final OrtEnvironment environment;
  protected final OrtSession session;

  protected OnnxInference(C config) {
    super(config);
    this.environment = OrtEnvironment.getEnvironment();
    this.session = getSession();
  }

  private OrtSession getSession() {
    try {
      OrtSession.SessionOptions options = new SessionOptions();
      if (getAvailableProviders().contains(OrtProvider.CUDA)) {
        options.addCUDA();
      }
      final String modelPath = config.getResource().getModel().toAbsolutePath().toString();
      return environment.createSession(modelPath, options);
    } catch (OrtException e) {
      throw new IllegalArgumentException(e);
    }
  }

  public void close() {
    try {
      this.session.close();
      this.environment.close();
    } catch (OrtException e) {
      throw new RuntimeException(e);
    }
  }
}
