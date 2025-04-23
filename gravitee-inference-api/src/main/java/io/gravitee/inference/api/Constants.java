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
package io.gravitee.inference.api;

import static java.util.Collections.singletonMap;

import java.util.Map;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class Constants {

  public static final int MAX_SEQUENCE_LENGTH_DEFAULT_VALUE = 510; // 512 - 2 (special tokens [CLS] and [SEP])

  private Constants() {}

  public static final String INPUT = "input";

  /*
    Resource Constants
  */
  public static final String MODEL_PATH = "modelPath";
  public static final String TOKENIZER_PATH = "tokenizerPath";
  public static final String CONFIG_JSON_PATH = "configJsonPath";

  /*
    ONNX Constants
  */

  public static final String INPUT_IDS = "input_ids";
  public static final String ATTENTION_MASK = "attention_mask";
  public static final String TOKEN_TYPE_IDS = "token_type_ids";

  /*
    Classifier Constants
  */

  public static final String CLASSIFIER_MODE = "mode";
  public static final String CLASSIFIER_LABELS = "labels";
  public static final String DISCARDED_LABELS = "discardedLabels";

  /*
   Embedding Constants
  */

  public static final String POOLING_MODE = "poolingMode";

  public static final String PADDING = "padding";
  public static final Map<String, String> DEFAULT_TOKENIZER_CONFIG = singletonMap(PADDING, "false");
  public static final String MAX_SEQUENCE_LENGTH = "maxSequenceLength";

  /*
   Inference Service
  */

  public static final String SERVICE_INFERENCE_MODELS_ADDRESS = "service:inference:models";
  public static final String SERVICE_INFERENCE_MODELS_INFER_TEMPLATE = "service:inference:models:%s";
  public static final String INFERENCE_TYPE = "inferenceType";
  public static final String INFERENCE_FORMAT = "inferenceFormat";
  public static final String MODEL_ADDRESS_KEY = "modelAddress";
}
