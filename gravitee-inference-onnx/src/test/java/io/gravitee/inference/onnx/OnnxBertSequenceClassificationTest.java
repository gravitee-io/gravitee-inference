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

import static io.gravitee.inference.api.Constants.CLASSIFIER_LABELS;
import static io.gravitee.inference.api.Constants.CLASSIFIER_MODE;
import static io.gravitee.inference.api.classifier.ClassifierMode.SEQUENCE;
import static org.junit.jupiter.api.Assertions.assertEquals;

import io.gravitee.inference.math.api.GioMaths;
import io.gravitee.inference.math.vanilla.NativeMath;
import io.gravitee.inference.onnx.bert.classifier.OnnxBertClassifierModel;
import io.gravitee.inference.onnx.bert.config.OnnxBertConfig;
import io.gravitee.inference.onnx.bert.resource.OnnxBertResource;
import java.net.URI;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public class OnnxBertSequenceClassificationTest extends OnnxBertBaseTest {

  private static final String HF_DISTILBERT_SEQUENCE_PATH =
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/onnx/";
  private static final String HF_DISTILBERT_ONNX_DOWNLOAD = HF_URL + HF_DISTILBERT_SEQUENCE_PATH + "model.onnx";
  private static final String HF_DISTILBERT_TOKENIZER = HF_URL + HF_DISTILBERT_SEQUENCE_PATH + "tokenizer.json";
  private static final String HF_DISTILBERT_CONFIG_JSON = HF_URL + HF_DISTILBERT_SEQUENCE_PATH + "config.json";

  private static final String SEQUENCE_TOKENIZER_JSON = "distilbert/tokenizer.json";
  private static final String SEQUENCE_MODEL_ONNX = "distilbert/model.onnx";
  private static final String SEQUENCE_CONFIG_JSON = "distilbert/config.json";

  private static final URI SEQ_HF_TOKENIZER = getUriIfExist(SEQUENCE_TOKENIZER_JSON, HF_DISTILBERT_TOKENIZER);
  private static final URI SEQ_MODEL_ONNX = getUriIfExist(SEQUENCE_MODEL_ONNX, HF_DISTILBERT_ONNX_DOWNLOAD);
  private static final URI SEQ_MODEL_CONFIG_JSON = getUriIfExist(SEQUENCE_CONFIG_JSON, HF_DISTILBERT_CONFIG_JSON);

  private static final OnnxBertResource SEQUENCE_ONNX_BERT_RESOURCE = new OnnxBertResource(
    Path.of(SEQ_MODEL_ONNX),
    Path.of(SEQ_HF_TOKENIZER)
  );

  private static final OnnxBertResource SEQUENCE_ONNX_BERT_RESOURCE_WITH_CONFIG = new OnnxBertResource(
    Path.of(SEQ_MODEL_ONNX),
    Path.of(SEQ_HF_TOKENIZER),
    Path.of(SEQ_MODEL_CONFIG_JSON)
  );

  public static Stream<Arguments> params_that_must_sequence_classify_and_return_first_highest_score() {
    return Stream.of(
      Arguments.of(
        SEQUENCE_ONNX_BERT_RESOURCE,
        NativeMath.INSTANCE,
        Map.of(CLASSIFIER_MODE, SEQUENCE, CLASSIFIER_LABELS, List.of("NEGATIVE", "POSITIVE"))
      ),
      Arguments.of(SEQUENCE_ONNX_BERT_RESOURCE_WITH_CONFIG, NativeMath.INSTANCE, Map.of(CLASSIFIER_MODE, SEQUENCE))
    );
  }

  @ParameterizedTest
  @MethodSource("params_that_must_sequence_classify_and_return_first_highest_score")
  public void must_sequence_classify_and_return_first_highest_score(
    OnnxBertResource resource,
    GioMaths maths,
    Map<String, Object> config
  ) {
    var sequenceModel = new OnnxBertClassifierModel(new OnnxBertConfig(resource, maths, config));
    var positive = sequenceModel.infer("I am so happy!").results().stream().toList();
    assertEquals(2, positive.size());
    assertEquals("POSITIVE", positive.getFirst().label());
    assertEquals(0.9905912280082703, positive.getFirst().score());

    var negative = sequenceModel.infer("I am so sad!").results().stream().toList();
    assertEquals(2, negative.size());
    assertEquals("NEGATIVE", negative.getFirst().label());
    assertEquals(0.9839373826980591, negative.getFirst().score());

    sequenceModel.close();
  }

  @ParameterizedTest
  @MethodSource("params_that_must_sequence_classify_and_return_first_highest_score")
  public void must_sequence_classify_and_return_first_highest_score_for_both_sentences(
    OnnxBertResource resource,
    GioMaths maths,
    Map<String, Object> config
  ) {
    var sequenceModel = new OnnxBertClassifierModel(new OnnxBertConfig(resource, maths, config));

    var sentiments = sequenceModel.inferAll(List.of("I am so happy!", "I am so sad!"));
    assertEquals(2, sentiments.size());

    var positive = sentiments.getFirst().results().stream().toList();
    assertEquals(2, positive.size());
    assertEquals("POSITIVE", positive.getFirst().label());
    assertEquals(0.9905912280082703, positive.getFirst().score());

    var negative = sentiments.getLast().results().stream().toList();
    assertEquals(2, negative.size());
    assertEquals("NEGATIVE", negative.getFirst().label());
    assertEquals(0.9839373826980591, negative.getFirst().score());

    sequenceModel.close();
  }
}
