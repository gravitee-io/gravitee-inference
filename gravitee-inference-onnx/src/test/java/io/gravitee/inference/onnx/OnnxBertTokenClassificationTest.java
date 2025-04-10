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

import static io.gravitee.inference.api.Constants.*;
import static io.gravitee.inference.api.classifier.ClassifierMode.TOKEN;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.onnxruntime.OrtException;
import io.gravitee.inference.api.classifier.ClassifierResults;
import io.gravitee.inference.math.vanilla.NativeMath;
import io.gravitee.inference.onnx.bert.classifier.OnnxBertClassifierModel;
import io.gravitee.inference.onnx.bert.config.OnnxBertConfig;
import io.gravitee.inference.onnx.bert.resource.OnnxBertResource;
import java.net.URI;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public class OnnxBertTokenClassificationTest extends OnnxBertBaseTest {

  protected static final String TOKEN_TOKENIZER_JSON = "bert-NER/tokenizer.json";
  protected static final String TOKEN_MODEL_ONNX = "bert-NER/model.onnx";

  private static final String HF_DISTILBERT_NER_TOKEN_PATH = "dslim/bert-base-NER/resolve/main/onnx/";

  protected static final String HF_DISTILBERT_NER_ONNX_DOWNLOAD = HF_URL + HF_DISTILBERT_NER_TOKEN_PATH + "model.onnx";
  protected static final String HF_DISTILBERT_NER_TOKENIZER = HF_URL + HF_DISTILBERT_NER_TOKEN_PATH + "tokenizer.json";

  private static final URI TKN_MODEL_ONNX = getUriIfExist(TOKEN_MODEL_ONNX, HF_DISTILBERT_NER_ONNX_DOWNLOAD);
  private static final URI TKN_HF_TOKENIZER = getUriIfExist(TOKEN_TOKENIZER_JSON, HF_DISTILBERT_NER_TOKENIZER);

  private static final OnnxBertResource TOKEN_ONNX_BERT_RESOURCE = new OnnxBertResource(
    Path.of(TKN_MODEL_ONNX),
    Path.of(TKN_HF_TOKENIZER)
  );

  public static final List<String> LABELS = List.of(
    "O",
    "B-MISC",
    "I-MISC",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC"
  );

  private static final OnnxBertClassifierModel TOKEN_MODEL = new OnnxBertClassifierModel(
    new OnnxBertConfig(
      TOKEN_ONNX_BERT_RESOURCE,
      NativeMath.INSTANCE,
      Map.of(CLASSIFIER_MODE, TOKEN, CLASSIFIER_LABELS, LABELS, DISCARDED_LABELS, List.of("O", "B-MISC", "I-MISC"))
    )
  );

  @Test
  public void must_classify_tokens_in_sentence() {
    var results = TOKEN_MODEL.infer("My name is Laura and I live in Houston, Texas").results().stream().toList();

    assertEquals(3, results.size());

    var token1 = results.getFirst();
    assertEquals("Laura", token1.token());
    assertEquals("B-PER", token1.label());
    assertTrue(token1.score() >= 0.95);
    assertEquals(11, token1.start());
    assertEquals(16, token1.end());

    var token2 = results.get(1);
    assertEquals("Houston", token2.token());
    assertEquals("B-LOC", token2.label());
    assertTrue(token2.score() >= 0.95);
    assertEquals(31, token2.start());
    assertEquals(38, token2.end());

    var token3 = results.getLast();
    assertEquals("Texas", token3.token());
    assertEquals("B-LOC", token3.label());
    assertTrue(token3.score() >= 0.95);
    assertEquals(40, token3.start());
    assertEquals(45, token3.end());
  }

  @Test
  public void must_classify_tokens_in_sentences() {
    var tokensList = TOKEN_MODEL
      .inferAll(
        List.of("My name is Laura and I live in Houston, Texas", "My name is Clara and I live in Berkley, California")
      )
      .stream()
      .map(ClassifierResults::results)
      .toList();

    assertEquals(2, tokensList.size());

    var firstSentence = tokensList.getFirst().stream().toList();
    assertEquals(3, firstSentence.size());

    var token1 = firstSentence.getFirst();
    assertEquals("Laura", token1.token());
    assertEquals("B-PER", token1.label());
    assertTrue(token1.score() >= 0.95);
    assertEquals(11, token1.start());
    assertEquals(16, token1.end());

    var token2 = firstSentence.get(1);
    assertEquals("Houston", token2.token());
    assertEquals("B-LOC", token2.label());
    assertTrue(token2.score() >= 0.95);
    assertEquals(31, token2.start());
    assertEquals(38, token2.end());

    var token3 = firstSentence.getLast();
    assertEquals("Texas", token3.token());
    assertEquals("B-LOC", token3.label());
    assertTrue(token3.score() >= 0.95);
    assertEquals(40, token3.start());
    assertEquals(45, token3.end());

    var secondSentence = tokensList.getLast().stream().toList();
    assertEquals(5, secondSentence.size());
    token1 = secondSentence.getFirst();
    assertEquals("Clara", token1.token());
    assertEquals("B-PER", token1.label());
    assertTrue(token1.score() >= 0.95);
    assertEquals(11, token1.start());
    assertEquals(16, token1.end());

    token2 = secondSentence.get(1);
    assertEquals("Be", token2.token());
    assertEquals("B-LOC", token2.label());
    assertTrue(token2.score() >= 0.95);
    assertEquals(31, token2.start());
    assertEquals(33, token2.end());

    token2 = secondSentence.get(2);
    assertEquals("##rk", token2.token());
    assertEquals("I-LOC", token2.label());
    assertTrue(token2.score() >= 0.85 && token2.score() <= 0.86);
    assertEquals(33, token2.start());
    assertEquals(35, token2.end());

    token2 = secondSentence.get(3);
    assertEquals("##ley", token2.token());
    assertEquals("I-LOC", token2.label());
    assertTrue(token2.score() >= 0.95);
    assertEquals(35, token2.start());
    assertEquals(38, token2.end());

    token2 = secondSentence.getLast();
    assertEquals("California", token2.token());
    assertEquals("B-LOC", token2.label());
    assertTrue(token2.score() >= 0.95);
    assertEquals(40, token2.start());
    assertEquals(50, token2.end());
  }

  @AfterAll
  public static void tearDown() {
    TOKEN_MODEL.close();
  }
}
