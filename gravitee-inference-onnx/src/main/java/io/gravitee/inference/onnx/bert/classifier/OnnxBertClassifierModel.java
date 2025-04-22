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
package io.gravitee.inference.onnx.bert.classifier;

import static io.gravitee.inference.api.Constants.*;
import static java.lang.String.valueOf;
import static java.util.Comparator.comparing;
import static java.util.Comparator.reverseOrder;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.jni.CharSpan;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtException;
import io.gravitee.inference.api.classifier.ClassifierMode;
import io.gravitee.inference.api.classifier.ClassifierResult;
import io.gravitee.inference.api.classifier.ClassifierResults;
import io.gravitee.inference.onnx.bert.OnnxBertInference;
import io.gravitee.inference.onnx.bert.config.OnnxBertConfig;
import java.util.ArrayList;
import java.util.List;
import java.util.TreeSet;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public class OnnxBertClassifierModel extends OnnxBertInference<ClassifierResults> {

  private final List<String> labels;
  private final List<String> discarded;

  public OnnxBertClassifierModel(OnnxBertConfig config) {
    super(config);
    this.labels = config.get(CLASSIFIER_LABELS, List.of());
    this.discarded = config.get(DISCARDED_LABELS, List.of());
  }

  @Override
  public ClassifierResults infer(String string) {
    return switch (config.<ClassifierMode>get(CLASSIFIER_MODE)) {
      case SEQUENCE -> getSequenceResults(encode(string)).getFirst();
      case TOKEN -> getTokenResults(encode(string)).getFirst();
    };
  }

  @Override
  public List<ClassifierResults> inferAll(List<String> input) {
    return switch (config.<ClassifierMode>get(CLASSIFIER_MODE)) {
      case SEQUENCE -> getSequenceResults(encodeAll(input));
      case TOKEN -> getTokenResults(encodeAll(input));
    };
  }

  private List<ClassifierResults> getTokenResults(EncodingResults encodingResult) {
    var input = this.getTokenLogits(encodingResult.result().get(0));
    var results = new ArrayList<ClassifierResults>(input.batchSize());

    for (int i = 0; i < input.batchSize(); i++) {
      final Encoding encoding = encodingResult.encoding().get(i);

      final String[] tokens = encoding.getTokens();
      final CharSpan[] spans = encoding.getCharTokenSpans();

      float[][] tokenLogits = input.logits()[i];
      var result = new ArrayList<ClassifierResult>();
      for (int j = 1; j < tokens.length - 1; j++) {
        final String sanitizedToken = tokens[j].trim();
        var classifierResult = computeTokenProb(tokenLogits[j], sanitizedToken, spans[j]);
        // we don't want all tokens to be present
        if (!discarded.contains(classifierResult.label())) {
          result.add(classifierResult);
        }
      }
      results.add(new ClassifierResults(result));
    }

    return results;
  }

  private ClassifierResult computeTokenProb(float[] logit, String token, CharSpan span) {
    var result = new TreeSet<>(comparing(ClassifierResult::score, reverseOrder()));
    float[] probabilities = config.gioMath().softmax(logit);
    for (int j = 0; j < probabilities.length; j++) {
      final String label = computeLabel(probabilities, j);
      result.add(new ClassifierResult(label, probabilities[j], token, span.getStart(), span.getEnd()));
    }
    return result.getFirst();
  }

  private List<ClassifierResults> getSequenceResults(EncodingResults encodingResult) {
    var input = this.getSequenceInput(encodingResult.result().get(0));
    var results = new ArrayList<ClassifierResults>(input.batchSize());
    for (int i = 0; i < input.batchSize(); i++) {
      results.add(new ClassifierResults(computeSequenceProb(input.logits()[i])));
    }

    return results;
  }

  private TreeSet<ClassifierResult> computeSequenceProb(float[] logit) {
    var result = new TreeSet<>(comparing(ClassifierResult::score, reverseOrder()));
    float[] probabilities = config.gioMath().sigmoid(logit);
    for (int j = 0; j < probabilities.length; j++) {
      result.add(new ClassifierResult(computeLabel(probabilities, j), probabilities[j]));
    }
    return result;
  }

  private String computeLabel(float[] probabilities, int j) {
    return !labels.isEmpty() && labels.size() == probabilities.length ? labels.get(j) : valueOf(j);
  }

  private SequenceInput getSequenceInput(OnnxValue value) {
    try {
      final float[][] logits = ((float[][]) value.getValue());
      return new SequenceInput(logits.length, logits);
    } catch (OrtException e) {
      throw new IllegalArgumentException(e);
    }
  }

  private TokenInput getTokenLogits(OnnxValue r) {
    try {
      final float[][][] tokens = (float[][][]) r.getValue();
      return new TokenInput(tokens.length, tokens);
    } catch (OrtException e) {
      throw new IllegalArgumentException(e);
    }
  }
}
