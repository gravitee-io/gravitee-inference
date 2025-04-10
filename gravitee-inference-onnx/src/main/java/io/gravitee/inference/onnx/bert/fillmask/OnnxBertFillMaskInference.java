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
package io.gravitee.inference.onnx.bert.fillmask;

import static java.util.Comparator.comparing;
import static java.util.Comparator.reverseOrder;
import static java.util.stream.Collectors.toCollection;
import static java.util.stream.IntStream.range;

import ai.onnxruntime.OrtException;
import io.gravitee.inference.onnx.bert.OnnxBertInference;
import io.gravitee.inference.onnx.bert.config.OnnxBertConfig;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public class OnnxBertFillMaskInference extends OnnxBertInference<List<FillMaskResult<String>>> {

  public static final String MASK_KEYWORD = "[MASK]";

  public OnnxBertFillMaskInference(OnnxBertConfig onnxBertConfig) {
    super(onnxBertConfig);
  }

  @Override
  public List<FillMaskResult<String>> infer(String input) {
    return fillMask(input, encode(input), 0);
  }

  @Override
  public List<List<FillMaskResult<String>>> inferAll(List<String> input) {
    var encodingResults = encodeAll(input);

    return range(0, input.size()).mapToObj(batchSize -> fillMask(input.get(batchSize), encodingResults, batchSize)).toList();
  }

  public List<FillMaskResult<String>> fillMask(String input, EncodingResults encodingResults, int batchInput) {
    try {
      var tokens = tokenizer.tokenize(input);
      int maskedIndex = tokens.indexOf(MASK_KEYWORD);
      return getPredictedWords(getLogits(encodingResults), batchInput, maskedIndex);
    } catch (OrtException e) {
      throw new RuntimeException(e);
    }
  }

  private List<FillMaskResult<String>> getPredictedWords(float[][][] logits, int batchNumber, int maskedIndex) {
    return getTokenIdWithHighestLogits(logits[batchNumber][maskedIndex])
      .stream()
      .map(fm -> new FillMaskResult<>(decodeTokenIds(fm.label()), fm.score()))
      .toList()
      .subList(0, 5);
  }

  private Set<FillMaskResult<Integer>> getTokenIdWithHighestLogits(float[] logits) {
    float[] probabilities = config.gioMath().softmax(logits);
    return range(0, probabilities.length)
      .mapToObj(index -> new FillMaskResult<>(index, probabilities[index]))
      .collect(toCollection(() -> new TreeSet<FillMaskResult<Integer>>(comparing(FillMaskResult::score, reverseOrder()))));
  }

  private String decodeTokenIds(long tokenId) {
    return tokenizer.decode(new long[] { tokenId });
  }

  private static float[][][] getLogits(EncodingResults encode) throws OrtException {
    return (float[][][]) encode.result().get(0).getValue();
  }
}
