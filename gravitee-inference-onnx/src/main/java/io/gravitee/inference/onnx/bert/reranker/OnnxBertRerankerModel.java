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
package io.gravitee.inference.onnx.bert.reranker;

import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession.Result;
import io.gravitee.inference.api.reranker.RerankPair;
import io.gravitee.inference.api.reranker.RerankScoring;
import io.gravitee.inference.api.reranker.RerankTokenCount;
import io.gravitee.inference.onnx.bert.OnnxBertInference;
import io.gravitee.inference.onnx.bert.config.OnnxBertConfig;
import java.util.ArrayList;
import java.util.List;

/**
 * Cross-encoder reranker model backed by a BERT classification head (e.g. BAAI/bge-reranker-v2-m3).
 *
 * <p>A cross-encoder takes a {@code (query, document)} pair as a single sequence and emits
 * a classification logit directly. It does not produce a pooled hidden-state vector, so it
 * cannot be used as an embedder. The input type is {@link RerankPair} (structured pair),
 * avoiding {@code UnsupportedOperationException}-style workarounds.
 *
 * <p>The inherited {@link #inferAll(List)} is overridden to batch all pairs that share
 * the same query into a single padded forward pass — matching the common rerank use case
 * of "one query, many documents". Pairs with different queries fall back to the default
 * sequential iteration.
 *
 * <p>Auto-detects output shape: {@code [batch, 1]} → SIGMOID default; {@code [batch, 2]} →
 * SOFTMAX default. Other shapes are rejected.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public class OnnxBertRerankerModel extends OnnxBertInference<RerankPair, RerankTokenCount> {

  private final RerankScoring scoring;

  public OnnxBertRerankerModel(OnnxBertConfig config, RerankScoring scoring) {
    super(config);
    this.scoring = scoring;
  }

  /** Convenience constructor: {@code null} scoring = auto-default based on output shape. */
  public OnnxBertRerankerModel(OnnxBertConfig config) {
    this(config, null);
  }

  @Override
  public RerankTokenCount infer(RerankPair input) {
    var enc = encodePair(input.query(), input.document());
    try (Result result = enc.result()) {
      float[] scores = extractScores(result);
      int tokens = enc.encoding().get(0).getIds().length;
      return new RerankTokenCount(scores[0], tokens);
    } catch (OrtException e) {
      throw new IllegalArgumentException(e);
    }
  }

  /**
   * Batched inference for reranking.
   *
   * <p>Optimised path: when all inputs share the same query, they are encoded together
   * as a single padded batch and sent to ONNX in one forward pass (the common case).
   *
   * <p>Fallback path: when queries differ across inputs, we fall back to sequential
   * single-pair inference.
   */
  @Override
  public List<RerankTokenCount> inferAll(List<RerankPair> input) {
    if (input == null || input.isEmpty()) {
      return List.of();
    }

    String query = input.getFirst().query();
    boolean sameQuery = input.stream().allMatch(p -> query.equals(p.query()));

    if (!sameQuery) {
      return input.stream().map(this::infer).toList();
    }

    List<String> documents = input.stream().map(RerankPair::document).toList();

    var enc = encodeAllPairs(query, documents);
    try (Result result = enc.result()) {
      float[] scores = extractScores(result);
      List<RerankTokenCount> out = new ArrayList<>(scores.length);
      for (int i = 0; i < scores.length; i++) {
        int tokens = enc.encoding().get(i).getIds().length;
        out.add(new RerankTokenCount(scores[i], tokens));
      }
      return out;
    } catch (OrtException e) {
      throw new IllegalArgumentException(e);
    }
  }

  private float[] extractScores(Result result) throws OrtException {
    Object raw = result.get(0).getValue();
    if (!(raw instanceof float[][] logits)) {
      throw new IllegalArgumentException(
        "Reranker output must be [batch, N] float32 — got: " + raw.getClass().getSimpleName()
      );
    }
    if (logits.length == 0) return new float[0];
    int numClasses = logits[0].length;
    if (numClasses != 1 && numClasses != 2) {
      throw new IllegalArgumentException("Reranker output must have 1 or 2 classes per row, got: " + numClasses);
    }

    RerankScoring mode = scoring != null ? scoring : (numClasses == 1 ? RerankScoring.SIGMOID : RerankScoring.SOFTMAX);

    return switch (mode) {
      case SIGMOID -> handleSigmoid(logits, numClasses);
      case SOFTMAX -> handleSoftmax(logits, numClasses);
      case LOGIT -> handleLogit(logits, numClasses);
    };
  }

  private float[] handleSigmoid(float[][] logits, int numClasses) {
    float[] out = new float[logits.length];
    int col = (numClasses == 1) ? 0 : 1;
    for (int i = 0; i < logits.length; i++) {
      out[i] = logits[i][col];
    }
    return config.gioMath().sigmoid(out);
  }

  private float[] handleSoftmax(float[][] logits, int numClasses) {
    float[] out = new float[logits.length];
    if (numClasses != 2) {
      throw new IllegalArgumentException("SOFTMAX scoring requires [batch, 2] output, got " + numClasses);
    }
    for (int i = 0; i < logits.length; i++) {
      out[i] = config.gioMath().softmax(logits[i])[1];
    }
    return out;
  }

  private static float[] handleLogit(float[][] logits, int numClasses) {
    float[] out = new float[logits.length];
    for (int i = 0; i < logits.length; i++) {
      out[i] = (numClasses == 1) ? logits[i][0] : logits[i][1] - logits[i][0];
    }
    return out;
  }
}
