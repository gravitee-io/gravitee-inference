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
package io.gravitee.inference.onnx.bert.embedding;

import static io.gravitee.inference.api.Constants.MAX_SEQUENCE_LENGTH;
import static io.gravitee.inference.api.Constants.POOLING_MODE;
import static java.lang.Math.max;
import static java.lang.Math.min;
import static java.lang.System.arraycopy;
import static java.util.stream.IntStream.iterate;

import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession.Result;
import io.gravitee.inference.api.embedding.EmbeddingTokenCount;
import io.gravitee.inference.api.embedding.PoolingMode;
import io.gravitee.inference.onnx.bert.OnnxBertInference;
import io.gravitee.inference.onnx.bert.config.OnnxBertConfig;
import java.util.List;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public class OnnxBertEmbeddingModel
  extends OnnxBertInference<EmbeddingTokenCount> {

  public OnnxBertEmbeddingModel(OnnxBertConfig onnxBertConfig) {
    super(onnxBertConfig);
  }

  @Override
  public EmbeddingTokenCount infer(String input) {
    var tokens = tokenizer.tokenize(input);
    var embeddings = getEmbeddings(input);

    return new EmbeddingTokenCount(
      embeddings.toNormalizedWeighted(config.gioMath()),
      tokens.size()
    );
  }

  private EmbeddingsWithWeights getEmbeddings(String input) {
    final int partitionSize = config.get(MAX_SEQUENCE_LENGTH);

    long[] ids = tokenizer.encode(input, true, false).getIds();
    final int lastIndex = ids.length - 1; // index of the [SEP] token
    final long clsId = ids[0];
    final long sepId = ids[lastIndex];

    final int contentSize = max(lastIndex - 1, 0); // tokens without [CLS]/[SEP]
    final int nbPartitions = max(
      (contentSize + partitionSize - 1) / partitionSize,
      1
    );
    var weights = new float[nbPartitions];
    var embeddings = new float[nbPartitions][];

    // we ignore first token (CLS) and last token (SEP)
    iterate(1, from -> from < lastIndex, from -> from + partitionSize).forEach(
      from -> {
        int to = min(lastIndex, from + partitionSize);
        int index = (from - 1) / partitionSize;
        try (var partition = encode(chunk(ids, clsId, sepId, from, to))) {
          embeddings[index] = toEmbedding(partition);
          weights[index] = (float) (to - from);
        }
      }
    );

    return new EmbeddingsWithWeights(embeddings, weights);
  }

  /**
   * Builds a standalone partition by re-wrapping the token id slice
   * {@code [from, to)} with the [CLS] and [SEP] special tokens.
   */
  private static long[] chunk(
    long[] ids,
    long clsId,
    long sepId,
    int from,
    int to
  ) {
    long[] partition = new long[(to - from) + 2];
    partition[0] = clsId;
    arraycopy(ids, from, partition, 1, to - from);
    partition[partition.length - 1] = sepId;
    return partition;
  }

  private float[] toEmbedding(Result result) {
    try {
      return pool(result);
    } catch (OrtException e) {
      throw new IllegalArgumentException(e);
    }
  }

  private float[] pool(Result result) throws OrtException {
    float[][] vectors = ((float[][][]) result.get(0).getValue())[0];
    return switch (config.<PoolingMode>get(POOLING_MODE)) {
      case CLS -> vectors[0];
      case MEAN -> config.gioMath().mean(vectors);
    };
  }
}
