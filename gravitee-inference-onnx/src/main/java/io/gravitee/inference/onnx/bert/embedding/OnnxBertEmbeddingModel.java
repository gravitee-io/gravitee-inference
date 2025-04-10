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
import static java.lang.Math.min;
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
public class OnnxBertEmbeddingModel extends OnnxBertInference<EmbeddingTokenCount> {

  public OnnxBertEmbeddingModel(OnnxBertConfig onnxBertConfig) {
    super(onnxBertConfig);
  }

  @Override
  public EmbeddingTokenCount infer(String input) {
    var tokens = tokenizer.tokenize(input);
    var embeddings = getEmbeddings(tokens);

    return new EmbeddingTokenCount(embeddings.toNormalizedWeighted(config.gioMath()), tokens.size());
  }

  private EmbeddingsWithWeights getEmbeddings(List<String> tokens) {
    final int partitionSize = config.get(MAX_SEQUENCE_LENGTH);
    final int lastIndex = tokens.size() - 1;

    final int nbPartitions = (tokens.size() / partitionSize) + 1;
    var weights = new float[nbPartitions];
    var embeddings = new float[nbPartitions][];

    // we ignore first token (CLS) and last token (SEP)
    iterate(1, from -> from < lastIndex, from -> from + partitionSize)
      .forEach(from -> {
        int to = min(lastIndex, from + partitionSize);
        var partition = encode(String.join("", tokens.subList(from, to))).result();
        embeddings[from - 1] = toEmbedding(partition);
        weights[from - 1] = partition.size();
      });

    return new EmbeddingsWithWeights(embeddings, weights);
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
