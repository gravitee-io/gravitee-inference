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
package io.gravitee.inference.onnx.bert;

import static ai.onnxruntime.OnnxTensor.createTensor;
import static io.gravitee.inference.api.Constants.*;
import static java.lang.System.arraycopy;
import static java.nio.LongBuffer.wrap;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession.Result;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.gravitee.inference.onnx.OnnxInference;
import io.gravitee.inference.onnx.bert.config.OnnxBertConfig;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Set;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public abstract class OnnxBertInference<OUTPUT> extends OnnxInference<OnnxBertConfig, String, OUTPUT> {

  protected static final ObjectMapper objectMapper = new ObjectMapper();
  protected final HuggingFaceTokenizer tokenizer;
  protected final JsonNode configJson;
  private boolean hasTokenTypeIds;

  protected OnnxBertInference(OnnxBertConfig onnxBertConfig) {
    super(onnxBertConfig);
    this.tokenizer = getTokenizer();
    this.configJson = getConfigJson();
    hasTokenTypeIds = session.getInputNames().contains(TOKEN_TYPE_IDS);
  }

  private HuggingFaceTokenizer getTokenizer() {
    try {
      return HuggingFaceTokenizer.newInstance(
        config.getResource().getTokenizer().toAbsolutePath(),
        config.getTokenizerConfig()
      );
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  private JsonNode getConfigJson() {
    try {
      Path configJson = this.config.getResource().getConfigJson();
      return configJson == null ? null : objectMapper.readTree(String.join("", Files.readAllLines(configJson)));
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  protected EncodingResults encode(String sentence) {
    var encoding = tokenizer.encode(sentence, true, false);

    long[] inputIds = encoding.getIds();
    long[] attentionMask = encoding.getAttentionMask();

    long[] shape = { 1, inputIds.length };

    try (
      var inputIdsTensor = createTensor(environment, wrap(inputIds), shape);
      var attentionMaskTensor = createTensor(environment, wrap(attentionMask), shape);
    ) {
      var inputs = new HashMap<String, OnnxTensor>();
      inputs.put(INPUT_IDS, inputIdsTensor);
      inputs.put(ATTENTION_MASK, attentionMaskTensor);

      if (hasTokenTypeIds) {
        var tokenTypeIdsTensor = createTensor(environment, wrap(encoding.getTypeIds()), shape);
        inputs.put(TOKEN_TYPE_IDS, tokenTypeIdsTensor);
      }
      return new EncodingResults(List.of(encoding), session.run(inputs));
    } catch (OrtException e) {
      throw new IllegalArgumentException(e);
    }
  }

  protected EncodingResults encodeAll(List<String> sentences) {
    List<Encoding> encodings = new ArrayList<>(sentences.size());
    int maxTokens = 0;

    for (String sentence : sentences) {
      var encoding = tokenizer.encode(sentence, true, false);
      maxTokens += encoding.getIds().length;
      encodings.add(encoding);
    }

    long[] inputIds = new long[sentences.size() * maxTokens];
    long[] attentionMask = new long[sentences.size() * maxTokens];
    long[] tokenTypeIds = new long[sentences.size() * maxTokens];

    for (int i = 0; i < sentences.size(); i++) {
      Encoding encoding = encodings.get(i);

      // Retrieve the tokens for the current sentence
      long[] sentenceInputIds = encoding.getIds();
      long[] sentenceAttentionMask = encoding.getAttentionMask();
      long[] sentenceTokenTypeIds = encoding.getTypeIds();

      int startIndex = i * maxTokens;
      arraycopy(sentenceInputIds, 0, inputIds, startIndex, sentenceInputIds.length);
      arraycopy(sentenceAttentionMask, 0, attentionMask, startIndex, sentenceAttentionMask.length);
      if (hasTokenTypeIds) {
        arraycopy(sentenceTokenTypeIds, 0, tokenTypeIds, startIndex, sentenceTokenTypeIds.length);
      }
    }

    long[] shape = { sentences.size(), maxTokens };

    try (
      var inputIdsTensor = createTensor(environment, wrap(inputIds), shape);
      var attentionMaskTensor = createTensor(environment, wrap(attentionMask), shape);
    ) {
      var inputs = new HashMap<String, OnnxTensor>();
      inputs.put(INPUT_IDS, inputIdsTensor);
      inputs.put(ATTENTION_MASK, attentionMaskTensor);

      if (session.getInputNames().contains(TOKEN_TYPE_IDS)) {
        var tokenTypeIdsTensor = createTensor(environment, wrap(tokenTypeIds), shape);
        inputs.put(TOKEN_TYPE_IDS, tokenTypeIdsTensor);
      }
      return new EncodingResults(encodings, session.run(inputs));
    } catch (OrtException e) {
      throw new IllegalArgumentException(e);
    }
  }

  public record EncodingResults(List<Encoding> encoding, Result result) {}
}
