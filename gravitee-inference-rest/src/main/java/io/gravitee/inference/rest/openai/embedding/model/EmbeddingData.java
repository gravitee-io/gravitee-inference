package io.gravitee.inference.rest.openai.embedding.model;

public record EmbeddingData(
        String object,
        int index,
        float[] embedding
) {
}