package io.gravitee.inference.rest.openai.embedding.model;

public record EmbeddingRequest(
        String model,
        String input,
        Integer dimensions,
        String encoding_format
) {
}
