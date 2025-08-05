package io.gravitee.inference.rest.openai.embedding.model;

import java.util.List;

public record EmbeddingResponse(
        String object,
        List<EmbeddingData> data,
        String model,
        Usage usage
) {
}