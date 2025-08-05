package io.gravitee.inference.rest.openai.embedding.model;

public record Usage(
        int prompt_tokens,
        int total_tokens
) {
}