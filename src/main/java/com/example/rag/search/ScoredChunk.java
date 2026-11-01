package com.example.rag.search;

/**
 * One retrieval hit, flattened for JSON.
 *
 * <p>{@code score} is the headline field. Everything downstream in Layer 3
 * depends on retrieval putting the right chunks at the top, so this endpoint
 * exists to make that number visible before an LLM is ever involved.
 */
public record ScoredChunk(
        int rank,
        Double score,
        String source,
        Object chunkIndex,
        String id,
        String excerpt) {
}
