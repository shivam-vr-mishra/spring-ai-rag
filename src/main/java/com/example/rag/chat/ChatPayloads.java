package com.example.rag.chat;

import java.util.List;

/** Request/response payloads for {@code POST /api/chat}. */
public final class ChatPayloads {

    private ChatPayloads() {
    }

    /**
     * @param question  the user's question
     * @param topK      optional override of how many chunks to retrieve
     * @param threshold optional override of the minimum similarity score
     */
    public record ChatRequest(String question, String source, Integer topK, Double threshold) {
    }

    /**
     * One chunk that was actually fed to the model as context.
     *
     * <p>Returning these is not decoration. An answer with no citations is an
     * ungrounded answer, and the only way to tell a good answer from a
     * confident hallucination is to see what it was built from.
     */
    public record Citation(int rank, Double score, String source, Object chunkIndex, String excerpt) {
    }

    public record TokenUsage(Integer promptTokens, Integer completionTokens, Integer totalTokens) {
    }

    /**
     * @param strategy    TOP_K or FULL_DOCUMENT — which retrieval path ran
     * @param strategyReason why that path was chosen, so the routing decision
     *                       is visible rather than mysterious
     * @param chunksUsed  how many chunks reached the model
     * @param chunksTotal how many exist for the scoped document (FULL_DOCUMENT only)
     * @param truncated   true when the document exceeded the prompt budget
     */
    public record ChatAnswer(
            String question,
            String source,
            String strategy,
            String strategyReason,
            Integer chunksUsed,
            Integer chunksTotal,
            boolean truncated,
            String answer,
            boolean grounded,
            int topK,
            double threshold,
            List<Citation> citations,
            TokenUsage usage) {
    }
}
