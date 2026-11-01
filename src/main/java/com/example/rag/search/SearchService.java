package com.example.rag.search;

import java.util.ArrayList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.document.Document;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.filter.FilterExpressionBuilder;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.stereotype.Service;

/**
 * Step 4 of the RAG pipeline: semantic search, with no LLM involved.
 *
 * <p>This is deliberately a separate, inspectable layer. When a RAG answer is
 * wrong, the cause is almost always that retrieval returned the wrong chunks --
 * not that the model reasoned badly over the right ones. You can only tell
 * those apart if you can look at retrieval in isolation.
 */
@Service
public class SearchService {

    private static final Logger log = LoggerFactory.getLogger(SearchService.class);

    private static final int EXCERPT_CHARS = 300;

    private final VectorStore vectorStore;

    public SearchService(VectorStore vectorStore) {
        this.vectorStore = vectorStore;
    }

    /**
     * @param query     natural-language question; embedded with the SAME model
     *                  used at ingestion time, which is what makes the vectors
     *                  comparable at all
     * @param topK      maximum hits to return
     * @param threshold minimum similarity, 0.0-1.0. 0.0 accepts everything.
     */
    public List<ScoredChunk> search(String query, int topK, double threshold) {
        return search(query, topK, threshold, null);
    }

    /**
     * @param source when set, restricts retrieval to one document.
     *               Semantic similarity alone has no notion of "search only
     *               inside this file" -- naming the document in the question
     *               text does nothing, it just becomes part of the embedding.
     */
    public List<ScoredChunk> search(String query, int topK, double threshold, String source) {
        SearchRequest.Builder builder = SearchRequest.builder()
                .query(query)
                .topK(topK)
                .similarityThreshold(threshold);

        if (source != null && !source.isBlank()) {
            builder.filterExpression(
                    new FilterExpressionBuilder().eq("source", source).build());
        }
        SearchRequest request = builder.build();

        List<Document> hits = vectorStore.similaritySearch(request);
        if (hits == null) {
            hits = List.of();
        }

        List<ScoredChunk> results = new ArrayList<>(hits.size());
        int rank = 1;
        for (Document doc : hits) {
            String text = doc.getText() == null ? "" : doc.getText();
            results.add(new ScoredChunk(
                    rank++,
                    doc.getScore(),
                    String.valueOf(doc.getMetadata().get("source")),
                    doc.getMetadata().get("chunk_index"),
                    doc.getId(),
                    excerpt(text)));
        }

        log.debug("query='{}' topK={} threshold={} -> {} hit(s)",
                query, topK, threshold, results.size());
        return results;
    }

    private static String excerpt(String text) {
        String collapsed = text.replaceAll("\\s+", " ").trim();
        return collapsed.length() <= EXCERPT_CHARS
                ? collapsed
                : collapsed.substring(0, EXCERPT_CHARS) + "...";
    }
}
