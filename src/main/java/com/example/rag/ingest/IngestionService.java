package com.example.rag.ingest;

import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.document.Document;
import org.springframework.ai.reader.tika.TikaDocumentReader;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.ai.vectorstore.filter.Filter;
import org.springframework.ai.vectorstore.filter.FilterExpressionBuilder;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.stereotype.Service;

/**
 * Steps 1-3 of the RAG pipeline: extract text, chunk it, embed + store it.
 *
 * <p>The embedding step is invisible here on purpose. {@link VectorStore#add}
 * calls the {@code EmbeddingModel} bean internally, which is why switching
 * embedding providers is a dependency swap rather than a code change.
 */
@Service
public class IngestionService {

    private static final Logger log = LoggerFactory.getLogger(IngestionService.class);

    /** Upper bound when counting a source's existing chunks before replacing them. */
    private static final int MAX_REPLACEABLE_CHUNKS = 10_000;

    private final VectorStore vectorStore;
    private final TokenTextSplitter splitter;

    /**
     * Chunk size is the highest-leverage knob in RAG, so it is configurable
     * rather than hardcoded -- expect to tune it and re-ingest several times.
     * Too small and chunks lose the context that makes them meaningful;
     * too large and retrieval returns mostly irrelevant filler.
     */
    public IngestionService(
            VectorStore vectorStore,
            @Value("${rag.chunking.chunk-size:800}") int chunkSize,
            @Value("${rag.chunking.min-chunk-size-chars:350}") int minChunkSizeChars) {

        this.vectorStore = vectorStore;
        this.splitter = TokenTextSplitter.builder()
                .withChunkSize(chunkSize)
                .withMinChunkSizeChars(minChunkSizeChars)
                .build();

        log.info("Chunking configured: chunkSize={} tokens, minChunkSizeChars={}",
                chunkSize, minChunkSizeChars);
    }

    /**
     * @param resource   the uploaded file
     * @param sourceName original filename, recorded as metadata so answers can cite it
     * @return number of chunks written to the vector store
     */
    public int ingest(Resource resource, String sourceName) {
        // IDEMPOTENCY - drop any chunks previously ingested from this same
        // source. Without this, re-uploading a file leaves BOTH copies in the
        // store, and retrieval then over-weights the duplicated content
        // (the same chunk occupies several of your topK slots).
        int removed = deleteBySource(sourceName);
        if (removed > 0) {
            log.info("Replacing '{}': removed {} existing chunk(s)", sourceName, removed);
        }

        // EXTRACT - Tika handles PDF, DOCX, PPTX, HTML and ~20 other formats.
        List<Document> documents = new TikaDocumentReader(resource).read();

        // Tag before splitting so every derived chunk inherits the source.
        documents.forEach(doc -> doc.getMetadata().put("source", sourceName));

        // TRANSFORM
        List<Document> chunks = splitter.apply(documents);

        // LOAD - embeddings are generated here, inside add().
        vectorStore.add(chunks);

        log.info("Ingested '{}': {} document(s) -> {} chunk(s)",
                sourceName, documents.size(), chunks.size());
        return chunks.size();
    }

    /**
     * Deletes every chunk whose {@code source} metadata matches.
     *
     * <p>Uses {@link FilterExpressionBuilder} rather than string-concatenating
     * a filter expression -- a filename containing a quote would otherwise
     * break the expression or inject into it.
     *
     * @return how many chunks were removed
     */
    public int deleteBySource(String sourceName) {
        Filter.Expression bySource = new FilterExpressionBuilder()
                .eq("source", sourceName)
                .build();

        List<Document> existing = vectorStore.similaritySearch(
                SearchRequest.builder()
                        .query(sourceName)
                        .topK(MAX_REPLACEABLE_CHUNKS)
                        .similarityThresholdAll()
                        .filterExpression(bySource)
                        .build());

        if (existing == null || existing.isEmpty()) {
            return 0;
        }
        vectorStore.delete(bySource);
        return existing.size();
    }
}
