package com.example.rag.ingest;

import java.util.List;

import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;

/**
 * Read-only view of what is currently ingested.
 *
 * <p>Queries the {@code vector_store} table directly with JDBC. The
 * {@code VectorStore} interface deliberately exposes only add / delete /
 * similaritySearch -- there is no "list everything" or "fetch all chunks of
 * document X" operation, because neither is a vector-search concern. Since
 * PGVector is a plain Postgres table, SQL is the straightforward way to get
 * them.
 */
@Service
public class DocumentLibraryService {

    /** One ingested source file and how many chunks it produced. */
    public record DocumentSummary(String source, int chunks) {
    }

    private final JdbcTemplate jdbcTemplate;

    public DocumentLibraryService(JdbcTemplate jdbcTemplate) {
        this.jdbcTemplate = jdbcTemplate;
    }

    public List<DocumentSummary> list() {
        return jdbcTemplate.query("""
                SELECT metadata->>'source' AS source, COUNT(*) AS chunk_count
                FROM vector_store
                WHERE metadata->>'source' IS NOT NULL
                GROUP BY metadata->>'source'
                ORDER BY metadata->>'source'
                """,
                (rs, rowNum) -> new DocumentSummary(
                        rs.getString("source"),
                        rs.getInt("chunk_count")));
    }

    public int totalChunks() {
        Integer count = jdbcTemplate.queryForObject(
                "SELECT COUNT(*) FROM vector_store", Integer.class);
        return count == null ? 0 : count;
    }

    public boolean exists(String source) {
        Integer count = jdbcTemplate.queryForObject(
                "SELECT COUNT(*) FROM vector_store WHERE metadata->>'source' = ?",
                Integer.class, source);
        return count != null && count > 0;
    }

    /**
     * Every chunk of one document, in original reading order.
     *
     * <p>This is the deliberate opposite of similarity search. Summarising a
     * document needs <em>all</em> of it; top-k returns only the passages that
     * happen to resemble the query, which is why asking a RAG chatbot to
     * "summarise the handbook" normally produces a partial answer stitched
     * from whichever chunks scored highest.
     *
     * <p>Ordered by {@code chunk_index}, not {@code id} -- ids are random
     * UUIDs, so ordering by them returns the document scrambled.
     */
    public List<String> allChunkTexts(String source) {
        return jdbcTemplate.query("""
                SELECT content
                FROM vector_store
                WHERE metadata->>'source' = ?
                ORDER BY (metadata->>'chunk_index')::int
                """,
                (rs, rowNum) -> rs.getString("content"),
                source);
    }
}
