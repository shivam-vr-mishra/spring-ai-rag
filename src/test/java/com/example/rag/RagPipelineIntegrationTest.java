package com.example.rag;

import static org.assertj.core.api.Assertions.assertThat;

import java.nio.charset.StandardCharsets;
import java.util.List;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.core.io.Resource;

import com.example.rag.ingest.DocumentLibraryService;
import com.example.rag.ingest.IngestionService;
import com.example.rag.search.ScoredChunk;
import com.example.rag.search.SearchService;

/**
 * End-to-end coverage of Layers 1 and 2 against the real PGVector instance.
 *
 * <p>Deliberately needs no {@code ANTHROPIC_API_KEY}: ingestion and retrieval
 * are fully local, and proving that is part of the point.
 *
 * <p>Uses a unique source name so it never collides with, or deletes, whatever
 * the developer has ingested through the UI.
 */
@SpringBootTest(properties = {
        "rag.chunking.chunk-size=120",
        "rag.chunking.min-chunk-size-chars=100"
})
class RagPipelineIntegrationTest {

    private static final String SOURCE = "__test__fixture.txt";

    private static final String FIXTURE = """
            ACME Robotics Test Fixture

            SECTION 1: ON-CALL COMPENSATION
            Engineers on the primary on-call rotation receive a stipend of 800
            dollars per week. Secondary on-call receives 300 dollars per week.
            Any engineer paged more than four times in a single night is
            entitled to the following day off.

            SECTION 2: DEPLOYMENT POLICY
            All production deployments must occur between 10:00 and 16:00 on
            Tuesday, Wednesday, or Thursday. Friday deployments are strictly
            forbidden because the on-call rotation changes at 18:00 on Fridays.

            SECTION 3: BATTERY SPECIFICATIONS
            The Hauler-7 warehouse transport unit has a 400 kilogram payload
            capacity and a nine hour battery life. The Scout-1 inspection drone
            has a 42 minute flight time and thermal imaging.
            """;

    @Autowired
    private IngestionService ingestionService;

    @Autowired
    private SearchService searchService;

    @Autowired
    private DocumentLibraryService libraryService;

    private static Resource fixture() {
        byte[] bytes = FIXTURE.getBytes(StandardCharsets.UTF_8);
        return new ByteArrayResource(bytes) {
            @Override
            public String getFilename() {
                return SOURCE;
            }
        };
    }

    @BeforeEach
    void ingestFixture() {
        ingestionService.ingest(fixture(), SOURCE);
    }

    @AfterEach
    void cleanUp() {
        ingestionService.deleteBySource(SOURCE);
    }

    @Test
    @DisplayName("Layer 1: a document is split into multiple chunks and stored")
    void ingestionProducesChunks() {
        List<DocumentLibraryService.DocumentSummary> docs = libraryService.list();

        assertThat(docs)
                .filteredOn(d -> d.source().equals(SOURCE))
                .singleElement()
                .satisfies(d -> assertThat(d.chunks())
                        .as("fixture is ~250 tokens at chunk-size 120, so it must split")
                        .isGreaterThan(1));
    }

    @Test
    @DisplayName("Layer 1: re-ingesting replaces rather than duplicates")
    void reIngestionIsIdempotent() {
        int before = chunkCountForSource();

        ingestionService.ingest(fixture(), SOURCE);
        ingestionService.ingest(fixture(), SOURCE);

        assertThat(chunkCountForSource())
                .as("uploading the same source three times must not triple the chunks")
                .isEqualTo(before);
    }

    @Test
    @DisplayName("Layer 2: semantic search matches meaning, not keywords")
    void searchMatchesMeaningNotKeywords() {
        // "paid" never appears in the fixture -- it says "receives a stipend".
        List<ScoredChunk> hits = searchService.search(
                "how much do I get paid for being on call", 4, 0.0);

        assertThat(hits).isNotEmpty();
        assertThat(hits.get(0).excerpt())
                .as("top hit should be the compensation section despite zero keyword overlap")
                .containsIgnoringCase("stipend");
        assertThat(hits.get(0).score())
                .as("a direct semantic hit should score well clear of noise")
                .isGreaterThan(0.3);
    }

    @Test
    @DisplayName("Layer 2: an off-topic query still returns hits at threshold 0")
    void offTopicQueryStillReturnsHitsWithoutAThreshold() {
        List<ScoredChunk> hits = searchService.search(
                "what is the best topping for a pizza", 4, 0.0);

        // This is the defining property of vector search: it returns NEAREST
        // neighbours, never "relevant" ones. Layer 3 depends on a threshold
        // precisely because of this.
        assertThat(hits)
                .as("nearest-neighbour search always returns something")
                .isNotEmpty();
        assertThat(hits.get(0).score())
                .as("...but the score should be near zero")
                .isLessThan(0.1);
    }

    @Test
    @DisplayName("Layer 2: the threshold filters out the off-topic query")
    void thresholdFiltersOutNoise() {
        List<ScoredChunk> relevant = searchService.search(
                "how much do I get paid for being on call", 4, 0.1);
        List<ScoredChunk> noise = searchService.search(
                "what is the best topping for a pizza", 4, 0.1);

        assertThat(relevant).as("a real question still returns hits at 0.1").isNotEmpty();
        assertThat(noise).as("an unrelated question returns nothing at 0.1").isEmpty();
    }

    @Test
    @DisplayName("Layer 2: topK caps the number of results")
    void topKCapsResults() {
        assertThat(searchService.search("on call deployment battery", 1, 0.0)).hasSize(1);
        assertThat(searchService.search("on call deployment battery", 2, 0.0)).hasSize(2);
    }

    @Test
    @DisplayName("Layer 2: a source filter confines retrieval to one document")
    void sourceFilterConfinesRetrieval() {
        // Unfiltered, a vague question can pull chunks from any document --
        // naming a document in the question text does nothing, because that
        // text just becomes part of the embedding.
        List<ScoredChunk> scoped = searchService.search(
                "provide all the details of the handbook", 4, 0.0, SOURCE);

        assertThat(scoped).isNotEmpty();
        assertThat(scoped)
                .as("every hit must come from the filtered source")
                .allSatisfy(c -> assertThat(c.source()).isEqualTo(SOURCE));
    }

    @Test
    @DisplayName("a filter for an unknown source returns nothing, not everything")
    void unknownSourceFilterReturnsNothing() {
        assertThat(searchService.search("on call pay", 4, 0.0, "no-such-file.txt"))
                .isEmpty();
    }

    @Test
    @DisplayName("summarisation reads EVERY chunk, unlike top-k retrieval")
    void allChunkTextsReturnsWholeDocumentInOrder() {
        int stored = chunkCountForSource();
        List<String> all = libraryService.allChunkTexts(SOURCE);

        assertThat(all)
                .as("summarisation needs the whole document, not the top k")
                .hasSize(stored);

        // Ordered by chunk_index, not id -- ids are random UUIDs, so ordering
        // by them would return the document scrambled.
        assertThat(all.get(0))
                .as("first chunk should be the start of the fixture")
                .contains("ACME Robotics Test Fixture");
    }

    @Test
    @DisplayName("Layer 1: delete removes every chunk for a source")
    void deleteRemovesAllChunksForSource() {
        assertThat(chunkCountForSource()).isGreaterThan(0);

        int removed = ingestionService.deleteBySource(SOURCE);

        assertThat(removed).isGreaterThan(0);
        assertThat(chunkCountForSource()).isZero();
    }

    private int chunkCountForSource() {
        return libraryService.list().stream()
                .filter(d -> d.source().equals(SOURCE))
                .mapToInt(DocumentLibraryService.DocumentSummary::chunks)
                .sum();
    }
}
