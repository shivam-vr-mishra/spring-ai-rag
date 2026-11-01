package com.example.rag;

import static org.assertj.core.api.Assertions.assertThat;

import java.nio.charset.StandardCharsets;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.core.io.Resource;

import com.example.rag.chat.RetrievalPlanner;
import com.example.rag.chat.RetrievalPlanner.Strategy;
import com.example.rag.ingest.IngestionService;

/**
 * Retrieval-strategy routing. No LLM calls — these are pure decision logic and
 * run in milliseconds.
 *
 * <p>Two documents are ingested so the "names a document" matching has
 * something to disambiguate between.
 */
@SpringBootTest(properties = {
        "rag.chunking.chunk-size=120",
        "rag.chunking.min-chunk-size-chars=100"
})
class RetrievalPlannerTest {

    private static final String HANDBOOK = "__test__zarquon-manual.txt";
    private static final String SECURITY = "__test__krillian-protocol.txt";

    @Autowired
    private RetrievalPlanner planner;

    @Autowired
    private IngestionService ingestionService;

    private static Resource doc(String name, String body) {
        return new ByteArrayResource(body.getBytes(StandardCharsets.UTF_8)) {
            @Override
            public String getFilename() {
                return name;
            }
        };
    }

    @BeforeEach
    void ingest() {
        ingestionService.ingest(doc(HANDBOOK, """
                Zarquon Manual. SECTION 1: DEPLOYMENT POLICY. Deployments happen
                Tuesday to Thursday. SECTION 2: ON-CALL COMPENSATION. Primary
                on-call receives a stipend of 800 dollars per week.
                """), HANDBOOK);
        ingestionService.ingest(doc(SECURITY, """
                Krillian Protocol. SECTION 1: PASSWORDS. Use a password
                manager. SECTION 2: DEVICES. Full-disk encryption is mandatory.
                """), SECURITY);
    }

    @AfterEach
    void cleanUp() {
        ingestionService.deleteBySource(HANDBOOK);
        ingestionService.deleteBySource(SECURITY);
    }

    @Test
    @DisplayName("overview question naming a document reads the whole document")
    void overviewPlusNamedDocumentGoesFullDocument() {
        // The exact phrasing that originally returned a confused partial answer.
        var plan = planner.plan(
                "provide the information for all the details of " + nameWords(HANDBOOK), null);

        assertThat(plan.strategy()).isEqualTo(Strategy.FULL_DOCUMENT);
        assertThat(plan.source()).isEqualTo(HANDBOOK);
    }

    @Test
    @DisplayName("a repeated word does not break planning")
    void repeatedWordsAreHandled() {
        // Set.of(...) throws IllegalArgumentException on duplicates; this
        // sentence has "the" three times and used to 500 the whole request.
        var plan = planner.plan(
                "give me all the details of the " + nameWords(HANDBOOK) + " the full text", null);

        assertThat(plan.strategy()).isEqualTo(Strategy.FULL_DOCUMENT);
    }

    @Test
    @DisplayName("intent is understood semantically, not by keyword")
    void understandsPhrasingsWithNoOverviewKeywords() {
        // Contains none of "summarise", "overview", "all the details",
        // "everything" -- the words a regex approach would have keyed on.
        for (String q : new String[] {
                "what is in " + nameWords(HANDBOOK) + " txt file?",
                "whats inside the " + nameWords(HANDBOOK) + "?",
                "run me through the " + nameWords(HANDBOOK) }) {
            var plan = planner.plan(q, null);
            assertThat(plan.strategy()).as("question: %s", q).isEqualTo(Strategy.FULL_DOCUMENT);
            assertThat(plan.source()).isEqualTo(HANDBOOK);
        }
    }

    @Test
    @DisplayName("'summarize' triggers the full-document path")
    void summariseVerbTriggersFullDocument() {
        for (String verb : new String[] { "summarize", "summarise", "give me an overview of" }) {
            var plan = planner.plan(verb + " " + nameWords(SECURITY), null);
            assertThat(plan.strategy())
                    .as("verb: %s", verb)
                    .isEqualTo(Strategy.FULL_DOCUMENT);
            assertThat(plan.source()).isEqualTo(SECURITY);
        }
    }

    @Test
    @DisplayName("a specific question still uses top-k")
    void specificQuestionStaysTopK() {
        var plan = planner.plan("how much is on-call pay?", null);

        assertThat(plan.strategy()).isEqualTo(Strategy.TOP_K);
        assertThat(plan.source()).as("no document named, so search everything").isNull();
    }

    @Test
    @DisplayName("naming a document in a specific question scopes retrieval to it")
    void namedDocumentScopesTopK() {
        var plan = planner.plan("what does the " + nameWords(SECURITY) + " say about passwords?", null);

        assertThat(plan.strategy()).isEqualTo(Strategy.TOP_K);
        assertThat(plan.source()).isEqualTo(SECURITY);
    }

    @Test
    @DisplayName("an off-topic question is not mistaken for an overview")
    void offTopicStaysTopK() {
        var plan = planner.plan("what is the best topping for a pizza?", null);

        assertThat(plan.strategy()).isEqualTo(Strategy.TOP_K);
        assertThat(plan.source()).isNull();
    }

    @Test
    @DisplayName("an explicit scope always wins over inference")
    void explicitScopeOverridesInference() {
        // Question names the handbook, but the caller pinned the security doc.
        var plan = planner.plan("all the details of " + nameWords(HANDBOOK), SECURITY);

        assertThat(plan.source())
                .as("a user's explicit choice is not second-guessed")
                .isEqualTo(SECURITY);
        assertThat(plan.strategy()).isEqualTo(Strategy.FULL_DOCUMENT);
    }

    @Test
    @DisplayName("an overview request naming no document does not guess")
    void ambiguousOverviewDoesNotGuess() {
        // Two documents ingested and neither named: picking one would be a
        // coin flip presented as an answer.
        var plan = planner.plan("summarize everything", null);

        assertThat(plan.strategy()).isEqualTo(Strategy.TOP_K);
        assertThat(plan.source()).isNull();
        assertThat(plan.reason()).contains("names no document");
    }

    @Test
    @DisplayName("every plan explains itself")
    void everyPlanHasAReason() {
        for (String q : new String[] {
                "how much is on-call pay?",
                "summarize " + nameWords(HANDBOOK),
                "what is the best topping for a pizza?" }) {
            assertThat(planner.plan(q, null).reason())
                    .as("question: %s", q)
                    .isNotBlank();
        }
    }

    /** "__test__company-handbook.txt" -> "test company handbook" */
    private static String nameWords(String source) {
        return source.replaceAll("\\.txt$", "").replaceAll("[^a-zA-Z0-9]+", " ").trim();
    }
}
