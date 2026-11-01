package com.example.rag.chat;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.stream.Collectors;

import org.springframework.stereotype.Component;
import org.springframework.util.StringUtils;

import com.example.rag.chat.IntentClassifier.Intent;
import com.example.rag.ingest.DocumentLibraryService;
import com.example.rag.ingest.DocumentLibraryService.DocumentSummary;

/**
 * Chooses a retrieval strategy per question, at request time.
 *
 * <h2>Why this exists</h2>
 *
 * Top-k similarity search answers "which passages resemble this question?".
 * That is the wrong question for an overview request, and no prompt fixes it,
 * because the model never receives the chunks that were not retrieved.
 *
 * <p>Measured on a 2-document corpus, asking
 * <em>"provide the information for all the details of company handbook"</em>:
 *
 * <pre>
 *   0.2316  security-policy.txt   ← wrong document, and the TOP hit
 *   0.2056  company-handbook.txt
 *   0.1857  security-policy.txt   ← wrong document
 *   0.1825  company-handbook.txt
 * </pre>
 *
 * Two failures at once: nothing confined retrieval to the named document, and
 * four chunks cannot represent a whole one. The uniformly low scores are the
 * tell — a question <em>about</em> a document does not resemble any particular
 * passage in it.
 *
 * <h2>Both inputs are semantic, not keyword lists</h2>
 *
 * <ul>
 *   <li><b>Intent</b> comes from {@link IntentClassifier}, which compares the
 *       question's embedding against prototype questions. It generalises to
 *       phrasings nobody enumerated.</li>
 *   <li><b>Document</b> is matched against the filenames actually ingested,
 *       scored by word overlap — so it adapts as documents come and go, with
 *       no hardcoded names.</li>
 * </ul>
 */
@Component
public class RetrievalPlanner {

    public enum Strategy { TOP_K, FULL_DOCUMENT }

    /**
     * @param strategy what to do
     * @param source   document to confine to, or {@code null} for all
     * @param reason   human-readable justification, surfaced in the response
     *                 so the choice is never invisible
     */
    public record Plan(Strategy strategy, String source, String reason) {
    }

    /** Tokens too generic to identify a document on their own. */
    private static final Set<String> STOP_TOKENS = Set.of(
            "the", "a", "an", "of", "and", "doc", "docs", "document", "documents",
            "file", "files", "txt", "pdf", "docx", "md", "data", "info");

    /**
     * Fraction of a filename's distinctive words that must appear in the
     * question. 0.5 lets "the handbook" match "company-handbook.txt" while
     * still requiring more than one incidental word in longer names.
     */
    private static final double MIN_NAME_OVERLAP = 0.5;

    private final DocumentLibraryService library;
    private final IntentClassifier intentClassifier;

    public RetrievalPlanner(DocumentLibraryService library, IntentClassifier intentClassifier) {
        this.library = library;
        this.intentClassifier = intentClassifier;
    }

    /**
     * @param explicitScope a document chosen in the UI, which always wins over
     *                      anything inferred from the text
     */
    public Plan plan(String question, String explicitScope) {
        List<DocumentSummary> documents = library.list();
        String named = matchDocument(question, documents);

        // Classify the REQUEST, not its subject. A document name is a noun
        // phrase that says nothing about whether the user wants an overview
        // or one fact -- but it is part of the sentence embedding, so an
        // unusual or lengthy name drags the vector away from both prototype
        // sets. Removing it first makes "run me through the X" classify the
        // same regardless of what X is called.
        String intentText = stripDocumentName(question, named != null ? named : explicitScope);
        IntentClassifier.Result intent = intentClassifier.classify(intentText);
        boolean wantsOverview = intent.intent() == Intent.OVERVIEW;

        // An explicit scope is a user decision; never second-guess it.
        if (StringUtils.hasText(explicitScope)) {
            return wantsOverview
                    ? new Plan(Strategy.FULL_DOCUMENT, explicitScope,
                        "scope set to " + explicitScope + " and the question reads as an overview request")
                    : new Plan(Strategy.TOP_K, explicitScope,
                        "scope set to " + explicitScope);
        }

        if (wantsOverview && named != null) {
            return new Plan(Strategy.FULL_DOCUMENT, named,
                    "read all of " + named + " — the question reads as an overview request"
                    + margin(intent));
        }

        // An overview request with exactly one document ingested is
        // unambiguous even though no document was named.
        if (wantsOverview && documents.size() == 1) {
            String only = documents.get(0).source();
            return new Plan(Strategy.FULL_DOCUMENT, only,
                    "read all of " + only + " — overview request, and it is the only document"
                    + margin(intent));
        }

        if (wantsOverview) {
            return new Plan(Strategy.TOP_K, null,
                    "overview request, but it names no document — searched everything. "
                    + "Name a document to get a full summary of it." + margin(intent));
        }

        if (named != null) {
            return new Plan(Strategy.TOP_K, named,
                    "specific question about " + named + " — top matches within it" + margin(intent));
        }

        return new Plan(Strategy.TOP_K, null,
                "specific question — top matches across all documents" + margin(intent));
    }

    /**
     * Removes a matched document's words from the question, leaving the
     * request's shape behind. Returns the original if stripping would leave
     * too little to classify.
     */
    private static String stripDocumentName(String question, String source) {
        if (!StringUtils.hasText(source)) {
            return question;
        }
        Set<String> nameTokens = tokenise(source);
        if (nameTokens.isEmpty()) {
            return question;
        }

        String stripped = Arrays.stream(normalise(question).split("\\s+"))
                .filter(w -> !nameTokens.contains(w))
                .collect(Collectors.joining(" "))
                .trim();

        // "what is in X?" -> "what is in" still classifies; but if a question
        // were nothing but the document name, keep the original.
        return stripped.length() < 3 ? question : stripped;
    }

    private static String margin(IntentClassifier.Result r) {
        return String.format(" (overview %.2f vs specific %.2f)", r.overviewScore(), r.specificScore());
    }

    /**
     * Finds the ingested document a question refers to.
     *
     * <p>Scored by how many of a filename's distinctive words appear in the
     * question, so "company handbook txt file" matches
     * {@code company-handbook.txt} without any hardcoding.
     *
     * <p>Returns {@code null} when nothing clears the bar, or when two
     * documents tie — guessing between them would be a coin flip presented as
     * an answer.
     */
    private String matchDocument(String question, List<DocumentSummary> documents) {
        // NOT Set.of(...) -- it throws IllegalArgumentException on duplicates,
        // and any sentence with a repeated word would blow up the request.
        Set<String> questionWords = Arrays.stream(normalise(question).split("\\s+"))
                .collect(Collectors.toSet());

        record Scored(String source, double overlap) {
        }

        List<Scored> scored = documents.stream()
                .map(d -> {
                    Set<String> tokens = tokenise(d.source());
                    if (tokens.isEmpty()) {
                        return new Scored(d.source(), 0);
                    }
                    long hits = tokens.stream().filter(questionWords::contains).count();
                    return new Scored(d.source(), (double) hits / tokens.size());
                })
                .filter(x -> x.overlap() >= MIN_NAME_OVERLAP)
                .sorted(Comparator.comparingDouble(Scored::overlap).reversed())
                .toList();

        if (scored.isEmpty()) {
            return null;
        }
        if (scored.size() > 1 && scored.get(0).overlap() == scored.get(1).overlap()) {
            return null; // tie -- ambiguous, do not guess
        }
        return scored.get(0).source();
    }

    private static Set<String> tokenise(String filename) {
        return Arrays.stream(filename.toLowerCase(Locale.ROOT).split("[^a-z0-9]+"))
                .filter(t -> t.length() > 2)
                .filter(t -> !STOP_TOKENS.contains(t))
                .collect(Collectors.toSet());
    }

    private static String normalise(String question) {
        return question.toLowerCase(Locale.ROOT).replaceAll("[^a-z0-9]+", " ").trim();
    }
}
