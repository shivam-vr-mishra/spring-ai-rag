package com.example.rag;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import com.example.rag.chat.IntentClassifier;
import com.example.rag.chat.IntentClassifier.Intent;

/**
 * Semantic intent classification.
 *
 * <p>Every phrase below is deliberately <strong>absent</strong> from the
 * prototype lists in {@code IntentClassifier}. That is the point: a keyword or
 * regex approach only recognises the wordings its author happened to think of,
 * and the previous regex version failed on "what is in company handbook txt
 * file?" — a phrasing that means exactly the same as "summarise the handbook".
 *
 * <p>These run against the local embedding model: no LLM, no network.
 */
@SpringBootTest
class IntentClassifierTest {

    @Autowired
    private IntentClassifier classifier;

    @ParameterizedTest(name = "OVERVIEW: {0}")
    @ValueSource(strings = {
            "what is in company handbook txt file?",   // the one the regex missed
            "give me the gist of the handbook",
            "run me through what the policy says",
            "I want a high level picture of this document",
            "break down the contents of the file for me",
            "outline everything covered here",
            "what ground does this document cover?",
            "can you recap the whole thing",
            "describe the document from top to bottom",
            "whats the full picture in that file"
    })
    @DisplayName("unenumerated overview phrasings are recognised")
    void recognisesOverviewPhrasings(String question) {
        IntentClassifier.Result r = classifier.classify(question);

        assertThat(r.intent())
                .as("'%s' -> overview=%s specific=%s", question, r.overviewScore(), r.specificScore())
                .isEqualTo(Intent.OVERVIEW);
    }

    @ParameterizedTest(name = "SPECIFIC: {0}")
    @ValueSource(strings = {
            "what day can I ship to prod?",
            "am I allowed to use a thumb drive at work?",
            "how long can the drone stay airborne?",
            "who signs off before a release goes out?",
            "what happens if everything goes down for every customer?",
            "do I need a hardware key for the prod database?",
            "how many dollars per week for backup rotation?",
            "is there a limit on how often I can be paged?"
    })
    @DisplayName("unenumerated specific phrasings are recognised")
    void recognisesSpecificPhrasings(String question) {
        IntentClassifier.Result r = classifier.classify(question);

        assertThat(r.intent())
                .as("'%s' -> overview=%s specific=%s", question, r.overviewScore(), r.specificScore())
                .isEqualTo(Intent.SPECIFIC);
    }

    @ParameterizedTest(name = "off-topic stays SPECIFIC: {0}")
    @ValueSource(strings = {
            "what is the best topping for a pizza?",
            "who won the world cup in 1998?"
    })
    @DisplayName("off-topic questions are not mistaken for overview requests")
    void offTopicIsNotOverview(String question) {
        // An off-topic question must not trigger a full-document read; the
        // threshold in the TOP_K path is what makes it answer "I don't know".
        assertThat(classifier.classify(question).intent()).isEqualTo(Intent.SPECIFIC);
    }

    @org.junit.jupiter.api.Test
    @DisplayName("classification reports a usable confidence margin")
    void reportsConfidence() {
        IntentClassifier.Result overview = classifier.classify("summarise the whole document");
        IntentClassifier.Result specific = classifier.classify("how much is the stipend?");

        assertThat(overview.confidence()).isGreaterThan(0.0);
        assertThat(specific.confidence()).isGreaterThan(0.0);
        assertThat(overview.overviewScore()).isGreaterThan(overview.specificScore());
        assertThat(specific.specificScore()).isGreaterThan(specific.overviewScore());
    }
}
