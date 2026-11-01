package com.example.rag.chat;

import java.util.List;

import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * Intent-classification examples, bound from {@code rag.intent.*}.
 *
 * <p>These are domain knowledge rather than logic, so they live in
 * configuration: a legal team and a support team phrase overview requests
 * differently, and neither should have to recompile. When a real question
 * routes the wrong way, adding one example is cheaper and more reliable than
 * writing a rule.
 *
 * <p>Bound with {@code @ConfigurationProperties}, not {@code @Value} --
 * {@code @Value} cannot bind a YAML sequence to a {@code List<String>}.
 *
 * @param overviewExamples      questions wanting a whole document
 * @param specificExamples      questions wanting one fact
 * @param minOverviewSimilarity floor below which a question is treated as
 *                              resembling neither list
 */
@ConfigurationProperties(prefix = "rag.intent")
public record IntentProperties(
        List<String> overviewExamples,
        List<String> specificExamples,
        Double minOverviewSimilarity) {

    public IntentProperties {
        if (overviewExamples == null || overviewExamples.isEmpty()) {
            throw new IllegalArgumentException(
                    "rag.intent.overview-examples must not be empty — "
                    + "intent classification has nothing to compare against");
        }
        if (specificExamples == null || specificExamples.isEmpty()) {
            throw new IllegalArgumentException(
                    "rag.intent.specific-examples must not be empty — "
                    + "intent classification has nothing to compare against");
        }
        if (minOverviewSimilarity == null) {
            minOverviewSimilarity = 0.35;
        }
    }
}
