package com.example.rag.chat;

import java.util.ArrayList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.ChatClientResponse;
import org.springframework.ai.chat.client.advisor.vectorstore.QuestionAnswerAdvisor;
import org.springframework.ai.chat.metadata.Usage;
import org.springframework.ai.document.Document;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.filter.FilterExpressionBuilder;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

import com.example.rag.chat.ChatPayloads.ChatAnswer;
import com.example.rag.chat.RetrievalPlanner.Plan;
import com.example.rag.chat.RetrievalPlanner.Strategy;
import com.example.rag.ingest.DocumentLibraryService;
import com.example.rag.chat.ChatPayloads.Citation;
import com.example.rag.chat.ChatPayloads.TokenUsage;

/**
 * Steps 5-6 of the RAG pipeline: retrieve, then generate.
 *
 * <p>{@link QuestionAnswerAdvisor} performs the Layer 2 similarity search and
 * splices the resulting chunks into the prompt before it reaches the model.
 * The retrieval is identical to {@code /api/search} -- same vector store, same
 * embedding model, same threshold -- which is why calibrating that threshold
 * in Layer 2 directly controls what the model is allowed to see here.
 *
 * <p>Each request is independent: there is no conversation memory, so a
 * follow-up like "what about secondary?" has no referent.
 */
@Service
public class ChatService {

    private static final Logger log = LoggerFactory.getLogger(ChatService.class);

    private static final int EXCERPT_CHARS = 200;

    /**
     * Grounding instructions. The advisor supplies the context block; this
     * governs what the model is permitted to do with it.
     *
     * <p>The "say you don't know" clause matters: retrieval can return weak
     * chunks that clear the threshold without actually containing the answer,
     * and without this the model will stitch together something plausible.
     */
    private static final String SYSTEM_PROMPT = """
            You answer questions strictly from the provided context documents.

            Rules:
            - Use only information present in the context. Do not supplement it
              with prior knowledge.
            - If the context does not contain the answer, say so plainly and
              stop. Do not speculate or infer beyond what is written.
            - When you state a fact, keep the wording close to the source so it
              stays verifiable.
            - Be concise. Answer the question that was asked.
            """;

    private final ChatClient chatClient;
    private final VectorStore vectorStore;
    private final RetrievalPlanner planner;
    private final DocumentLibraryService library;
    private final int defaultTopK;
    private final double defaultThreshold;
    private final int maxFullDocumentChunks;

    public ChatService(
            ChatClient.Builder chatClientBuilder,
            VectorStore vectorStore,
            RetrievalPlanner planner,
            DocumentLibraryService library,
            @Value("${rag.retrieval.top-k:4}") int defaultTopK,
            @Value("${rag.retrieval.similarity-threshold:0.1}") double defaultThreshold,
            @Value("${rag.retrieval.max-full-document-chunks:40}") int maxFullDocumentChunks) {

        this.chatClient = chatClientBuilder.defaultSystem(SYSTEM_PROMPT).build();
        this.vectorStore = vectorStore;
        this.planner = planner;
        this.library = library;
        this.defaultTopK = defaultTopK;
        this.defaultThreshold = defaultThreshold;
        this.maxFullDocumentChunks = maxFullDocumentChunks;

        log.info("Chat retrieval defaults: topK={}, similarityThreshold={}, maxFullDocumentChunks={}",
                defaultTopK, defaultThreshold, maxFullDocumentChunks);
    }

    public ChatAnswer ask(String question, Integer topKOverride, Double thresholdOverride) {
        return ask(question, null, topKOverride, thresholdOverride);
    }

    /**
     * @param explicitScope optional document chosen by the caller. When absent,
     *                      {@link RetrievalPlanner} infers both the scope and
     *                      the retrieval strategy from the question itself.
     */
    public ChatAnswer ask(String question, String explicitScope,
                          Integer topKOverride, Double thresholdOverride) {

        int topK = topKOverride != null ? topKOverride : defaultTopK;
        double threshold = thresholdOverride != null ? thresholdOverride : defaultThreshold;

        Plan plan = planner.plan(question, explicitScope);
        log.debug("question='{}' -> {} ({})", question, plan.strategy(), plan.reason());

        return plan.strategy() == Strategy.FULL_DOCUMENT
                ? answerFromWholeDocument(question, plan, topK, threshold)
                : answerFromTopK(question, plan, topK, threshold);
    }

    /**
     * Ordinary RAG: embed the question, take the nearest chunks.
     */
    private ChatAnswer answerFromTopK(String question, Plan plan, int topK, double threshold) {
        SearchRequest.Builder search = SearchRequest.builder()
                .topK(topK)
                .similarityThreshold(threshold);
        if (StringUtils.hasText(plan.source())) {
            search.filterExpression(
                    new FilterExpressionBuilder().eq("source", plan.source()).build());
        }

        QuestionAnswerAdvisor advisor = QuestionAnswerAdvisor.builder(vectorStore)
                .searchRequest(search.build())
                .build();

        ChatClientResponse response = chatClient.prompt()
                .advisors(advisor)
                .user(question)
                .call()
                .chatClientResponse();

        List<Citation> citations = extractCitations(response);

        return new ChatAnswer(
                question, plan.source(), Strategy.TOP_K.name(), plan.reason(),
                citations.size(), null, false,
                extractAnswer(response), !citations.isEmpty(),
                topK, threshold, citations, extractUsage(response));
    }

    /**
     * Overview path: skip similarity search entirely and read the document.
     *
     * <p>No embedding, no scores, no threshold. Chunks arrive in reading order
     * (by {@code chunk_index}, not by {@code id} -- ids are random UUIDs, so
     * ordering by them returns the document scrambled).
     */
    private ChatAnswer answerFromWholeDocument(String question, Plan plan,
                                               int topK, double threshold) {
        List<String> chunks = library.allChunkTexts(plan.source());
        if (chunks.isEmpty()) {
            // The planner matched a name that has since been deleted.
            return answerFromTopK(question, new Plan(Strategy.TOP_K, null,
                    "no chunks found for " + plan.source() + " -- fell back to searching everything"),
                    topK, threshold);
        }

        int total = chunks.size();
        boolean truncated = total > maxFullDocumentChunks;
        List<String> used = truncated ? chunks.subList(0, maxFullDocumentChunks) : chunks;

        String context = String.join("\n\n", used);

        String answer = chatClient.prompt()
                .user(u -> u.text("""
                        Answer using the complete document below.

                        Document: {source}
                        ---------------------
                        {document}
                        ---------------------

                        Question: {question}

                        Cover every section. Do not ask which part to focus on.""")
                        .param("source", plan.source())
                        .param("document", context)
                        .param("question", question))
                .call()
                .content();

        // Citations are the chunks actually sent. No scores: nothing was ranked.
        List<Citation> citations = new ArrayList<>(used.size());
        for (int i = 0; i < used.size(); i++) {
            citations.add(new Citation(i + 1, null, plan.source(), i, excerpt(used.get(i))));
        }

        log.info("Full-document answer for '{}': {}/{} chunk(s){}",
                plan.source(), used.size(), total, truncated ? " (TRUNCATED)" : "");

        return new ChatAnswer(
                question, plan.source(), Strategy.FULL_DOCUMENT.name(), plan.reason(),
                used.size(), total, truncated,
                answer, true, topK, threshold, citations, null);
    }

    private static String extractAnswer(ChatClientResponse response) {
        if (response.chatResponse() == null || response.chatResponse().getResult() == null) {
            return "";
        }
        return response.chatResponse().getResult().getOutput().getText();
    }

    /**
     * Pulls the chunks the advisor retrieved out of the advisor context, so the
     * caller can see exactly what grounded the answer.
     */
    @SuppressWarnings("unchecked")
    private static List<Citation> extractCitations(ChatClientResponse response) {
        Object raw = response.context().get(QuestionAnswerAdvisor.RETRIEVED_DOCUMENTS);
        if (!(raw instanceof List<?> documents) || documents.isEmpty()) {
            return List.of();
        }

        List<Citation> citations = new ArrayList<>(documents.size());
        int rank = 1;
        for (Object element : (List<Document>) documents) {
            if (!(element instanceof Document doc)) {
                continue;
            }
            String text = doc.getText() == null ? "" : doc.getText();
            citations.add(new Citation(
                    rank++,
                    doc.getScore(),
                    String.valueOf(doc.getMetadata().get("source")),
                    doc.getMetadata().get("chunk_index"),
                    excerpt(text)));
        }
        return citations;
    }

    private static TokenUsage extractUsage(ChatClientResponse response) {
        if (response.chatResponse() == null || response.chatResponse().getMetadata() == null) {
            return null;
        }
        Usage usage = response.chatResponse().getMetadata().getUsage();
        if (usage == null) {
            return null;
        }
        return new TokenUsage(
                usage.getPromptTokens(),
                usage.getCompletionTokens(),
                usage.getTotalTokens());
    }

    private static String excerpt(String text) {
        String collapsed = text.replaceAll("\\s+", " ").trim();
        return collapsed.length() <= EXCERPT_CHARS
                ? collapsed
                : collapsed.substring(0, EXCERPT_CHARS) + "...";
    }
}
