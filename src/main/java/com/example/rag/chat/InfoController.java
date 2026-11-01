package com.example.rag.chat;

import java.util.LinkedHashMap;
import java.util.Map;

import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * Reports which chat provider is actually live.
 *
 * <p>Three chat starters sit on the classpath and {@code spring.ai.model.chat}
 * decides which one wins. That is easy to get wrong silently -- a typo in the
 * property leaves you on a different model than you think -- so the UI shows
 * the resolved value rather than the configured one. The model name is read
 * off the {@link ChatModel} bean itself, not from configuration.
 */
@RestController
public class InfoController {

    private final ChatModel chatModel;
    private final EmbeddingModel embeddingModel;
    private final String configuredProvider;

    public InfoController(
            ChatModel chatModel,
            EmbeddingModel embeddingModel,
            @Value("${spring.ai.model.chat:unknown}") String configuredProvider) {
        this.chatModel = chatModel;
        this.embeddingModel = embeddingModel;
        this.configuredProvider = configuredProvider;
    }

    @GetMapping("/api/info")
    public Map<String, Object> info() {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("chatProvider", configuredProvider);
        body.put("chatModel", resolveChatModelName());
        // Proves which ChatModel implementation actually won the wiring.
        body.put("chatImplementation", chatModel.getClass().getSimpleName());
        body.put("embeddingModel", embeddingModel.getClass().getSimpleName());
        body.put("embeddingDimensions", embeddingModel.dimensions());
        body.put("requiresApiKey", !"ollama".equalsIgnoreCase(configuredProvider));
        return body;
    }

    private String resolveChatModelName() {
        try {
            var options = chatModel.getDefaultOptions();
            if (options != null && options.getModel() != null) {
                return options.getModel();
            }
        } catch (RuntimeException ex) {
            // Not every provider populates default options; fall through.
        }
        return "unknown";
    }
}
