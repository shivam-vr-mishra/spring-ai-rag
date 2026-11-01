package com.example.rag.chat;

import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.server.ResponseStatusException;

import com.example.rag.chat.ChatPayloads.ChatAnswer;
import com.example.rag.chat.ChatPayloads.ChatRequest;

@RestController
public class ChatController {

    private final ChatService chatService;

    public ChatController(ChatService chatService) {
        this.chatService = chatService;
    }

    /**
     * POST /api/chat
     * {"question": "...", "source": "handbook.txt", "topK": 4, "threshold": 0.1}
     *
     * <p>{@code source} is optional: retrieval strategy and document scope are
     * chosen automatically from the question (see {@link RetrievalPlanner}).
     * Supplying it pins the scope explicitly.
     *
     * <p>{@code topK} and {@code threshold} are optional and fall back to the
     * {@code rag.retrieval.*} configuration.
     */
    @PostMapping(path = "/api/chat",
            consumes = MediaType.APPLICATION_JSON_VALUE,
            produces = MediaType.APPLICATION_JSON_VALUE)
    public ChatAnswer chat(@RequestBody ChatRequest request) {
        if (request == null || request.question() == null || request.question().isBlank()) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "'question' must not be blank");
        }
        return chatService.ask(request.question(), request.source(),
                request.topK(), request.threshold());
    }
}
