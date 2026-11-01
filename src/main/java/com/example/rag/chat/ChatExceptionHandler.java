package com.example.rag.chat;

import java.util.LinkedHashMap;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

import com.anthropic.errors.AnthropicServiceException;

/**
 * Turns Anthropic SDK failures into responses a caller can act on.
 *
 * <p>Without this, a bad API key surfaces as a bare HTTP 500 with no body --
 * indistinguishable from a bug in the retrieval code. The distinction matters:
 * Layers 1 and 2 run fine without any credentials, so "the chat endpoint 500s"
 * is a question of *where* it broke.
 */
@RestControllerAdvice
public class ChatExceptionHandler {

    private static final Logger log = LoggerFactory.getLogger(ChatExceptionHandler.class);

    @ExceptionHandler(AnthropicServiceException.class)
    public ResponseEntity<Map<String, Object>> handleAnthropic(AnthropicServiceException ex) {
        int status = ex.statusCode();
        log.error("Anthropic API call failed with status {}", status, ex);

        String hint = switch (status) {
            case 401 -> "ANTHROPIC_API_KEY is missing or invalid. "
                    + "Export a valid key and restart. Layers 1-2 (/api/documents, /api/search) "
                    + "do not need one.";
            case 403 -> "The API key is valid but lacks permission for this model.";
            case 404 -> "Unknown model id. Check spring.ai.anthropic.chat.model.";
            case 429 -> "Rate limited by Anthropic. Retry after a short delay.";
            default -> status >= 500
                    ? "Anthropic service error. Safe to retry."
                    : "Request rejected by the Anthropic API.";
        };

        // 502 for upstream faults -- this service is healthy, its dependency
        // is not. A 500 would wrongly implicate our own code.
        HttpStatus responseStatus = (status == 429)
                ? HttpStatus.TOO_MANY_REQUESTS
                : HttpStatus.BAD_GATEWAY;

        Map<String, Object> body = new LinkedHashMap<>();
        body.put("error", "anthropic_api_error");
        body.put("upstreamStatus", status);
        body.put("hint", hint);
        return ResponseEntity.status(responseStatus).body(body);
    }
}
