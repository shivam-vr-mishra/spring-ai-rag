package com.example.rag.ingest;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.springframework.core.io.ByteArrayResource;
import org.springframework.core.io.Resource;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.server.ResponseStatusException;

import com.example.rag.ingest.DocumentLibraryService.DocumentSummary;

@RestController
@RequestMapping("/api/documents")
public class IngestionController {

    private final IngestionService ingestionService;
    private final DocumentLibraryService libraryService;

    public IngestionController(IngestionService ingestionService,
                               DocumentLibraryService libraryService) {
        this.ingestionService = ingestionService;
        this.libraryService = libraryService;
    }

    /**
     * Upload one or many documents.
     *
     * <p>Accepts either {@code files} (repeatable, used by the UI) or
     * {@code file} (single, kept so the Layer 1 curl examples still work).
     *
     * <p>Each file is ingested independently: one malformed PDF reports its own
     * error without aborting the rest of the batch.
     */
    @PostMapping(consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public Map<String, Object> upload(
            @RequestParam(name = "files", required = false) MultipartFile[] files,
            @RequestParam(name = "file", required = false) MultipartFile file) {

        List<MultipartFile> uploads = new ArrayList<>();
        if (files != null) {
            uploads.addAll(List.of(files));
        }
        if (file != null) {
            uploads.add(file);
        }
        uploads.removeIf(f -> f == null || f.isEmpty());

        if (uploads.isEmpty()) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST,
                    "No files provided. Send multipart field 'files' (repeatable) or 'file'.");
        }

        List<Map<String, Object>> results = new ArrayList<>();
        int totalChunks = 0;
        int failed = 0;

        for (MultipartFile upload : uploads) {
            String filename = upload.getOriginalFilename() != null
                    ? upload.getOriginalFilename()
                    : "unnamed";
            Map<String, Object> entry = new LinkedHashMap<>();
            entry.put("source", filename);
            try {
                int chunks = ingestionService.ingest(toResource(upload, filename), filename);
                entry.put("chunks", chunks);
                entry.put("status", "ok");
                totalChunks += chunks;
            } catch (IOException | RuntimeException ex) {
                // Keep going -- a batch upload should not be all-or-nothing.
                entry.put("status", "error");
                entry.put("error", ex.getMessage());
                failed++;
            }
            results.add(entry);
        }

        Map<String, Object> body = new LinkedHashMap<>();
        body.put("uploaded", uploads.size() - failed);
        body.put("failed", failed);
        body.put("totalChunks", totalChunks);
        body.put("results", results);
        return body;
    }

    /** Everything currently ingested, with per-document chunk counts. */
    @GetMapping
    public Map<String, Object> list() {
        List<DocumentSummary> documents = libraryService.list();
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("documentCount", documents.size());
        body.put("totalChunks", libraryService.totalChunks());
        body.put("documents", documents);
        return body;
    }

    /** Removes every chunk belonging to one source document. */
    @DeleteMapping("/{source}")
    public Map<String, Object> delete(@PathVariable("source") String source) {
        int removed = ingestionService.deleteBySource(source);
        if (removed == 0) {
            throw new ResponseStatusException(HttpStatus.NOT_FOUND,
                    "No document named '" + source + "'");
        }
        return Map.of("source", source, "removedChunks", removed);
    }

    /**
     * Tika partly sniffs content type from the filename, which a plain
     * ByteArrayResource does not carry -- hence the override.
     */
    private static Resource toResource(MultipartFile upload, String filename) throws IOException {
        byte[] bytes = upload.getBytes();
        return new ByteArrayResource(bytes) {
            @Override
            public String getFilename() {
                return filename;
            }
        };
    }
}
