package com.example.rag.search;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class SearchController {

    private final SearchService searchService;

    public SearchController(SearchService searchService) {
        this.searchService = searchService;
    }

    /**
     * GET /api/search?q=when+can+we+deploy&topK=4&threshold=0.0
     *
     * <p>The echoed request parameters are intentional: when you are sweeping
     * thresholds, the response should record which settings produced it.
     */
    @GetMapping("/api/search")
    public Map<String, Object> search(
            @RequestParam("q") String query,
            @RequestParam(name = "topK", defaultValue = "4") int topK,
            @RequestParam(name = "threshold", defaultValue = "0.0") double threshold,
            @RequestParam(name = "source", required = false) String source) {

        List<ScoredChunk> results = searchService.search(query, topK, threshold, source);

        Map<String, Object> body = new LinkedHashMap<>();
        body.put("query", query);
        body.put("topK", topK);
        body.put("threshold", threshold);
        body.put("source", source);
        body.put("resultCount", results.size());
        body.put("results", results);
        return body;
    }
}
