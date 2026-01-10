package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"

	"agemem/internal/memory"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
)

func main() {
	// Initialize Memory Managers
	weaviateHost := os.Getenv("WEAVIATE_HOST")
	if weaviateHost == "" {
		weaviateHost = "localhost:8080"
	}

	// Check if vector search is enabled
	useVectorSearch := os.Getenv("AGEMEM_VECTOR_SEARCH") == "true"

	ltm, err := memory.NewManagerWithConfig(memory.ManagerConfig{
		Host:            weaviateHost,
		UseVectorSearch: useVectorSearch,
	})
	if err != nil {
		log.Fatalf("Failed to connect to Weaviate at %s: %v", weaviateHost, err)
	}

	// Configure STM with context limits
	maxTokens := 128000 // Default for Gemini 2.5 Flash
	if envMax := os.Getenv("AGEMEM_MAX_TOKENS"); envMax != "" {
		fmt.Sscanf(envMax, "%d", &maxTokens)
	}

	stm := memory.NewSTMManagerWithConfig(memory.STMConfig{
		MaxTokens:            maxTokens,
		CompressionThreshold: 0.7,
	})

	// Create MCP Server
	s := server.NewMCPServer(
		"Agentic Memory Server",
		"1.0.0",
		server.WithResourceCapabilities(true, true),
		server.WithToolCapabilities(true),
	)

	// Register tools
	registerLTMTools(s, ltm)
	registerSTMTools(s, stm)

	// Start the server
	log.Println("Starting Agentic Memory MCP Server...")
	if err := server.ServeStdio(s); err != nil {
		log.Fatalf("Server error: %v", err)
	}
}

// getStringArg safely extracts a string argument from the request.
func getStringArg(args interface{}, key string) string {
	m, ok := args.(map[string]interface{})
	if !ok {
		return ""
	}
	val, ok := m[key]
	if !ok {
		return ""
	}
	s, ok := val.(string)
	if !ok {
		return ""
	}
	return s
}

// getIntArg safely extracts an integer argument from the request.
func getIntArg(args interface{}, key string, defaultVal int) int {
	m, ok := args.(map[string]interface{})
	if !ok {
		return defaultVal
	}
	val, ok := m[key]
	if !ok {
		return defaultVal
	}
	switch v := val.(type) {
	case float64:
		return int(v)
	case int:
		return v
	default:
		return defaultVal
	}
}

// getFloatArg safely extracts a float argument from the request.
func getFloatArg(args interface{}, key string, defaultVal float64) float64 {
	m, ok := args.(map[string]interface{})
	if !ok {
		return defaultVal
	}
	val, ok := m[key]
	if !ok {
		return defaultVal
	}
	switch v := val.(type) {
	case float64:
		return v
	case int:
		return float64(v)
	default:
		return defaultVal
	}
}

// parseTags splits a comma-separated string into trimmed tags.
func parseTags(tagsStr string) []string {
	if tagsStr == "" {
		return nil
	}
	parts := strings.Split(tagsStr, ",")
	tags := make([]string, 0, len(parts))
	for _, p := range parts {
		t := strings.TrimSpace(p)
		if t != "" {
			tags = append(tags, t)
		}
	}
	return tags
}

// registerLTMTools registers Long-Term Memory tools.
func registerLTMTools(s *server.MCPServer, ltm *memory.Manager) {
	// Tool: add_memory
	s.AddTool(mcp.NewTool("add_memory",
		mcp.WithDescription("Add new knowledge to Long-Term Memory (LTM). Use this when the user provides important information that should be remembered for future sessions."),
		mcp.WithString("content", mcp.Required(), mcp.Description("The content to store in memory")),
		mcp.WithString("tags", mcp.Description("Comma-separated tags for categorization")),
		mcp.WithString("session_id", mcp.Description("Optional session identifier for scoping")),
		mcp.WithString("user_id", mcp.Description("Optional user identifier for scoping")),
		mcp.WithNumber("quality", mcp.Description("Initial quality score 0-1 (default 0.5)")),
	), func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		content := getStringArg(request.Params.Arguments, "content")
		if content == "" {
			return mcp.NewToolResultError("Content is required"), nil
		}

		tagsStr := getStringArg(request.Params.Arguments, "tags")
		tags := parseTags(tagsStr)

		opts := memory.AddOptions{
			SessionID: getStringArg(request.Params.Arguments, "session_id"),
			UserID:    getStringArg(request.Params.Arguments, "user_id"),
			Quality:   getFloatArg(request.Params.Arguments, "quality", 0),
		}

		entry, err := ltm.AddWithContext(ctx, content, tags, opts)
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("Failed to add memory: %v", err)), nil
		}

		return mcp.NewToolResultText(fmt.Sprintf("Memory added successfully. ID: %s (Quality: %.2f)", entry.ID, entry.Quality)), nil
	})

	// Tool: update_memory
	s.AddTool(mcp.NewTool("update_memory",
		mcp.WithDescription("Update an existing memory entry. Use this to refine or correct information in LTM."),
		mcp.WithString("id", mcp.Required(), mcp.Description("The ID of the memory to update")),
		mcp.WithString("content", mcp.Required(), mcp.Description("The new content")),
	), func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		id := getStringArg(request.Params.Arguments, "id")
		if id == "" {
			return mcp.NewToolResultError("ID is required"), nil
		}

		content := getStringArg(request.Params.Arguments, "content")
		if content == "" {
			return mcp.NewToolResultError("Content is required"), nil
		}

		_, err := ltm.UpdateSafeWithContext(ctx, id, content)
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("Failed to update memory: %v", err)), nil
		}

		return mcp.NewToolResultText(fmt.Sprintf("Memory %s updated.", id)), nil
	})

	// Tool: delete_memory
	s.AddTool(mcp.NewTool("delete_memory",
		mcp.WithDescription("Delete a memory entry from LTM. Use this to remove obsolete or incorrect information."),
		mcp.WithString("id", mcp.Required(), mcp.Description("The ID of the memory to delete")),
	), func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		id := getStringArg(request.Params.Arguments, "id")
		if id == "" {
			return mcp.NewToolResultError("ID is required"), nil
		}

		err := ltm.DeleteWithContext(ctx, id)
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("Failed to delete memory: %v", err)), nil
		}

		return mcp.NewToolResultText(fmt.Sprintf("Memory %s deleted.", id)), nil
	})

	// Tool: retrieve_memory
	s.AddTool(mcp.NewTool("retrieve_memory",
		mcp.WithDescription("Retrieve relevant memories from LTM based on a query. Use this to recall past information."),
		mcp.WithString("query", mcp.Required(), mcp.Description("The search query")),
		mcp.WithNumber("limit", mcp.Description("Maximum number of results (default 5, max 20)")),
		mcp.WithString("session_id", mcp.Description("Filter by session ID")),
		mcp.WithString("user_id", mcp.Description("Filter by user ID")),
		mcp.WithNumber("min_quality", mcp.Description("Minimum quality score to include (0-1)")),
	), func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		query := getStringArg(request.Params.Arguments, "query")
		if query == "" {
			return mcp.NewToolResultError("Query is required"), nil
		}

		limit := getIntArg(request.Params.Arguments, "limit", 5)
		if limit <= 0 {
			limit = 5
		}
		if limit > 20 {
			limit = 20
		}

		opts := memory.RetrieveOptions{
			SessionID:   getStringArg(request.Params.Arguments, "session_id"),
			UserID:      getStringArg(request.Params.Arguments, "user_id"),
			MinQuality:  getFloatArg(request.Params.Arguments, "min_quality", 0),
			UpdateUsage: true, // Track usage for quality metrics
		}

		results, err := ltm.RetrieveWithContext(ctx, query, limit, opts)
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("Retrieval failed: %v", err)), nil
		}

		if len(results) == 0 {
			return mcp.NewToolResultText("No relevant memories found."), nil
		}

		var sb strings.Builder
		sb.WriteString(fmt.Sprintf("Found %d memories:\n", len(results)))
		for _, r := range results {
			sb.WriteString(fmt.Sprintf("- [%s] (Q:%.2f, Used:%d) %s", r.ID, r.Quality, r.UsageCount, r.Content))
			if len(r.Tags) > 0 {
				sb.WriteString(fmt.Sprintf(" [Tags: %s]", strings.Join(r.Tags, ", ")))
			}
			sb.WriteString("\n")
		}

		return mcp.NewToolResultText(sb.String()), nil
	})

	// Tool: rate_memory (new - for adjusting quality)
	s.AddTool(mcp.NewTool("rate_memory",
		mcp.WithDescription("Adjust the quality rating of a memory. Higher quality memories are prioritized in retrieval."),
		mcp.WithString("id", mcp.Required(), mcp.Description("The ID of the memory to rate")),
		mcp.WithNumber("quality", mcp.Required(), mcp.Description("New quality score (0-1)")),
	), func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		id := getStringArg(request.Params.Arguments, "id")
		if id == "" {
			return mcp.NewToolResultError("ID is required"), nil
		}

		quality := getFloatArg(request.Params.Arguments, "quality", -1)
		if quality < 0 || quality > 1 {
			return mcp.NewToolResultError("Quality must be between 0 and 1"), nil
		}

		err := ltm.UpdateQuality(ctx, id, quality)
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("Failed to rate memory: %v", err)), nil
		}

		return mcp.NewToolResultText(fmt.Sprintf("Memory %s quality updated to %.2f", id, quality)), nil
	})

	// Tool: memory_stats (new - for monitoring)
	s.AddTool(mcp.NewTool("memory_stats",
		mcp.WithDescription("Get statistics about the long-term memory store."),
	), func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		stats, err := ltm.GetMemoryStats(ctx)
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("Failed to get stats: %v", err)), nil
		}

		data, _ := json.MarshalIndent(stats, "", "  ")
		return mcp.NewToolResultText(string(data)), nil
	})
}

// registerSTMTools registers Short-Term Memory tools.
func registerSTMTools(s *server.MCPServer, stm *memory.STMManager) {
	// Tool: summarize_context
	s.AddTool(mcp.NewTool("summarize_context",
		mcp.WithDescription("Summarize a block of text or context (STM). Use this to compress information and save context window space."),
		mcp.WithString("content", mcp.Required(), mcp.Description("The text to summarize")),
		mcp.WithBoolean("aggressive", mcp.Description("If true, use more aggressive compression")),
	), func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		content := getStringArg(request.Params.Arguments, "content")
		if content == "" {
			return mcp.NewToolResultError("Content is required"), nil
		}

		aggressive := false
		if m, ok := request.Params.Arguments.(map[string]interface{}); ok {
			if v, ok := m["aggressive"].(bool); ok {
				aggressive = v
			}
		}

		opts := memory.SummaryOptions{Aggressive: aggressive}
		summary := stm.SummaryWithOptions(content, opts)

		// Track context change
		originalTokens := stm.EstimateTokens(content)
		newTokens := stm.EstimateTokens(summary)
		savings := originalTokens - newTokens

		result := fmt.Sprintf("%s\n\n[Compression: %d → %d tokens, saved %d]", summary, originalTokens, newTokens, savings)
		return mcp.NewToolResultText(result), nil
	})

	// Tool: filter_context
	s.AddTool(mcp.NewTool("filter_context",
		mcp.WithDescription("Filter text to keep only relevant information based on keywords (STM). Use this to remove noise from context."),
		mcp.WithString("content", mcp.Required(), mcp.Description("The text to filter")),
		mcp.WithString("keywords", mcp.Required(), mcp.Description("Keywords to keep (comma separated)")),
		mcp.WithNumber("keep_context", mcp.Description("Number of surrounding lines to keep (default 0)")),
	), func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		content := getStringArg(request.Params.Arguments, "content")
		if content == "" {
			return mcp.NewToolResultError("Content is required"), nil
		}

		keywords := getStringArg(request.Params.Arguments, "keywords")
		if keywords == "" {
			return mcp.NewToolResultError("Keywords are required"), nil
		}

		keepContext := getIntArg(request.Params.Arguments, "keep_context", 0)

		opts := memory.FilterOptions{KeepContext: keepContext}
		filtered := stm.FilterWithOptions(content, keywords, opts)

		return mcp.NewToolResultText(filtered), nil
	})

	// Tool: context_stats (new - for monitoring context usage)
	s.AddTool(mcp.NewTool("context_stats",
		mcp.WithDescription("Get current context window usage statistics. Use this to decide when to summarize or filter."),
		mcp.WithString("current_context", mcp.Description("Optional: provide current context to analyze")),
	), func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		currentContext := getStringArg(request.Params.Arguments, "current_context")
		if currentContext != "" {
			stm.TrackContext(currentContext)
		}

		stats := stm.GetStats()
		data, _ := json.MarshalIndent(stats, "", "  ")

		result := string(data)
		if stats.ShouldSummarize {
			result += "\n\n⚠️ RECOMMENDATION: Context usage is high. Consider using summarize_context or filter_context."
		}

		return mcp.NewToolResultText(result), nil
	})
}
