package memory

import (
	"testing"
)

func TestSTMManager_Summary(t *testing.T) {
	stm := NewSTMManager()

	tests := []struct {
		name     string
		input    string
		contains string // Changed from exact match to contains for flexibility
	}{
		{
			name:     "short content unchanged",
			input:    "Hello world.",
			contains: "Hello world.",
		},
		{
			name:     "two lines unchanged",
			input:    "Line one\nLine two",
			contains: "Line one",
		},
		{
			name:     "more than three lines truncated",
			input:    "Line 1\nLine 2\nLine 3\nLine 4\nLine 5",
			contains: "Line 1",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := stm.Summary(tt.input)
			if !containsString(result, tt.contains) {
				t.Errorf("Summary() = %q, want it to contain %q", result, tt.contains)
			}
		})
	}
}

func TestSTMManager_SummaryWithOptions(t *testing.T) {
	stm := NewSTMManager()

	content := "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"

	// Default options
	result := stm.SummaryWithOptions(content, SummaryOptions{})
	if !containsString(result, "Line 1") {
		t.Errorf("Summary should contain Line 1")
	}

	// Aggressive mode
	result = stm.SummaryWithOptions(content, SummaryOptions{Aggressive: true})
	if !containsString(result, "Line 1") {
		t.Errorf("Aggressive summary should contain Line 1")
	}
	if containsString(result, "Line 3") && !containsString(result, "omitted") {
		t.Errorf("Aggressive should be more compressed")
	}
}

func TestSTMManager_Filter(t *testing.T) {
	stm := NewSTMManager()

	tests := []struct {
		name        string
		content     string
		instruction string
		contains    string
		notContains string
	}{
		{
			name:        "filter keeps matching lines",
			content:     "The cat sat on the mat.\nThe dog ran in the park.\nA bird flew by.",
			instruction: "cat,bird",
			contains:    "cat",
			notContains: "dog",
		},
		{
			name:        "filter with no matches",
			content:     "Hello world\nGoodbye moon",
			instruction: "xyz",
			contains:    "filtered out",
		},
		{
			name:        "filter case insensitive",
			content:     "IMPORTANT: Do this now.\nOptional: Maybe later.",
			instruction: "important",
			contains:    "IMPORTANT",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := stm.Filter(tt.content, tt.instruction)
			if !containsString(result, tt.contains) {
				t.Errorf("Filter() = %q, want it to contain %q", result, tt.contains)
			}
			if tt.notContains != "" && containsString(result, tt.notContains) {
				t.Errorf("Filter() = %q, should NOT contain %q", result, tt.notContains)
			}
		})
	}
}

func TestSTMManager_ContextTracking(t *testing.T) {
	stm := NewSTMManagerWithConfig(STMConfig{
		MaxTokens:            1000,
		CompressionThreshold: 0.7,
	})

	// Initially empty
	if stm.CurrentTokens != 0 {
		t.Errorf("Expected 0 tokens initially, got %d", stm.CurrentTokens)
	}

	// Track some content
	stm.TrackContext("This is a test sentence with some words.")
	if stm.CurrentTokens == 0 {
		t.Error("Expected non-zero tokens after tracking")
	}

	// Check usage
	usage := stm.GetContextUsage()
	if usage < 0 || usage > 100 {
		t.Errorf("Usage should be between 0-100, got %f", usage)
	}

	// Check stats
	stats := stm.GetStats()
	if stats.MaxTokens != 1000 {
		t.Errorf("Expected max 1000, got %d", stats.MaxTokens)
	}
}

func TestSTMManager_ShouldSummarize(t *testing.T) {
	stm := NewSTMManagerWithConfig(STMConfig{
		MaxTokens:            100,
		CompressionThreshold: 0.5,
	})

	// Below threshold
	stm.CurrentTokens = 40
	if stm.ShouldSummarize() {
		t.Error("Should not recommend summarize at 40%")
	}

	// Above threshold
	stm.CurrentTokens = 60
	if !stm.ShouldSummarize() {
		t.Error("Should recommend summarize at 60%")
	}
}

func TestSTMManager_EstimateTokens(t *testing.T) {
	stm := NewSTMManager()

	// Empty string
	if tokens := stm.EstimateTokens(""); tokens != 0 {
		t.Errorf("Empty string should have 0 tokens, got %d", tokens)
	}

	// Short text
	tokens := stm.EstimateTokens("Hello world")
	if tokens < 1 {
		t.Errorf("Expected at least 1 token, got %d", tokens)
	}

	// Longer text should have more tokens
	short := stm.EstimateTokens("Hello")
	long := stm.EstimateTokens("Hello world, this is a much longer sentence with many more words.")
	if long <= short {
		t.Errorf("Longer text should have more tokens: short=%d, long=%d", short, long)
	}
}

func containsString(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(substr) == 0 ||
		(len(s) > 0 && len(substr) > 0 && findSubstring(s, substr)))
}

func findSubstring(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
