package memory

import (
	"fmt"
	"strings"
	"unicode/utf8"
)

// STMManager handles Short-Term Memory operations (Context Management).
// It provides tools for summarizing and filtering context to manage the
// agent's working memory efficiently.
type STMManager struct {
	// MaxTokens is the estimated context window limit (for tracking only)
	MaxTokens int
	// CurrentTokens tracks estimated tokens in current context
	CurrentTokens int
	// CompressionThreshold is the percentage at which to recommend compression
	CompressionThreshold float64
}

// STMConfig holds configuration for the STM Manager.
type STMConfig struct {
	MaxTokens            int     // Default: 128000 (Gemini 2.5 Flash)
	CompressionThreshold float64 // Default: 0.7 (recommend compression at 70%)
}

// NewSTMManager creates a new STMManager with default settings.
func NewSTMManager() *STMManager {
	return NewSTMManagerWithConfig(STMConfig{
		MaxTokens:            128000,
		CompressionThreshold: 0.7,
	})
}

// NewSTMManagerWithConfig creates a new STMManager with custom configuration.
func NewSTMManagerWithConfig(cfg STMConfig) *STMManager {
	if cfg.MaxTokens <= 0 {
		cfg.MaxTokens = 128000
	}
	if cfg.CompressionThreshold <= 0 {
		cfg.CompressionThreshold = 0.7
	}
	return &STMManager{
		MaxTokens:            cfg.MaxTokens,
		CompressionThreshold: cfg.CompressionThreshold,
	}
}

// EstimateTokens provides a rough token count estimate.
// Rule of thumb: 1 token ≈ 4 characters for English text.
func (s *STMManager) EstimateTokens(content string) int {
	// More accurate: count words and add some overhead for punctuation
	words := len(strings.Fields(content))
	chars := utf8.RuneCountInString(content)
	
	// Hybrid estimate: words give better semantic count, chars handle punctuation
	return (words + chars/4) / 2
}

// TrackContext updates the current token count.
func (s *STMManager) TrackContext(content string) {
	s.CurrentTokens = s.EstimateTokens(content)
}

// AddToContext increments the token count.
func (s *STMManager) AddToContext(content string) {
	s.CurrentTokens += s.EstimateTokens(content)
}

// ShouldSummarize returns true if context usage exceeds the compression threshold.
func (s *STMManager) ShouldSummarize() bool {
	if s.MaxTokens == 0 {
		return false
	}
	usage := float64(s.CurrentTokens) / float64(s.MaxTokens)
	return usage >= s.CompressionThreshold
}

// GetContextUsage returns the current context usage as a percentage.
func (s *STMManager) GetContextUsage() float64 {
	if s.MaxTokens == 0 {
		return 0
	}
	return float64(s.CurrentTokens) / float64(s.MaxTokens) * 100
}

// ContextStats returns statistics about current context usage.
type ContextStats struct {
	CurrentTokens        int     `json:"current_tokens"`
	MaxTokens            int     `json:"max_tokens"`
	UsagePercent         float64 `json:"usage_percent"`
	ShouldSummarize      bool    `json:"should_summarize"`
	CompressionThreshold float64 `json:"compression_threshold"`
}

// GetStats returns current context statistics.
func (s *STMManager) GetStats() ContextStats {
	return ContextStats{
		CurrentTokens:        s.CurrentTokens,
		MaxTokens:            s.MaxTokens,
		UsagePercent:         s.GetContextUsage(),
		ShouldSummarize:      s.ShouldSummarize(),
		CompressionThreshold: s.CompressionThreshold * 100,
	}
}

// SummaryOptions configures the Summary operation.
type SummaryOptions struct {
	MaxLines     int  // Maximum lines to keep (default: 3)
	MaxSentences int  // Maximum sentences if lines aren't enough (default: 3)
	Aggressive   bool // If true, be more aggressive with compression
}

// Summary creates a concise version of the input content.
// In production, this should integrate with an LLM for intelligent summarization.
// Current implementation uses heuristics as a fallback.
func (s *STMManager) Summary(content string) string {
	return s.SummaryWithOptions(content, SummaryOptions{})
}

// SummaryWithOptions creates a summary with custom options.
func (s *STMManager) SummaryWithOptions(content string, opts SummaryOptions) string {
	if opts.MaxLines <= 0 {
		opts.MaxLines = 3
	}
	if opts.MaxSentences <= 0 {
		opts.MaxSentences = 3
	}
	if opts.Aggressive {
		opts.MaxLines = 1
		opts.MaxSentences = 2
	}

	// Strategy 1: Line-based truncation (good for structured content)
	lines := strings.Split(content, "\n")
	nonEmptyLines := make([]string, 0, len(lines))
	for _, line := range lines {
		if strings.TrimSpace(line) != "" {
			nonEmptyLines = append(nonEmptyLines, line)
		}
	}

	if len(nonEmptyLines) > opts.MaxLines {
		kept := strings.Join(nonEmptyLines[:opts.MaxLines], "\n")
		omitted := len(nonEmptyLines) - opts.MaxLines
		return kept + fmt.Sprintf("\n... (%d more lines omitted)", omitted)
	}

	// Strategy 2: Sentence-based truncation (good for prose)
	// Split on sentence boundaries (., !, ?)
	sentences := splitSentences(content)
	if len(sentences) > opts.MaxSentences {
		kept := strings.Join(sentences[:opts.MaxSentences], ". ") + "."
		omitted := len(sentences) - opts.MaxSentences
		return kept + fmt.Sprintf(" (%d more sentences omitted)", omitted)
	}

	// Content is already short enough
	return content
}

// splitSentences splits text into sentences.
func splitSentences(text string) []string {
	// Simple sentence splitting - production should use better NLP
	var sentences []string
	var current strings.Builder

	for _, r := range text {
		current.WriteRune(r)
		if r == '.' || r == '!' || r == '?' {
			s := strings.TrimSpace(current.String())
			if s != "" && len(s) > 1 { // Avoid single punctuation
				sentences = append(sentences, s)
			}
			current.Reset()
		}
	}

	// Don't forget trailing content without terminal punctuation
	if s := strings.TrimSpace(current.String()); s != "" {
		sentences = append(sentences, s)
	}

	return sentences
}

// FilterOptions configures the Filter operation.
type FilterOptions struct {
	CaseSensitive bool // If true, matching is case-sensitive
	KeepContext   int  // Number of surrounding lines to keep (default: 0)
}

// Filter removes irrelevant segments based on keywords.
// Keywords are comma-separated terms to KEEP.
func (s *STMManager) Filter(content, keywords string) string {
	return s.FilterWithOptions(content, keywords, FilterOptions{})
}

// FilterWithOptions filters content with custom options.
func (s *STMManager) FilterWithOptions(content, keywords string, opts FilterOptions) string {
	// Parse keywords
	kwList := parseKeywords(keywords, !opts.CaseSensitive)
	if len(kwList) == 0 {
		return "[No keywords provided for filtering]"
	}

	lines := strings.Split(content, "\n")
	matchIndices := make(map[int]bool)

	// Find matching lines
	for i, line := range lines {
		checkLine := line
		if !opts.CaseSensitive {
			checkLine = strings.ToLower(line)
		}

		for _, kw := range kwList {
			if strings.Contains(checkLine, kw) {
				matchIndices[i] = true
				// Also keep surrounding context
				for j := i - opts.KeepContext; j <= i+opts.KeepContext; j++ {
					if j >= 0 && j < len(lines) {
						matchIndices[j] = true
					}
				}
				break
			}
		}
	}

	if len(matchIndices) == 0 {
		return fmt.Sprintf("[Content filtered out - no matches for: %s]", keywords)
	}

	// Build result preserving order
	var kept []string
	for i, line := range lines {
		if matchIndices[i] {
			kept = append(kept, line)
		}
	}

	filtered := len(lines) - len(kept)
	result := strings.Join(kept, "\n")
	if filtered > 0 {
		result += fmt.Sprintf("\n[%d lines filtered out]", filtered)
	}

	return result
}

// parseKeywords splits and normalizes keyword string.
func parseKeywords(keywords string, lowercase bool) []string {
	parts := strings.Split(keywords, ",")
	result := make([]string, 0, len(parts))

	for _, p := range parts {
		kw := strings.TrimSpace(p)
		if kw == "" {
			continue
		}
		if lowercase {
			kw = strings.ToLower(kw)
		}
		result = append(result, kw)
	}

	return result
}
