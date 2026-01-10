package memory

import (
	"context"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/weaviate/weaviate-go-client/v4/weaviate"
	"github.com/weaviate/weaviate-go-client/v4/weaviate/auth"
	"github.com/weaviate/weaviate-go-client/v4/weaviate/filters"
	"github.com/weaviate/weaviate-go-client/v4/weaviate/graphql"
	"github.com/weaviate/weaviate/entities/models"
)

// MemoryEntry represents a single memory stored in LTM.
type MemoryEntry struct {
	ID         string    `json:"id"`
	Content    string    `json:"content"`
	Tags       []string  `json:"tags"`
	SessionID  string    `json:"session_id,omitempty"`  // Optional scoping
	UserID     string    `json:"user_id,omitempty"`     // Optional scoping
	Quality    float64   `json:"quality"`               // 0.0-1.0 quality score
	UsageCount int       `json:"usage_count"`           // How often retrieved
	CreatedAt  time.Time `json:"created_at"`
	UpdatedAt  time.Time `json:"updated_at"`
	LastUsedAt time.Time `json:"last_used_at,omitempty"`
}

// Manager handles Long-Term Memory operations backed by Weaviate.
type Manager struct {
	client          *weaviate.Client
	useVectorSearch bool // Whether to use semantic search
}

// ClassName is the Weaviate class name for memory entries.
const ClassName = "MemoryEntry"

// MaxContentLength is the maximum allowed content size (100KB).
const MaxContentLength = 100000

// DefaultQuality is the initial quality score for new memories.
const DefaultQuality = 0.5

// standardFields defines the GraphQL fields to retrieve.
var standardFields = []graphql.Field{
	{Name: "content"},
	{Name: "tags"},
	{Name: "original_id"},
	{Name: "session_id"},
	{Name: "user_id"},
	{Name: "quality"},
	{Name: "usage_count"},
	{Name: "created_at"},
	{Name: "updated_at"},
	{Name: "last_used_at"},
}

// ManagerConfig holds configuration for the Manager.
type ManagerConfig struct {
	Host            string
	UseVectorSearch bool // Set true if Weaviate has a vectorizer enabled
}

// NewManager creates a new Manager connected to the specified Weaviate host.
func NewManager(host string) (*Manager, error) {
	return NewManagerWithConfig(ManagerConfig{
		Host:            host,
		UseVectorSearch: false, // Default to keyword search for backward compat
	})
}

// NewManagerWithConfig creates a new Manager with custom configuration.
func NewManagerWithConfig(cfg ManagerConfig) (*Manager, error) {
	weaviateCfg := weaviate.Config{
		Host:       cfg.Host,
		Scheme:     "http",
		AuthConfig: auth.ApiKey{Value: ""}, // Anonymous access
		Headers:    nil,
	}

	client, err := weaviate.NewClient(weaviateCfg)
	if err != nil {
		return nil, err
	}

	m := &Manager{
		client:          client,
		useVectorSearch: cfg.UseVectorSearch,
	}

	ctx := context.Background()
	if err := m.ensureSchema(ctx); err != nil {
		return nil, fmt.Errorf("schema init failed: %v", err)
	}

	return m, nil
}

func (m *Manager) ensureSchema(ctx context.Context) error {
	exists, err := m.client.Schema().ClassExistenceChecker().WithClassName(ClassName).Do(ctx)
	if err != nil {
		return err
	}
	if exists {
		return nil
	}

	vectorizer := "none"
	if m.useVectorSearch {
		vectorizer = "text2vec-transformers" // Or configurable
	}

	class := &models.Class{
		Class:      ClassName,
		Vectorizer: vectorizer,
		Properties: []*models.Property{
			{Name: "content", DataType: []string{"text"}},
			{Name: "tags", DataType: []string{"text[]"}},
			{Name: "original_id", DataType: []string{"string"}},
			{Name: "session_id", DataType: []string{"string"}},
			{Name: "user_id", DataType: []string{"string"}},
			{Name: "quality", DataType: []string{"number"}},
			{Name: "usage_count", DataType: []string{"int"}},
			{Name: "created_at", DataType: []string{"date"}},
			{Name: "updated_at", DataType: []string{"date"}},
			{Name: "last_used_at", DataType: []string{"date"}},
		},
	}

	return m.client.Schema().ClassCreator().WithClass(class).Do(ctx)
}

// AddOptions configures the Add operation.
type AddOptions struct {
	SessionID string
	UserID    string
	Quality   float64 // If 0, defaults to DefaultQuality
}

// Add creates a new memory entry.
func (m *Manager) Add(content string, tags []string) (*MemoryEntry, error) {
	return m.AddWithContext(context.Background(), content, tags, AddOptions{})
}

// AddWithContext creates a new memory entry with options.
func (m *Manager) AddWithContext(ctx context.Context, content string, tags []string, opts ...AddOptions) (*MemoryEntry, error) {
	if len(content) > MaxContentLength {
		return nil, fmt.Errorf("content exceeds maximum length of %d bytes", MaxContentLength)
	}

	var opt AddOptions
	if len(opts) > 0 {
		opt = opts[0]
	}

	quality := opt.Quality
	if quality <= 0 {
		quality = DefaultQuality
	}

	id := uuid.New().String()
	now := time.Now()

	props := map[string]interface{}{
		"content":      content,
		"tags":         tags,
		"original_id":  id,
		"session_id":   opt.SessionID,
		"user_id":      opt.UserID,
		"quality":      quality,
		"usage_count":  0,
		"created_at":   now,
		"updated_at":   now,
		"last_used_at": now,
	}

	_, err := m.client.Data().Creator().
		WithClassName(ClassName).
		WithProperties(props).
		Do(ctx)

	if err != nil {
		return nil, err
	}

	return &MemoryEntry{
		ID:         id,
		Content:    content,
		Tags:       tags,
		SessionID:  opt.SessionID,
		UserID:     opt.UserID,
		Quality:    quality,
		UsageCount: 0,
		CreatedAt:  now,
		UpdatedAt:  now,
		LastUsedAt: now,
	}, nil
}

// UpdateSafe updates an existing memory entry by its original ID.
func (m *Manager) UpdateSafe(id string, content string) (*MemoryEntry, error) {
	return m.UpdateSafeWithContext(context.Background(), id, content)
}

// UpdateSafeWithContext updates an existing memory entry with a custom context.
func (m *Manager) UpdateSafeWithContext(ctx context.Context, id string, content string) (*MemoryEntry, error) {
	if len(content) > MaxContentLength {
		return nil, fmt.Errorf("content exceeds maximum length of %d bytes", MaxContentLength)
	}

	uuidStr, err := m.findWeaviateID(ctx, id)
	if err != nil {
		return nil, fmt.Errorf("memory not found: %s", id)
	}

	now := time.Now()
	err = m.client.Data().Updater().
		WithClassName(ClassName).
		WithID(uuidStr).
		WithProperties(map[string]interface{}{
			"content":    content,
			"updated_at": now,
		}).
		Do(ctx)

	if err != nil {
		return nil, err
	}

	return &MemoryEntry{
		ID:        id,
		Content:   content,
		UpdatedAt: now,
	}, nil
}

// Delete removes a memory entry by its original ID.
func (m *Manager) Delete(id string) error {
	return m.DeleteWithContext(context.Background(), id)
}

// DeleteWithContext removes a memory entry with a custom context.
func (m *Manager) DeleteWithContext(ctx context.Context, id string) error {
	uuidStr, err := m.findWeaviateID(ctx, id)
	if err != nil {
		return fmt.Errorf("memory not found: %s", id)
	}

	return m.client.Data().Deleter().
		WithClassName(ClassName).
		WithID(uuidStr).
		Do(ctx)
}

// RetrieveOptions configures the Retrieve operation.
type RetrieveOptions struct {
	SessionID      string
	UserID         string
	MinQuality     float64 // Filter by minimum quality
	UpdateUsage    bool    // If true, increment usage_count for results
}

// Retrieve searches for memories matching the query.
func (m *Manager) Retrieve(query string, limit int) ([]*MemoryEntry, error) {
	return m.RetrieveWithContext(context.Background(), query, limit, RetrieveOptions{})
}

// RetrieveWithContext searches for memories with options.
func (m *Manager) RetrieveWithContext(ctx context.Context, query string, limit int, opts ...RetrieveOptions) ([]*MemoryEntry, error) {
	var opt RetrieveOptions
	if len(opts) > 0 {
		opt = opts[0]
	}

	builder := m.client.GraphQL().Get().
		WithClassName(ClassName).
		WithFields(standardFields...).
		WithLimit(limit)

	// Build filters
	var whereFilters []*filters.WhereBuilder

	// Scope filtering
	if opt.SessionID != "" {
		whereFilters = append(whereFilters, filters.Where().
			WithPath([]string{"session_id"}).
			WithOperator(filters.Equal).
			WithValueString(opt.SessionID))
	}
	if opt.UserID != "" {
		whereFilters = append(whereFilters, filters.Where().
			WithPath([]string{"user_id"}).
			WithOperator(filters.Equal).
			WithValueString(opt.UserID))
	}
	if opt.MinQuality > 0 {
		whereFilters = append(whereFilters, filters.Where().
			WithPath([]string{"quality"}).
			WithOperator(filters.GreaterThan).
			WithValueNumber(opt.MinQuality))
	}

	// Search strategy: vector or keyword
	if m.useVectorSearch && query != "" {
		// Use hybrid search if vectors are available
		// Note: The Go client API for hybrid search may vary by version
		// This is a basic NearText approach
		builder = builder.WithNearText((&graphql.NearTextArgumentBuilder{}).WithConcepts([]string{query}))
	} else if query != "" {
		// Keyword search fallback
		whereFilters = append(whereFilters, filters.Where().
			WithPath([]string{"content"}).
			WithOperator(filters.Like).
			WithValueString(fmt.Sprintf("*%s*", query)))
	}

	// Combine filters if multiple
	if len(whereFilters) == 1 {
		builder = builder.WithWhere(whereFilters[0])
	} else if len(whereFilters) > 1 {
		combined := filters.Where().
			WithOperator(filters.And).
			WithOperands(whereFilters)
		builder = builder.WithWhere(combined)
	}

	resp, err := builder.Do(ctx)
	if err != nil {
		return nil, err
	}

	results, err := parseResults(resp)
	if err != nil {
		return nil, err
	}

	// Update usage if requested
	if opt.UpdateUsage && len(results) > 0 {
		m.incrementUsage(ctx, results)
	}

	return results, nil
}

// incrementUsage updates usage_count and last_used_at for retrieved memories.
func (m *Manager) incrementUsage(ctx context.Context, entries []*MemoryEntry) {
	now := time.Now()
	for _, entry := range entries {
		uuidStr, err := m.findWeaviateID(ctx, entry.ID)
		if err != nil {
			continue
		}
		_ = m.client.Data().Updater().
			WithClassName(ClassName).
			WithID(uuidStr).
			WithProperties(map[string]interface{}{
				"usage_count":  entry.UsageCount + 1,
				"last_used_at": now,
			}).
			Do(ctx)
	}
}

// UpdateQuality adjusts the quality score of a memory.
func (m *Manager) UpdateQuality(ctx context.Context, id string, quality float64) error {
	if quality < 0 || quality > 1 {
		return fmt.Errorf("quality must be between 0 and 1")
	}

	uuidStr, err := m.findWeaviateID(ctx, id)
	if err != nil {
		return fmt.Errorf("memory not found: %s", id)
	}

	return m.client.Data().Updater().
		WithClassName(ClassName).
		WithID(uuidStr).
		WithProperties(map[string]interface{}{
			"quality":    quality,
			"updated_at": time.Now(),
		}).
		Do(ctx)
}

// GetAll returns all memory entries (up to 100).
func (m *Manager) GetAll() ([]*MemoryEntry, error) {
	return m.GetAllWithContext(context.Background())
}

// GetAllWithContext returns all memory entries with a custom context.
func (m *Manager) GetAllWithContext(ctx context.Context) ([]*MemoryEntry, error) {
	resp, err := m.client.GraphQL().Get().
		WithClassName(ClassName).
		WithFields(standardFields...).
		WithLimit(100).
		Do(ctx)

	if err != nil {
		return nil, err
	}

	return parseResults(resp)
}

// GetMemoryStats returns statistics about the memory store.
func (m *Manager) GetMemoryStats(ctx context.Context) (map[string]interface{}, error) {
	all, err := m.GetAllWithContext(ctx)
	if err != nil {
		return nil, err
	}

	var totalQuality float64
	var totalUsage int
	for _, e := range all {
		totalQuality += e.Quality
		totalUsage += e.UsageCount
	}

	avgQuality := 0.0
	if len(all) > 0 {
		avgQuality = totalQuality / float64(len(all))
	}

	return map[string]interface{}{
		"total_memories":   len(all),
		"average_quality":  avgQuality,
		"total_retrievals": totalUsage,
	}, nil
}

// findWeaviateID looks up the Weaviate internal UUID for a given original_id.
func (m *Manager) findWeaviateID(ctx context.Context, originalID string) (string, error) {
	res, err := m.client.GraphQL().Get().
		WithClassName(ClassName).
		WithFields(graphql.Field{Name: "_additional", Fields: []graphql.Field{{Name: "id"}}}).
		WithWhere(filters.Where().
			WithPath([]string{"original_id"}).
			WithOperator(filters.Equal).
			WithValueString(originalID),
		).
		Do(ctx)

	if err != nil {
		return "", err
	}

	return extractWeaviateID(res)
}

// extractWeaviateID safely extracts the Weaviate UUID from a GraphQL response.
func extractWeaviateID(res *models.GraphQLResponse) (string, error) {
	if res == nil || res.Data == nil {
		return "", fmt.Errorf("empty response")
	}

	data, ok := res.Data["Get"].(map[string]interface{})
	if !ok {
		return "", fmt.Errorf("unexpected response format: missing Get")
	}

	objects, ok := data[ClassName].([]interface{})
	if !ok {
		return "", fmt.Errorf("unexpected response format: missing %s", ClassName)
	}

	if len(objects) == 0 {
		return "", fmt.Errorf("not found")
	}

	obj, ok := objects[0].(map[string]interface{})
	if !ok {
		return "", fmt.Errorf("unexpected object format")
	}

	add, ok := obj["_additional"].(map[string]interface{})
	if !ok {
		return "", fmt.Errorf("missing _additional field")
	}

	id, ok := add["id"].(string)
	if !ok {
		return "", fmt.Errorf("missing id field")
	}

	return id, nil
}

// parseResults converts a GraphQL response to MemoryEntry slice.
func parseResults(res *models.GraphQLResponse) ([]*MemoryEntry, error) {
	var entries []*MemoryEntry

	if res == nil || res.Data == nil {
		return entries, nil
	}

	data, ok := res.Data["Get"].(map[string]interface{})
	if !ok {
		return entries, nil
	}

	objects, ok := data[ClassName].([]interface{})
	if !ok {
		return entries, nil
	}

	for _, o := range objects {
		obj, ok := o.(map[string]interface{})
		if !ok {
			continue
		}

		entry := &MemoryEntry{}
		if val, ok := obj["content"].(string); ok {
			entry.Content = val
		}
		if val, ok := obj["original_id"].(string); ok {
			entry.ID = val
		}
		if val, ok := obj["session_id"].(string); ok {
			entry.SessionID = val
		}
		if val, ok := obj["user_id"].(string); ok {
			entry.UserID = val
		}
		if val, ok := obj["quality"].(float64); ok {
			entry.Quality = val
		}
		if val, ok := obj["usage_count"].(float64); ok {
			entry.UsageCount = int(val)
		}
		if val, ok := obj["tags"].([]interface{}); ok {
			for _, t := range val {
				if s, ok := t.(string); ok {
					entry.Tags = append(entry.Tags, s)
				}
			}
		}
		if val, ok := obj["created_at"].(string); ok {
			entry.CreatedAt, _ = time.Parse(time.RFC3339, val)
		}
		if val, ok := obj["updated_at"].(string); ok {
			entry.UpdatedAt, _ = time.Parse(time.RFC3339, val)
		}
		if val, ok := obj["last_used_at"].(string); ok {
			entry.LastUsedAt, _ = time.Parse(time.RFC3339, val)
		}

		entries = append(entries, entry)
	}

	return entries, nil
}
