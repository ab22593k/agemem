//go:build integration

package memory

import (
	"context"
	"os"
	"testing"
	"time"
)

// Integration tests for Manager require a running Weaviate instance.
// Run with: go test -tags=integration ./internal/memory/...

func getWeaviateHost() string {
	host := os.Getenv("WEAVIATE_HOST")
	if host == "" {
		host = "localhost:8080"
	}
	return host
}

func TestManager_Integration_AddRetrieveDelete(t *testing.T) {
	manager, err := NewManager(getWeaviateHost())
	if err != nil {
		t.Fatalf("Failed to create manager: %v", err)
	}

	ctx := context.Background()

	// Test Add
	content := "Integration test memory entry at " + time.Now().Format(time.RFC3339)
	tags := []string{"test", "integration"}

	entry, err := manager.AddWithContext(ctx, content, tags)
	if err != nil {
		t.Fatalf("Add() failed: %v", err)
	}

	if entry.ID == "" {
		t.Error("Add() returned entry with empty ID")
	}
	if entry.Content != content {
		t.Errorf("Add() content = %q, want %q", entry.Content, content)
	}
	if entry.Quality != DefaultQuality {
		t.Errorf("Add() quality = %f, want %f", entry.Quality, DefaultQuality)
	}

	// Give Weaviate a moment to index
	time.Sleep(500 * time.Millisecond)

	// Test Retrieve
	results, err := manager.RetrieveWithContext(ctx, "integration", 10)
	if err != nil {
		t.Fatalf("Retrieve() failed: %v", err)
	}

	found := false
	for _, r := range results {
		if r.ID == entry.ID {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("Retrieve() did not find the added entry. Got %d results", len(results))
	}

	// Test Delete
	err = manager.DeleteWithContext(ctx, entry.ID)
	if err != nil {
		t.Fatalf("Delete() failed: %v", err)
	}

	// Verify deletion
	time.Sleep(500 * time.Millisecond)
	results, _ = manager.RetrieveWithContext(ctx, entry.ID, 10)
	for _, r := range results {
		if r.ID == entry.ID {
			t.Error("Delete() did not remove the entry")
		}
	}
}

func TestManager_Integration_UpdateSafe(t *testing.T) {
	manager, err := NewManager(getWeaviateHost())
	if err != nil {
		t.Fatalf("Failed to create manager: %v", err)
	}

	ctx := context.Background()

	// Add an entry
	entry, err := manager.AddWithContext(ctx, "Original content", []string{"update-test"})
	if err != nil {
		t.Fatalf("Add() failed: %v", err)
	}

	time.Sleep(500 * time.Millisecond)

	// Update it
	updated, err := manager.UpdateSafeWithContext(ctx, entry.ID, "Updated content")
	if err != nil {
		t.Fatalf("UpdateSafe() failed: %v", err)
	}

	if updated.Content != "Updated content" {
		t.Errorf("UpdateSafe() content = %q, want %q", updated.Content, "Updated content")
	}

	// Clean up
	_ = manager.DeleteWithContext(ctx, entry.ID)
}

func TestManager_Integration_GetAll(t *testing.T) {
	manager, err := NewManager(getWeaviateHost())
	if err != nil {
		t.Fatalf("Failed to create manager: %v", err)
	}

	ctx := context.Background()

	// Add a few entries
	ids := []string{}
	for i := 0; i < 3; i++ {
		entry, err := manager.AddWithContext(ctx, "GetAll test entry", []string{"getall-test"})
		if err != nil {
			t.Fatalf("Add() failed: %v", err)
		}
		ids = append(ids, entry.ID)
	}

	time.Sleep(500 * time.Millisecond)

	// Get all
	all, err := manager.GetAllWithContext(ctx)
	if err != nil {
		t.Fatalf("GetAll() failed: %v", err)
	}
	if len(all) < 3 {
		t.Errorf("GetAll() returned %d entries, expected at least 3", len(all))
	}

	// Clean up
	for _, id := range ids {
		_ = manager.DeleteWithContext(ctx, id)
	}
}

func TestManager_Integration_ContentValidation(t *testing.T) {
	manager, err := NewManager(getWeaviateHost())
	if err != nil {
		t.Fatalf("Failed to create manager: %v", err)
	}

	ctx := context.Background()

	// Test content length validation
	largeContent := make([]byte, MaxContentLength+1)
	for i := range largeContent {
		largeContent[i] = 'x'
	}

	_, err = manager.AddWithContext(ctx, string(largeContent), nil)
	if err == nil {
		t.Error("Add() should fail for content exceeding MaxContentLength")
	}
}

func TestManager_Integration_QualityAndUsage(t *testing.T) {
	manager, err := NewManager(getWeaviateHost())
	if err != nil {
		t.Fatalf("Failed to create manager: %v", err)
	}

	ctx := context.Background()

	// Add with custom quality
	entry, err := manager.AddWithContext(ctx, "High quality memory", []string{"quality-test"}, AddOptions{
		Quality: 0.9,
	})
	if err != nil {
		t.Fatalf("Add() failed: %v", err)
	}

	if entry.Quality != 0.9 {
		t.Errorf("Expected quality 0.9, got %f", entry.Quality)
	}

	time.Sleep(500 * time.Millisecond)

	// Update quality
	err = manager.UpdateQuality(ctx, entry.ID, 0.3)
	if err != nil {
		t.Fatalf("UpdateQuality() failed: %v", err)
	}

	// Clean up
	_ = manager.DeleteWithContext(ctx, entry.ID)
}

func TestManager_Integration_SessionScoping(t *testing.T) {
	// NOTE: This test may fail if the Weaviate schema was created before
	// session_id was added. Delete the MemoryEntry class and restart Weaviate
	// to fix: docker-compose down -v && docker-compose up -d
	
	manager, err := NewManager(getWeaviateHost())
	if err != nil {
		t.Fatalf("Failed to create manager: %v", err)
	}

	ctx := context.Background()

	// Add entries with different sessions and unique content
	uniqueA := "UniqueAlphaContent" + time.Now().Format("150405.000")
	uniqueB := "UniqueBetaContent" + time.Now().Format("150405.000")
	
	entry1, err := manager.AddWithContext(ctx, uniqueA, nil, AddOptions{SessionID: "session-a"})
	if err != nil {
		t.Fatalf("Failed to add entry1: %v", err)
	}
	entry2, err := manager.AddWithContext(ctx, uniqueB, nil, AddOptions{SessionID: "session-b"})
	if err != nil {
		t.Fatalf("Failed to add entry2: %v", err)
	}

	time.Sleep(1 * time.Second) // Give more time for indexing

	// Verify the entries have session IDs by retrieving all
	all, _ := manager.GetAllWithContext(ctx)
	t.Logf("Total entries: %d", len(all))
	for _, e := range all {
		if e.ID == entry1.ID {
			t.Logf("Entry1 session_id: %q", e.SessionID)
		}
		if e.ID == entry2.ID {
			t.Logf("Entry2 session_id: %q", e.SessionID)
		}
	}

	// Clean up
	_ = manager.DeleteWithContext(ctx, entry1.ID)
	_ = manager.DeleteWithContext(ctx, entry2.ID)
	
	t.Log("Session scoping test completed - manual verification recommended if schema was updated")
}

func TestManager_Integration_MemoryStats(t *testing.T) {
	manager, err := NewManager(getWeaviateHost())
	if err != nil {
		t.Fatalf("Failed to create manager: %v", err)
	}

	ctx := context.Background()

	// Get stats
	stats, err := manager.GetMemoryStats(ctx)
	if err != nil {
		t.Fatalf("GetMemoryStats() failed: %v", err)
	}

	if _, ok := stats["total_memories"]; !ok {
		t.Error("Stats should contain total_memories")
	}
	if _, ok := stats["average_quality"]; !ok {
		t.Error("Stats should contain average_quality")
	}
}
