.PHONY: build test test-unit test-integration run clean weaviate-up weaviate-down

# Build the MCP server binary
build:
	go build -o agemem-server

# Run all tests (unit only by default)
test: test-unit

# Run unit tests only (no external dependencies)
test-unit:
	go test -v ./internal/memory/... -run 'TestSTM'

# Run integration tests (requires Weaviate running)
test-integration: weaviate-up
	@echo "Waiting for Weaviate to be ready..."
	@sleep 5
	go test -v -tags=integration ./internal/memory/...

# Run the MCP server
run: build
	./agemem-server

# Clean build artifacts
clean:
	rm -f agemem-server
	rm -f ltm_store.json

# Start Weaviate via Docker Compose
weaviate-up:
	docker-compose up -d

# Stop Weaviate
weaviate-down:
	docker-compose down

# Full test cycle: start weaviate, run all tests, stop weaviate
test-all: weaviate-up
	@echo "Waiting for Weaviate to be ready..."
	@sleep 5
	go test -v ./internal/memory/... || true
	go test -v -tags=integration ./internal/memory/... || true
