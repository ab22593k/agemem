.PHONY: test test-unit test-integration run clean weaviate-up weaviate-down

# Run Python tests
test: test-unit

# Run STM unit tests (no external dependencies)
test-unit:
	PYTHONPATH=/home/abdelwahab/azul/python/agemem uv run pytest tests/test_stm.py -v

# Run LTM integration tests (requires Weaviate running)
test-integration: weaviate-up
	@echo "Waiting for Weaviate to be ready..."
	@sleep 5
	uv run pytest tests/ltm.py -v
	docker-compose down

# Run all tests
test-all: test-unit test-integration

# Run MCP server
run:
	uv run python main.py

# Clean build artifacts
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .venv

# Start Weaviate via Docker Compose
weaviate-up:
	docker-compose up -d

# Stop Weaviate
weaviate-down:
	docker-compose down
