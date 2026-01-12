# Agentic Memory (AgeMem) MCP Server

An implementation of **Agentic Memory** framework for LLM agents, based on paper *"Agentic Memory: Learning Unified Long-Term and Short-Term Memory Management for Large Language Model Agents"*.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     MCP Server (Python)                     │
├─────────────────────────────────────────────────────────────┤
│  LTM Tools          │  STM Tools          │  Monitoring     │
│  • add_memory       │  • summarize_context│  • memory_stats │
│  • update_memory    │  • filter_context   │  • context_stats│
│  • delete_memory    │                     │                 │
│  • retrieve_memory  │                     │                 │
│  • rate_memory      │                     │                 │
└─────────┬───────────┴──────────┬──────────┴─────────────────┘
          │                      │
          ▼                      ▼
┌─────────────────┐    ┌─────────────────┐
│    Weaviate     │    │    LangChain    │
│  (Vector Store) │    │  (Context Mgr)  │
└─────────────────┘    └─────────────────┘
```

## Tools

### Long-Term Memory (LTM)

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `add_memory` | Store new knowledge | `content`, `tags`, `session_id`, `user_id`, `quality` |
| `update_memory` | Update existing entry | `id`, `content` |
| `delete_memory` | Remove entry | `id` |
| `retrieve_memory` | Search memories | `query`, `limit`, `session_id`, `user_id`, `min_quality` |
| `rate_memory` | Adjust quality score | `id`, `quality` (0-1) |

### Short-Term Memory (STM)

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `summarize_context` | Compress text using LLM | `content`, `aggressive` |
| `filter_context` | Keep relevant lines/keywords | `content`, `keywords`, `keep_context` |

### Monitoring

| Tool | Description |
|------|-------------|
| `memory_stats` | Get LTM statistics (count, avg quality, total retrievals) |
| `context_stats` | Get STM usage (tokens, threshold, recommendations) |

## Features

- **Memory Quality Scoring**: Each memory has a 0-1 quality score that can be adjusted
- **Usage Tracking**: Retrieval counts and timestamps are tracked automatically
- **Session/User Scoping**: Memories can be scoped to specific sessions or users
- **Vector Search**: Optional semantic search when Weaviate has vectorizer enabled
- **Intelligent Context Management**: Uses LangChain (Gemini) for smart summarization and filtering

## Quick Start

### 1. Prerequisites
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- Weaviate instance (local or cloud)
- Google API Key (for Gemini)

### 2. Setup and Install
```bash
# uv will handle environment and dependencies automatically
uv sync
```

### 3. Configure Environment
```bash
export WEAVIATE_HOST="localhost:8080"
export GOOGLE_API_KEY="your-api-key"
export AGEMEM_VECTOR_SEARCH="true" # optional
```

### 4. Run Server
```bash
make run
# or
uv run python main.py
```

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `WEAVIATE_HOST` | `localhost:8080` | Weaviate server address |
| `AGEMEM_VECTOR_SEARCH` | `false` | Enable semantic search |
| `AGEMEM_MAX_TOKENS` | `128000` | Context window limit for tracking |
| `GOOGLE_API_KEY` | - | Required for LangChain + Gemini summarization |

## Project Structure

```
.
├── src/
│   ├── main.py               # MCP Server entry point
│   ├── ltm.py                # LTM Manager (Weaviate v4)
│   ├── stm.py                # STM Manager (LangChain)
│   └── bridge_cli.py         # CLI bridge for compaction
├── tests/
│   ├── test_ltm.py           # LTM integration tests
│   └── test_stm.py           # STM unit tests
├── requirements.txt          # Python dependencies
└── pyproject.toml            # Project metadata
```

## Paper Implementation Status

| Paper Concept | Status | Notes |
|---------------|--------|-------|
| 6 Memory Tools | ✅ | ADD, UPDATE, DELETE, RETRIEVE, SUMMARY, FILTER |
| Persistent LTM | ✅ | Weaviate backend (v4) |
| Memory Quality | ✅ | Quality scoring with `rate_memory` |
| Usage Tracking | ✅ | Automatic retrieval counting |
| Session Scoping | ✅ | `session_id` and `user_id` support |
| Vector Search | ✅ | Optional with `AGEMEM_VECTOR_SEARCH=true` |
| Context Tracking | ✅ | Token estimation and threshold alerts |
| Intelligent STM | ✅ | LangChain + Gemini for summarization |
| RL Training | ❌ | Out of scope for MCP server |

## Agent Usage Example

To enable autonomous memory management, include this in your agent's system prompt:

> You have access to `add_memory` and `retrieve_memory`.
> 1. **Before answering**, always search LTM: `retrieve_memory(query="user topic")`
> 2. **After answering**, save important facts: `add_memory(content="User likes X")`
> 3. Monitor context usage with `context_stats` and run `summarize_context` if usage > 80%.

## Testing

### Run Tests
```bash
make test                 # Unit tests (STM only)
make test-integration     # Integration tests (requires Weaviate)
uv run pytest -v          # All tests
```

### Manual Testing
```bash
# Start Weaviate
docker-compose up -d

# Run Python server
python main.py

# Test with MCP client (Claude Desktop or other MCP-compatible client)
```

## License

MIT
