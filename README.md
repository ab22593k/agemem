# Agentic Memory (AgeMem) MCP Server

An implementation of the **Agentic Memory** framework for LLM agents, based on the paper *"Agentic Memory: Learning Unified Long-Term and Short-Term Memory Management for Large Language Model Agents"*.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     MCP Server (Go)                        │
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
│    Weaviate     │    │  Context Tracker │
│  (Vector Store) │    │   (In-Memory)    │
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
| `summarize_context` | Compress text | `content`, `aggressive` |
| `filter_context` | Keep relevant lines | `content`, `keywords`, `keep_context` |

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
- **Context Tracking**: Monitor token usage and get compression recommendations

## Quick Start

### 1. Start Weaviate
```bash
make weaviate-up
```

### 2. Build & Run
```bash
make build && make run
```

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `WEAVIATE_HOST` | `localhost:8080` | Weaviate server address |
| `AGEMEM_VECTOR_SEARCH` | `false` | Enable semantic search |
| `AGEMEM_MAX_TOKENS` | `128000` | Context window limit for tracking |

## Testing

```bash
make test-unit        # Unit tests (no dependencies)
make test-integration # Integration tests (requires Weaviate)
```

## Project Structure

```
.
├── main.go                    # MCP Server entry point
├── internal/memory/
│   ├── manager.go             # LTM Manager (Weaviate)
│   ├── manager_test.go        # Integration tests
│   ├── stm.go                 # STM Manager (Context)
│   └── stm_test.go            # Unit tests
├── docker-compose.yml         # Weaviate container
└── Makefile                   # Development tasks
```

## Paper Implementation Status

| Paper Concept | Status | Notes |
|---------------|--------|-------|
| 6 Memory Tools | ✅ | ADD, UPDATE, DELETE, RETRIEVE, SUMMARY, FILTER |
| Persistent LTM | ✅ | Weaviate backend |
| Memory Quality | ✅ | Quality scoring with `rate_memory` |
| Usage Tracking | ✅ | Automatic retrieval counting |
| Session Scoping | ✅ | `session_id` and `user_id` support |
| Vector Search | ✅ | Optional with `AGEMEM_VECTOR_SEARCH=true` |
| Context Tracking | ✅ | Token estimation and threshold alerts |
| RL Training | ❌ | Out of scope for MCP server |

## Agent Usage Example

To enable autonomous memory management, include this in your agent's system prompt:

> You have access to `add_memory` and `retrieve_memory`.
> 1. **Before answering**, always search LTM: `retrieve_memory(query="user topic")`
> 2. **After answering**, save important facts: `add_memory(content="User likes X")`
> 3. Monitor context usage with `context_stats` and run `summarize_context` if usage > 80%.
