# AGENTS.md — Tests

Testing infrastructure: pytest fixtures, evaluation pipeline, CI/CD integration.

## Structure

```
tests/
├── conftest.py                         # Shared fixtures (755 lines)
├── unit/                               # Unit tests (no external deps)
│   ├── test_vector_store_optimizations.py  # W1-B, W3-A, W3-D: Search performance optimization
│   ├── test_cache_w3b.py               # W3-B: Tokenizer/Query cache
│   ├── test_w4c_metrics_circuit.py     # W4-C: Monitoring indicators and circuit breakers
│   ├── test_agent_imports.py           # W4-B: Circular import eliminates validation
│   ├── test_agent_tools_retrieval_service.py  # W4-A: Tool layer decoupling
│   ├── test_cli_output.py              # W2-D: CLI Output specification
│   ├── test_paper_manager_paths.py     # W2-B: Unified path configuration
│   └── ...
├── integration/                        # Integration tests (requires Qdrant/GPU)
│   └── test_api_error_contract.py      # W2-A: API Error response specification
├── evaluation/                         # Offline evaluation pipeline
│   ├── runner.py                       # Evaluation orchestrator
│   ├── metrics.py                      # Retrieval metrics
│   ├── dataset.py                      # Dataset loader
│   └── thresholds.json                 # Pass/fail criteria
└── fixtures/                           # Test data
    └── pdfs/                           # Sample PDFs
```

## Test Categories

| Category | Mark | External Deps | When to Run |
|----------|------|---------------|-------------|
| Unit | `@pytest.mark.unit` | None | Always (CI default) |
| Integration | `@pytest.mark.integration` | Qdrant, GPU | Local/CI with services |
| Slow | `@pytest.mark.slow` | Varies | Optional |

## Running Tests

```bash
# Unit tests only (default for CI)
pytest tests -q -k "not integration"

# All tests (requires Qdrant + GPU)
pytest tests -q

# Specific category
pytest tests/unit -v
pytest tests/evaluation -v

# With explicit env isolation
env -u OPENAI_API_KEY -u EMBEDDING_MODEL pytest tests -q
```

## Fixtures (conftest.py)

| Fixture | Purpose |
|---------|---------|
| `test_env` | Auto-mocked env vars (session-scoped) |
| `temp_db` | Isolated SQLite database |
| `mock_vector_store` | Fake vector store (no GPU) |
| `sample_paper_payload` | Representative parsed paper data |
| `sample_pdf_path` | Minimal test PDF path |

## Architecture Improvement Tests

### Performance Optimization Tests
- **test_vector_store_optimizations.py**: verify N+1 Query elimination, batch_size configuration, connection reuse
  - `test_similarity_search_no_n_plus_1`: Make sure the retrieval is only executed once search + 1 Second-rate batch retrieve
  - `test_embedding_batch_size_config_default_and_override`: Configurable validation batch size
  - `test_qdrant_client_singleton_reuse`: Verify client singleton reuse

### Caching Tests
- **test_cache_w3b.py**: Validation Tokenizer and Query Cache
  - `test_tokenizer_initialized_once_across_calls`: Tokenizer Singleton
  - `test_query_cache_hit_avoids_second_vector_search`: Cache hits avoid repeated retrieval
  - `test_query_cache_hit_latency_under_10ms`: Cache hit latency <10ms

### Architecture Decoupling Tests
- **test_agent_imports.py**: Verify circular import elimination
- **test_agent_tools_retrieval_service.py**: Verification tool layer decouples vector storage through protocols

### Error Handling Tests
- **test_api_error_contract.py**: Verify API error response format is unified

## Evaluation Pipeline

**Offline evaluation** for RAG quality metrics:

```bash
# Run evaluation
python -m tests.evaluation.runner \
  --dataset tests/evaluation/dataset.json \
  --output reports/evaluation_report.json \
  --thresholds-file tests/evaluation/thresholds.json
```

**Metrics tracked:**
- Retrieval Hit Rate
- Page Hit Rate  
- Keyword Match Rate
- Citation Coverage Rate
- Current Version Leak Rate
- Failed Query Rate

**CI behavior:** Evaluation runs with `continue-on-error: true` — failures don't block builds.

## Testing Conventions

1. **Mock external deps**: Unit tests use mocks; never call real APIs
2. **Fixture reuse**: Shared fixtures in conftest.py, not per-file setup
3. **Env isolation**: Tests set `EMBEDDING_MODEL=mock-model`, `OPENAI_API_KEY=test-key-mock`
4. **Temp paths**: Use `/tmp/scholarrag_test_*` for test artifacts
5. **Deterministic**: Fixtures provide fixed seeds/UUIDs where possible

## CI Integration

GitHub Actions runs:
1. `ruff check .`
2. `pytest tests -k "not integration"` (unit tests only)
3. `python -m tests.evaluation.runner` (non-blocking)

See `.github/workflows/ci.yml` for full pipeline.
