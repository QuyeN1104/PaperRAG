# AGENTS.md — Custom Models

Qwen3-VL Model encapsulation: multi-modal embedding, visual tools。

## Structure

```
src/custom/
├── qwen3_vl_embedding.py  # Qwen3-VL Embedded model (LangChain compatible）
├── qwen3_vl_base.py     # Shared base classes and utility functions
└── vision_utils.py      # visual processing tools
```

## Key Classes

| Class | Purpose | Extends |
|-------|---------|---------|
| `Qwen3VLEmbeddings` | multimodal embedding | `langchain_core.embeddings.Embeddings` |
| `BaseQwen3VLModel` | shared base class | - |

## Critical Patterns

### Embedding Interface
```python
# Unified API - input can be str or dict
embed_query(input: str | dict) -> list[float]
embed_documents(inputs: list[str | dict]) -> list[list[float]]

# Async versions
aembed_query(input: str | dict)
aembed_documents(inputs: list[str | dict])
```

### Multimodal Input Format
```python
{"text": "...", "image": "path/to/image.jpg"}
```

### PyTorch Rules
- `@torch.no_grad()` on inference methods
- Explicit `tensor.to(self.model.device)`
- Call `model.eval()` after loading
- Use `bfloat16` if supported, else `float16`
- Call `torch.cuda.empty_cache()` after batch loops

## VRAM Optimization
- from 19GB → ~10GB（~50% reduction）
- use `bfloat16`/`float16` precision
- Batch processing and cache cleaning
