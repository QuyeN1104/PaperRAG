# AGENTS.md — API Module

FastAPI rear end：RESTful endpoints, SSE Streaming response, service layer pattern。

## Structure

```
api/
├── main.py              # FastAPI Application factory, life cycle management
├── config.py            # API Configuration（host, port, upload dir）
├── schemas.py           # Pydantic v2 ask/response model
├── models.py            # SQLAlchemy Database model
├── database.py          # Database connection management
├── routes/
│   ├── papers.py        # Paper upload/delete/list
│   ├── query.py         # Query Endpoint (SSE Streaming）
│   └── conversations.py # Conversation history management
└── services/
    ├── paper_service.py   # Thesis Business Logic
    ├── query_service.py   # Query business logic
    └── conversation_service.py
```

## Key Patterns

### Route Handler Pattern
- Routes Keep it thin, the business logic is in services/
- Validate using Pydantic v2 models
- Client error `HTTPException(status_code=400, detail="...")`
- Only return safe and brief information in external error responses; do not disclose internal exception details

### SSE Streaming
```python
from fastapi.responses import StreamingResponse

@router.post("/stream")
async def stream_query(request: QueryRequest):
    return StreamingResponse(
        generate_events(request.question),
        media_type="text/event-stream"
    )
```

### Service Layer
```python
class PaperService:
    def _get_vector_store(self):  # Delayed import
        from src.rag.vector_store import get_vector_store
        return get_vector_store()
```

## Critical Constraints

1. **Lazy Imports**: Import inside service method `get_vector_store()`，Not at the module level
2. **Temp Files**: PDF Upload to temporary directory and clean up after processing
3. **CORS**: configured as `localhost:5173`（Front-end development server）
4. **Errors**: API routes Only return sanitized messages, do not reveal them traceback / exception repr
5. **Exception Handling**: Using a unified exception hierarchy, the API layer is uniformly converted to HTTPException

## Error Handling (W2-A)

### Exception Hierarchy
```python
from src.utils.exceptions import AppError, ValidationError, NotFoundError

# Services raise specific exceptions
if not pdf_name:
    raise ValidationError("pdf_name is required")

if paper is None:
    raise NotFoundError(f"Paper '{pdf_name}' not found")
```

### API Error Response Format
```python
# Unified error response format
{"error": {"code": "VALIDATION_ERROR", "message": "..."}}
```

## Background Task Execution (W1-A, W3-C)

### Database Lease Pattern
```python
# Use database leases to prevent duplicate processing by multiple workers
USE_DB_JOB_LEASE=true  # Enable lease mechanism
JOB_LEASE_TTL_SECONDS=300  # Lease expiration time
```

### Executor Configuration
```python
# Configurable actuator type
EXECUTOR_TYPE=thread  # or 'process' for CPU-bound tasks
BACKGROUND_EXECUTOR_WORKERS=2  # Number of parallel workers
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/papers/upload` | upload PDF |
| GET | `/api/papers` | list papers |
| DELETE | `/api/papers/{name}` | Delete paper |
| POST | `/api/query/stream` | SSE Streaming query |
| GET | `/api/conversations` | Get conversation |
| DELETE | `/api/conversations/{id}` | Delete conversation |
