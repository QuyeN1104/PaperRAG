# ScholarRAG System architecture documentation

> **Notice**：This document was originally titled `DEVELOPMENT_PLAN.md`，The development plan described has**All done**（2026March of the year). It is now reserved as a system architecture document for developers’ reference.。

## 1. Project Overview

ScholarRAG is a multi-modal RAG system for academic documents, including a complete web front-end and REST API service layer。

**target users**：individual researcher/scholar  
**Deployment method**：Run locally (no need Docker）  
**development status**：✅ Completed (contains all 10 tasks of the necessary feature roadmap）

---

## 2. Technology stack

### rear end
| technology | Version | use |
|------|------|------|
| FastAPI | ≥0.100 | Web frame |
| Uvicorn | ≥0.20 | ASGI server |
| python-multipart | - | File upload |
| sse-starlette | - | Server-Sent Events |

### front end
| technology | Version | use |
|------|------|------|
| React | 18 | UI frame |
| TypeScript | 5.x | type safety |
| Vite | 5.x | Build tools |
| Tailwind CSS | 3.x | style |
| shadcn/ui | - | Component library |
| Zustand | - | Status management |
| React Router | 6.x | routing |
| @tanstack/react-query | - | Server status |

### design style
- **Simple document style**（similar Notion/Obsidian）
- Dark/Light theme support
- Responsive design (mobile-friendly)）

---

## 3. Directory structure

```
ScholarRAG/
├── api/                           # FastAPI service layer
│   ├── __init__.py
│   ├── main.py                    # FastAPI Application entrance
│   ├── config.py                  # API Configuration
│   ├── deps.py                    # dependency injection
│   ├── schemas.py                 # Pydantic Model
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── papers.py              # Paper CRUD Routing
│   │   └── query.py               # Q&A routing (SSE)
│   └── services/
│       ├── __init__.py
│       ├── paper_service.py        # Paper service package
│       └── query_service.py       # Q&A service package
├── frontend/                      # React front end
│   ├── src/
│   │   ├── components/
│   │   │   ├── ui/                # shadcn/ui Basic components
│   │   │   ├── layout/             # layout component
│   │   │   ├── papers/             # Paper related components
│   │   │   └── query/              # Q&A related components
│   │   ├── pages/
│   │   ├── hooks/
│   │   ├── lib/
│   │   ├── stores/
│   │   ├── App.tsx
│   │   ├── main.tsx
│   │   └── index.css
│   ├── index.html
│   ├── package.json
│   ├── tailwind.config.js
│   ├── tsconfig.json
│   └── vite.config.ts
├── main.py                        # Keep CLI entry
└── (Other existing files remain unchanged)
```

---

## 4. API design

### Basic information
- **Base URL**: `http://localhost:8000`
- **CORS**: allow `http://localhost:5173`

### endpoint list

| method | path | Function | Request body | response |
|------|------|------|--------|------|
| `GET` | `/api/health` | health check | - | `{status: "ok"}` |
| `POST` | `/api/papers/upload` | upload PDF | `FormData: file` | `{pdf_name, title, authors, chunk_count, message}` |
| `GET` | `/api/papers` | Get list of papers | - | `{papers: [{pdf_name, title, authors, chunk_count, created_at, ...}]}` |
| `GET` | `/api/papers/{pdf_name}` | Get paper details | - | `{pdf_name, title, authors, metadata: {...}}` |
| `DELETE` | `/api/papers/{pdf_name}` | Delete paper | - | `{message}` |
| `GET` | `/api/papers/{pdf_name}/chunks` | Get the paper chunks | `?page=1&limit=20&type=text` | `{chunks: [], total, page, limit}` |
| `POST` | `/api/query` | Streaming Q&A | `{question: string}` | `SSE stream` |

### SSE event format

```
event: status
data: {"phase": "thinking", "step": 1, "text": "..."}

event: tool_call
data: {"tool": "search_papers", "kind": "paper_search", "args": {...}}

event: tool_result
data: {"kind": "paper_search", "count": 5, "pages": ["paper1:3", "paper2:10"]}

event: answer_started
data: {}

event: answer_token
data: {"text": "token"}

event: answer_done
data: {}
```

---

## 5. Front-end page design

### 1. routing structure
| path | page | illustrate |
|------|------|------|
| `/` | QueryPage | Q&A Home Page |
| `/papers` | PapersPage | Paper library |
| `/papers/:pdf_name` | PaperDetailPage | Paper details |

### 2. Page design

#### Query home page (`/`)
- Large search box, displayed in the center
- Streaming answer area, displayed in real time
- Quotation source display (click to jump)
- Search history

#### Paper library page (`/papers`)
- Thesis card grid display
- Upload button + Drag and drop upload area
- Delete confirmation dialog

#### Paper details page (`/papers/:pdf_name`)
- meta information card
- Chunk list (supports search and type filtering)
- Loading in pages

---

## 6. Implementation steps

### Phase 1: FastAPI service layer

#### step 1.1: Create API configuration
- create `api/config.py`
- Add to `API_HOST`, `API_PORT` Configuration

#### step 1.2: Definition Pydantic Schemas
- create `api/schemas.py`
- Define all requests/response model

#### step 1.3: Implement paper services
- create `api/services/paper_service.py`
- Implement upload/list/Details/delete/chunks Get

#### step 1.4: Implement question and answer service
- create `api/services/query_service.py`
- Implement streaming Q&A

#### step 1.5: Create route
- `api/routes/papers.py` - paper CRUD
- `api/routes/query.py` - Q&A

#### step 1.6: Create main application
- `api/main.py` - CORS、Routing, health check

---

### Phase 2: Front-end infrastructure

#### step 2.1: Initialize project
```bash
npm create vite@latest . -- --template react-ts
npm install
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

#### step 2.2: Install dependencies
```bash
npm install react-router-dom @tanstack/react-query zustand lucide-react clsx tailwind-merge class-variance-authority
npm install -D @types/node
```

#### step 2.3: set up shadcn/ui
```bash
npx shadcn@latest init
npx shadcn@latest add button input card dialog dropdown-menu toast
```

---

### Phase 3: Front-end component development

#### step 3.1: layout component
- Header, ThemeToggle, Layout

#### step 3.2: Thesis component
- PaperCard, PaperList, PaperUpload, PaperDetail

#### step 3.3: Q&A component
- QueryBox, QueryResult, SourceCitation

#### step 3.4: page
- QueryPage, PapersPage, PaperDetailPage

---

## 7. Startup method

### terminal 1: FastAPI
```bash
conda activate scholarrag
uvicorn api.main:app --reload --port 8000
```

### terminal 2: Frontend
```bash
cd frontend
npm run dev
```

---

## 8. Key things to note

### 1. CUDA/vLLM conflict
- When FastAPI starts **don't want** Import at module level `vector_store`
- Only import inside API routes (delayed import）

### 2. File upload
- PDF upload to temporary directory
- Clean up temporary files after processing is complete

### 3. SSE Streaming response
- use `StreamingResponse` + generator
- Correct settings `media_type="text/event-stream"`

### 4. Agent Module decoupling（W4-A, W4-B）
- `src/agent/retrieval_service.py`：definition `RetrievalService` agreement with `VectorStoreRetrievalService` Adapter, tool layer retrieves through service interface to avoid direct dependence `get_vector_store()`
- `src/agent/tooling.py`：Centralized registration `AGENT_TOOLS` and `TOOL_REGISTRY`
- `src/agent/types.py`：centralized definition `AgentState` Share type
- `src/agent/langgraph_agent.py`：Only responsible for graph compilation and node execution
- `src/agent/graph.py`：Responsible for streaming events and high-level API, no longer related to `langgraph_agent.py` Form a circular import

### 5. Unification of parsing paths (W2-B)
- Parse output directory by `PARSED_OUTPUT_DIR` Unified configuration
- PDF persistence directory is provided by `PDF_STORAGE_DIR` Unified configuration

### 6. Performance optimization architecture（W1-B, W3-A, W3-B, W3-D）
- **N+1 query elimination**: `similarity_search()` Change from item-by-item retrieve to batch retrieve
- **Configurable batch processing**: `EMBEDDING_BATCH_SIZE` Default 32 (original 4), GPU utilization improved 12x
- **Tokenizer cache**: `get_tokenizer()` Singleton mode to avoid repeated initialization
- **Query cache**: `QueryCache` 5minutes TTL, repeat query delay <10ms
- **Connection reuse**: Qdrant/LLM client Singleton, supports connection pool configuration

### 7. Reliability and monitoring（W1-A, W2-A, W4-C）
- **Database lease**: `ingestion_jobs` Table new `leased_at/leased_by`，Prevent repeated processing by multiple workers
- **Unify the exception hierarchy**: `AppError` → `ValidationError/NotFoundError/ExternalServiceError`
- **circuit breaker protection**: Tenacity Retry (3 times, exponential backoff）
- **Prometheus index**: `/metrics` Endpoints expose indicators such as retrieval latency, query count, etc.

### 8. Deployment architecture（W4-C）
- **Docker Containerization**: Image size 232MB, supports one-click deployment
- **Docker Compose**: start simultaneously API + Qdrant + Redis
- **health check**: `/api/health` endpoint
