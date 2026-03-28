# AGENTS.md — Agent Module

LangGraph Agent implementation: state machine definition, tool chain, multi-modal answer generation。

## Structure

```
src/agent/
├── graph.py              # LangGraph State machines and streaming event handling
├── tools.py              # Agent Tool definition（search_papers, search_visuals, get_page_context）
├── langgraph_agent.py    # Agent core implementation
├── evidence_builder.py   # Evidence assembly and visual context
├── multimodal_answerer.py # Multimodal answer generation
├── retrieval_service.py  # Retrieve service protocols and adapters
├── tooling.py            # Tool Registration Center
└── types.py              # Shared type definitions (AgentState, etc.）
```

## Key Patterns

### Graph State Machine
- `graph.py` Define complete LangGraph state flow
- Streaming events via `stream_answer_events()` generate
- Tool calling must call at least one tool to answer

### Tool Implementation
- All tools are defined in `tooling.py` of `AGENT_TOOLS` and `TOOL_REGISTRY`
- Tool The selection rules are in `_SYSTEM_PROMPT` in definition
- Supports multi-tool invocation and dynamic reassessment
- Search tool only uses `similarity_search`；Don't quote the old `client.search()`

### Evidence Assembly
- `evidence_builder.py`: Assemble retrieval evidence
- `multimodal_answerer.py`: Generate answers with visual context
-Visual evidence passed `_multimodal_input` transfer

### Retrieval Service Protocol (W4-A)
- Tool layer passes `RetrievalService` Protocol decoupling vector storage
- `VectorStoreRetrievalService` Provide adapter implementation
- Supports injecting Mock implementation during testing
```python
from src.agent.retrieval_service import RetrievalService, get_retrieval_service

# Tool functions obtain services through dependency injection
@tool
def search_papers(query: str, retrieval_service: RetrievalService = Depends(get_retrieval_service)):
    return retrieval_service.search_papers(query)
```

### Shared Types (W4-B)
- `types.py` centralized definition `AgentState` TypedDict
- Avoid circular import problems
- Clear state structure and support type checking

## Critical Constraints

1. **Message Types**: System prompt words `SystemMessage`，prohibit `HumanMessage`
2. **Tool Calls**: Must call at least one before answering tool
3. **Stream Mode**: Don't switch to `"values"`（unless the internal iterator）
4. **CUDA/vLLM**: Do not import in this module `get_vector_store()`（see root level AGENTS.md）

## API

```python
from src.agent.graph import stream_answer_events

for event in stream_answer_events(question):
    # event types: agent_status, tool_call, tool_result, 
    #              agent_observation, agent_visual_context, 
    #              answer_started, answer_token
    pass
```
