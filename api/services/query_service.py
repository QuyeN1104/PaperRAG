import json
from collections.abc import Generator

from langchain_core.messages import AIMessage, HumanMessage

from api.schemas import MessageHistory


def stream_query(
    question: str,
    history: list[MessageHistory] | None = None,
) -> Generator[str, None, None]:
    """
    Execute a query and stream back SSE events。

    Args:
        question: User issues
        history: List of historical messages for multi-turn conversation context

    Yields:
        SSE format event string
    """
    from src.agent.graph import stream_answer_events

    # Convert historical messages to LangChain message format
    langchain_history = []
    if history:
        for msg in history:
            if msg.role == "user":
                langchain_history.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                langchain_history.append(AIMessage(content=msg.content))

    for event in stream_answer_events(question, langchain_history):
        event_type = event.get("type", "")

        if event_type == "agent_status":
            yield f"event: status\ndata: {json.dumps(event)}\n\n"
        elif event_type == "tool_call":
            yield f"event: tool_call\ndata: {json.dumps(event)}\n\n"
        elif event_type == "tool_result":
            yield f"event: tool_result\ndata: {json.dumps(event)}\n\n"
        elif event_type == "agent_observation":
            yield f"event: agent_observation\ndata: {json.dumps(event)}\n\n"
        elif event_type == "agent_visual_context":
            yield f"event: agent_visual_context\ndata: {json.dumps(event)}\n\n"
        elif event_type == "answer_started":
            yield f"event: answer_started\ndata: {json.dumps(event)}\n\n"
        elif event_type == "answer_token":
            yield f"event: answer_token\ndata: {json.dumps(event)}\n\n"
        elif event_type == "answer_done":
            yield f"event: answer_done\ndata: {json.dumps(event)}\n\n"
