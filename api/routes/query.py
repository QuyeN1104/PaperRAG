# pyright: reportMissingImports=false

"""
Query routing.

Handle user query requests, support streaming responses and multi-turn conversation context。
"""

import logging
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from api.database import get_db_session
from api.schemas import MessageCreate, QueryRequest
from api.services import conversation_service, query_service
from src.utils.exceptions import (
    AppError,
    ExternalServiceError,
    ValidationError,
    app_error_to_dict,
)

router = APIRouter()
logger = logging.getLogger(__name__)


def _as_http_exception(error: AppError) -> HTTPException:
    return HTTPException(status_code=error.status_code, detail=app_error_to_dict(error))


@router.post("")
async def query(request: QueryRequest):
    """
    Execute the query and stream the results back.

    If conversation_id is provided, user messages will be saved to the database.
    And transfer historical messages to the Agent to support multiple rounds of dialogue。
    """
    if not request.question.strip():
        raise _as_http_exception(ValidationError("Question cannot be empty"))

    # If conversation_id is provided, save user message
    if request.conversation_id:
        async with get_db_session() as session:
            await conversation_service.add_message(
                session,
                request.conversation_id,
                MessageCreate(
                    id=f"msg_user_{uuid.uuid4().hex[:12]}",
                    role="user",
                    content=request.question,
                    created_at=int(uuid.uuid1().time // 1000),
                ),
            )

    try:
        return StreamingResponse(
            query_service.stream_query(
                question=request.question,
                history=request.history,
            ),
            media_type="text/event-stream",
        )
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        logger.exception("Query processing failed: %s", exc)
        raise _as_http_exception(
            ExternalServiceError("Query processing failed", log_message=str(exc))
        )
