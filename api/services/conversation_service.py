"""
Conversation history service layer.

Provides CRUD operations for conversations and messages to persist the user's conversation history。
"""

import json
import time
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from api.models import Conversation, Message
from api.schemas import (
    AgentStepSchema,
    ConversationDetail,
    ConversationListItem,
    MessageCreate,
    MessageResponse,
    SourceSchema,
)


async def create_conversation(
    session: AsyncSession,
    conversation_id: str,
    title: str = "New Chat",
) -> Conversation:
    """
    Create new conversation。

    Args:
        session: database session
        conversation_id: Conversation ID (front-end generated）
        title: Conversation title

    Returns:
        Conversation object created
    """
    now = int(time.time() * 1000)
    conversation = Conversation(
        id=conversation_id,
        title=title,
        created_at=now,
        updated_at=now,
    )
    session.add(conversation)
    await session.flush()
    return conversation


async def get_conversation(
    session: AsyncSession,
    conversation_id: str,
) -> Optional[Conversation]:
    """
    Get conversation details。

    Args:
        session: database session
        conversation_id: dialogue ID

    Returns:
        Dialog object, returned if it does not exist None
    """
    result = await session.execute(
        select(Conversation)
        .options(selectinload(Conversation.messages))
        .where(Conversation.id == conversation_id)
    )
    return result.scalar_one_or_none()


async def get_all_conversations(session: AsyncSession) -> list[ConversationListItem]:
    """
    Get a list of all conversations.

    Sorted in descending order of update time, including number of messages。

    Args:
        session: database session

    Returns:
        Conversation list
    """
    subquery = (
        select(
            Message.conversation_id,
            func.count(Message.id).label("message_count"),
        )
        .group_by(Message.conversation_id)
        .subquery()
    )

    result = await session.execute(
        select(
            Conversation.id,
            Conversation.title,
            Conversation.created_at,
            Conversation.updated_at,
            func.coalesce(subquery.c.message_count, 0).label("message_count"),
        )
        .outerjoin(subquery, Conversation.id == subquery.c.conversation_id)
        .order_by(Conversation.updated_at.desc())
    )

    conversations = []
    for row in result:
        conversations.append(
            ConversationListItem(
                id=row.id,
                title=row.title,
                created_at=row.created_at,
                updated_at=row.updated_at,
                message_count=row.message_count,
            )
        )
    return conversations


async def delete_conversation(
    session: AsyncSession,
    conversation_id: str,
) -> bool:
    """
    Delete a conversation and all its messages。

    Args:
        session: database session
        conversation_id: dialogue ID

    Returns:
        Deleted successfully or not
    """
    result = await session.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    conversation = result.scalar_one_or_none()
    if conversation is None:
        return False

    await session.delete(conversation)
    return True


async def add_message(
    session: AsyncSession,
    conversation_id: str,
    message: MessageCreate,
) -> Optional[Message]:
    """
    Add a message to the conversation.

    Also updates the updated_at timestamp of the conversation。

    Args:
        session: database session
        conversation_id: dialogue ID
        message: message data

    Returns:
        Created message object, returned if the conversation does not exist None
    """
    result = await session.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    conversation = result.scalar_one_or_none()
    if conversation is None:
        return None

    steps_json = (
        json.dumps([s.model_dump() for s in message.steps]) if message.steps else None
    )
    sources_json = (
        json.dumps([s.model_dump() for s in message.sources])
        if message.sources
        else None
    )

    db_message = Message(
        id=message.id,
        conversation_id=conversation_id,
        role=message.role,
        content=message.content,
        steps=steps_json,
        sources=sources_json,
        created_at=message.created_at,
    )
    session.add(db_message)

    conversation.updated_at = int(time.time() * 1000)

    await session.flush()
    return db_message


def message_to_response(message: Message) -> MessageResponse:
    """
    Convert database message object to response model。

    Args:
        message: Database message object

    Returns:
        Message response model
    """
    steps = None
    if message.steps:
        try:
            steps_data = json.loads(message.steps)
            steps = [AgentStepSchema(**s) for s in steps_data]
        except (json.JSONDecodeError, TypeError):
            pass

    sources = None
    if message.sources:
        try:
            sources_data = json.loads(message.sources)
            sources = [SourceSchema(**s) for s in sources_data]
        except (json.JSONDecodeError, TypeError):
            pass

    return MessageResponse(
        id=message.id,
        role=message.role,
        content=message.content,
        steps=steps,
        sources=sources,
        created_at=message.created_at,
    )


def conversation_to_detail(conversation: Conversation) -> ConversationDetail:
    """
    Convert database conversation object to detail response model。

    Args:
        conversation: database conversation object

    Returns:
        Conversation details response model
    """
    return ConversationDetail(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages=[message_to_response(m) for m in conversation.messages],
    )
