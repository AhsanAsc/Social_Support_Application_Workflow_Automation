from fastapi import APIRouter, Depends
from apps.api.deps import get_current_active_user
from pydantic import BaseModel
from typing import List

from services.orchestration.graphs import run_master_graph

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    application_id: int | None = None


class ChatResponse(BaseModel):
    reply: str


@router.post("/", response_model=ChatResponse)
async def chat_with_agent(
    payload: ChatRequest,
    user=Depends(get_current_active_user),
):
    """
    Conversational agent orchestrated by LangGraph.
    """
    reply = run_master_graph(
        application_id=payload.application_id,
        messages=payload.messages,
    )
    return ChatResponse(reply=reply)
