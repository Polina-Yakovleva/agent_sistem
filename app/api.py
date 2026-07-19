"""HTTP API мультиагентной системы (FastAPI).

Запуск:
    uvicorn app.api:app --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import settings
from app.llm import check_llm_reachable
from app.observability import flush_observability, init_observability
from app.service import AgentService, InterruptInfo, TurnResult


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    user_id: Optional[str] = None


class ResumeRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    decision: str = Field(..., min_length=1)
    user_id: Optional[str] = None


class InterruptBody(BaseModel):
    type: str = "booking"
    summary: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    session_id: str
    user_id: str
    status: str
    answer: Optional[str] = None
    interrupt: Optional[InterruptBody] = None


def _to_response(result: TurnResult) -> ChatResponse:
    interrupt: Optional[InterruptBody] = None
    if result.interrupt is not None:
        info: InterruptInfo = result.interrupt
        interrupt = InterruptBody(
            type=info.type,
            summary=info.summary,
            payload=info.payload,
        )
    return ChatResponse(
        session_id=result.session_id,
        user_id=result.user_id,
        status=result.status,
        answer=result.answer,
        interrupt=interrupt,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_observability()
    service = AgentService()
    service.start()
    app.state.agent_service = service
    try:
        yield
    finally:
        service.stop()
        flush_observability()


app = FastAPI(
    title="Agent Runtime API",
    version="1.0.0",
    lifespan=lifespan,
)


def _service() -> AgentService:
    service = getattr(app.state, "agent_service", None)
    if service is None or service.graph is None:
        raise HTTPException(status_code=503, detail="Agent service is not ready")
    return service


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]:
    try:
        check_llm_reachable()
    except ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ready", "llm_base_url": settings.llm_base_url}


@app.post("/v1/chat", response_model=ChatResponse)
def chat(body: ChatRequest) -> ChatResponse:
    service = _service()
    try:
        result = service.run_turn(
            query=body.message,
            session_id=body.session_id,
            user_id=body.user_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return _to_response(result)


@app.post("/v1/chat/resume", response_model=ChatResponse)
def chat_resume(body: ResumeRequest) -> ChatResponse:
    service = _service()
    try:
        result = service.resume_turn(
            session_id=body.session_id,
            decision=body.decision,
            user_id=body.user_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return _to_response(result)
