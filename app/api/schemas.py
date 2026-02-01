"""Request/Response schemas"""
from pydantic import BaseModel
from typing import Optional, List


class ChatRequest(BaseModel):
    """Chat request schema"""
    message: str


class ChatResponse(BaseModel):
    """Chat response schema"""
    response: str
    corpus: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    corpus: Optional[str] = None


class CorpusInfo(BaseModel):
    """Corpus information"""
    name: str
    id: str


class CorpusListResponse(BaseModel):
    """List of all corpus"""
    total: int
    corpus: List[CorpusInfo]


class APIInfo(BaseModel):
    """API information"""
    name: str
    version: str
    docs: str
    health: str
    chat: str
