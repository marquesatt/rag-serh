"""API Routes"""
from fastapi import APIRouter, HTTPException
from app.services.rag_service import get_rag_service
from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    CorpusListResponse,
    APIInfo
)

router = APIRouter()

rag_service = get_rag_service()


@router.get("/", response_model=APIInfo, tags=["Info"])
def root():
    """API Info"""
    return APIInfo(
        name="RAG Engine API",
        version="1.0.0",
        docs="/docs",
        health="/health",
        chat="/chat"
    )


@router.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    """Health check"""
    if not rag_service.initialized or rag_service.corpus is None:
        raise HTTPException(status_code=503, detail="Corpus not loaded")
    
    return HealthResponse(
        status="ok",
        corpus=rag_service.corpus.display_name
    )


@router.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(request: ChatRequest):
    """Chat endpoint - Responde perguntas usando RAG"""
    
    if not rag_service.initialized or rag_service.chat_model is None:
        raise HTTPException(
            status_code=503,
            detail="Corpus não está carregado. Inicialize a API primeiro."
        )
    
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Mensagem vazia")
    
    try:
        response_text = rag_service.chat(request.message)
        return ChatResponse(
            response=response_text,
            corpus=rag_service.corpus.display_name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/corpus", tags=["Corpus"])
def get_corpus():
    """Info sobre o corpus carregado"""
    corpus_info = rag_service.get_corpus_info()
    if not corpus_info:
        raise HTTPException(status_code=404, detail="Nenhum corpus carregado")
    
    return corpus_info


@router.get("/corpus/list", response_model=CorpusListResponse, tags=["Corpus"])
def list_corpus():
    """Lista todos os corpus disponíveis"""
    try:
        corpus_list = rag_service.list_all_corpus()
        return CorpusListResponse(
            total=len(corpus_list),
            corpus=corpus_list
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
