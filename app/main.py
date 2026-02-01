"""FastAPI Application"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.services.rag_service import get_rag_service
from app.api.routes import router
from app.core.config import PORT


rag_service = get_rag_service()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events"""
    print("🚀 Iniciando API RAG...")
    
    if rag_service.initialize():
        corpus_info = rag_service.get_corpus_info()
        if corpus_info:
            print(f"✓ Corpus: {corpus_info['name']}")
        print(f"✓ Chat model pronto")
    else:
        print("⚠️  Falha ao inicializar RAG")
    
    print("✓ API pronta!\n")
    yield
    print("🛑 Encerrando API...")


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    
    app = FastAPI(
        title="RAG Engine API",
        description="API para Q&A com documentos usando Vertex AI RAG",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routes
    app.include_router(router)
    
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    print(f"\n🌐 API rodando em http://0.0.0.0:{PORT}")
    print(f"📚 Docs em http://localhost:{PORT}/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
