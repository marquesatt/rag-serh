"""api rag fastapi - deploy railway"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import os

from auth import authenticate_vertex_ai
from config import PROJECT_ID, LOCATION, PORT

import vertexai
from vertexai import rag
from vertexai.generative_models import GenerativeModel, Tool
from dotenv import load_dotenv

load_dotenv()

# variaveis globais
current_corpus = None
chat_model = None


def init_rag():
    """inicializa rag"""
    global current_corpus, chat_model
    
    try:
        # autentica
        authenticate_vertex_ai(PROJECT_ID, LOCATION)
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        
        # carrega corpus
        corpus_list = list(rag.list_corpora())
        if corpus_list:
            current_corpus = corpus_list[0]
            
            # configura rag retrieval
            rag_retrieval_config = rag.RagRetrievalConfig(
                top_k=3,
                filter=rag.Filter(vector_distance_threshold=0.5),
            )
            
            rag_retrieval_tool = Tool.from_retrieval(
                retrieval=rag.Retrieval(
                    source=rag.VertexRagStore(
                        rag_resources=[rag.RagResource(rag_corpus=current_corpus.name)],
                        rag_retrieval_config=rag_retrieval_config,
                    ),
                )
            )
            
            # inicializa modelo chat
            chat_model = GenerativeModel(
                model_name="gemini-2.0-flash-001",
                tools=[rag_retrieval_tool],
                system_instruction="""voce eh um assistente qa que responde perguntas sobre documentos.

regras:
1. pergunta especifica sobre documentos → use ferramenta rag
2. pergunta geral/casual → responda sem rag
3. saudacoes → responda naturalmente sem ferramenta
4. respostas curtas em portugues
5. se nao achar info no documento → diga 'nao tenho essa informacao'

exemplos:
- 'ola' → responda sem usar ferramenta
- 'qual eh a data?' → use ferramenta
- 'como voce funciona?' → responda sem ferramenta
- 'qual hotel?' → use ferramenta"""
            )
            return True
    except Exception as e:
        print(f"erro ao inicializar rag: {e}")
        return False


# cria app
@asynccontextmanager
async def lifespan(app: FastAPI):
    """eventos de lifespan"""
    print("iniciando api rag...")
    init_rag()
    if current_corpus:
        print(f"ok corpus: {current_corpus.display_name}")
        print(f"ok modelo chat pronto")
    else:
        print("aviso: nenhum corpus encontrado")
    print("ok api pronta!\n")
    yield
    print("encerrando api...")


app = FastAPI(
    title="RAG Engine API",
    description="api rest para qa com documentos usando rag",
    version="1.0.0",
    lifespan=lifespan
)

# cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# models
class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    corpus: str = None


class HealthResponse(BaseModel):
    status: str
    corpus: str = None


# routes
@app.get("/", tags=["info"])
def root():
    """info api"""
    return {
        "name": "RAG Engine API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "chat": "/chat"
    }


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health():
    """health check"""
    if current_corpus is None:
        raise HTTPException(status_code=503, detail="corpus nao carregado")
    
    return HealthResponse(
        status="ok",
        corpus=current_corpus.display_name
    )


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
def chat(request: ChatRequest):
    """processa pergunta com rag"""
    
    if not current_corpus or not chat_model:
        raise HTTPException(
            status_code=503,
            detail="corpus nao inicializado"
        )
    
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="mensagem vazia")
    
    try:
        response = chat_model.generate_content(request.message)
        return ChatResponse(
            response=response.text,
            corpus=current_corpus.display_name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/corpus", tags=["corpus"])
def get_corpus():
    """info do corpus carregado"""
    if not current_corpus:
        raise HTTPException(status_code=404, detail="corpus nao carregado")
    
    return {
        "name": current_corpus.display_name,
        "id": current_corpus.name
    }


@app.get("/corpus/list", tags=["corpus"])
def list_corpus_endpoint():
    """lista corpus disponiveis"""
    try:
        corpus_list = list(rag.list_corpora())
        return {
            "total": len(corpus_list),
            "corpus": [
                {"name": c.display_name, "id": c.name}
                for c in corpus_list
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    print(f"\napi rodando em http://0.0.0.0:{PORT}")
    print(f"docs em http://localhost:{PORT}/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
