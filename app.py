#!/usr/bin/env python3
"""SERH RAG Chatbot com Langgraph - Limpo e Simples"""

import os
import json
import uuid
import tempfile
from typing import Annotated
from dotenv import load_dotenv

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

import vertexai
from vertexai import rag
from vertexai.generative_models import GenerativeModel, Tool, Part
from langchain_google_vertexai import ChatVertexAI

# Setup
load_dotenv()
PROJECT_ID = os.getenv("PROJECT_ID", "serhrag")
LOCATION = os.getenv("LOCATION", "europe-west4")
PORT = int(os.getenv("PORT", 8000))

# Credenciais
creds_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
if creds_json:
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write(creds_json)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = f.name

# Inicializa Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION)

# ============================================================================
# SETUP RAG
# ============================================================================
corpus = None
rag_tool = None

def init_rag():
    global corpus, rag_tool
    try:
        corpora = list(rag.list_corpora())
        for c in corpora:
            files = list(rag.list_files(corpus_name=c.name))
            if files:
                corpus = c
                print(f"✓ Corpus: {corpus.display_name}")
                
                # Config RAG
                config = rag.RagRetrievalConfig(top_k=3)
                rag_tool = Tool.from_retrieval(
                    retrieval=rag.Retrieval(
                        source=rag.VertexRagStore(
                            rag_resources=[rag.RagResource(rag_corpus=corpus.name)],
                            rag_retrieval_config=config,
                        ),
                    )
                )
                print("✓ RAG Tool pronto")
                return True
        print("✗ Nenhum corpus com arquivos encontrado")
        return False
    except Exception as e:
        print(f"✗ Erro ao inicializar RAG: {e}")
        return False

# ============================================================================
# LANGGRAPH STATE & AGENT
# ============================================================================

class AgentState(TypedDict):
    """Estado da conversa"""
    messages: list
    conversation_id: str

def create_agent_graph():
    """Cria o grafo do Langgraph"""
    
    # LLM Vertex AI
    llm = ChatVertexAI(
        model_name="gemini-2.0-flash",
        temperature=0.7,
        max_tokens=1024,
    )
    
    # LLM com RAG tool
    if rag_tool:
        llm_with_tools = llm.bind_tools([rag_tool])
    else:
        llm_with_tools = llm
    
    def chat_node(state: AgentState) -> AgentState:
        """Node que chama o modelo"""
        # Pega última mensagem (user)
        user_message = state["messages"][-1]["content"]
        
        # Converte histórico para formato de LangChain
        chat_history = [
            {"role": m["role"], "content": m["content"]}
            for m in state["messages"]
        ]
        
        # System prompt
        system_prompt = """Você é um assistente especializado em SERH (Sistema Eletrônico de Recursos Humanos).
        
Instruções:
1. Busque informações no corpus quando apropriado
2. Raciocine com inteligência - não precisa de corpus para tudo
3. Seja transparente: se não souber, recomende contato com suporte
4. Mantenha contexto da conversa
5. Tom: Profissional, amigável, direto"""
        
        # Chama modelo
        try:
            response = llm_with_tools.invoke(
                chat_history + [{"role": "system", "content": system_prompt}]
            )
            
            # Extrai resposta
            if hasattr(response, 'content'):
                assistant_message = response.content
            else:
                assistant_message = str(response)
            
            # Adiciona ao histórico
            state["messages"].append({
                "role": "assistant",
                "content": assistant_message
            })
        except Exception as e:
            print(f"Erro ao chamar modelo: {e}")
            state["messages"].append({
                "role": "assistant",
                "content": f"Erro ao processar: {str(e)}"
            })
        
        return state
    
    # Cria grafo
    graph = StateGraph(AgentState)
    graph.add_node("chat", chat_node)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)
    
    return graph.compile()

agent = None

def init_agent():
    global agent
    if init_rag():
        agent = create_agent_graph()
        print("✓ Agent Langgraph pronto")
        return True
    return False

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(title="SERH RAG", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Armazena conversas (id -> messages)
conversations = {}

# ============================================================================
# MODELS
# ============================================================================

class Message(BaseModel):
    text: str
    conversation_id: str = None

class Response(BaseModel):
    response: str
    conversation_id: str
    history_length: int

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.on_event("startup")
def startup():
    init_agent()

@app.get("/health")
def health():
    status = "ok" if agent else "initializing"
    return {"status": status, "corpus": corpus.display_name if corpus else None}

@app.post("/chat")
def chat(msg: Message):
    """Chat com histórico"""
    if not agent:
        return JSONResponse({"error": "Agent não inicializado"}, 503)
    
    try:
        # Gera ID se não tiver
        conversation_id = msg.conversation_id or str(uuid.uuid4())
        
        # Pega ou cria conversa
        if conversation_id not in conversations:
            conversations[conversation_id] = []
        
        messages = conversations[conversation_id]
        
        # Adiciona mensagem do usuário
        messages.append({"role": "user", "content": msg.text})
        
        # Chama agent
        state = {
            "messages": messages,
            "conversation_id": conversation_id
        }
        
        result = agent.invoke(state)
        
        # Atualiza histórico
        conversations[conversation_id] = result["messages"]
        
        return Response(
            response=result["messages"][-1]["content"],
            conversation_id=conversation_id,
            history_length=len(result["messages"])
        )
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, 500)

@app.get("/conversation/{conversation_id}")
def get_conversation(conversation_id: str):
    """Retorna histórico de conversa"""
    if conversation_id not in conversations:
        return JSONResponse({"error": "conversa não encontrada"}, 404)
    
    return {
        "conversation_id": conversation_id,
        "history": conversations[conversation_id],
        "message_count": len(conversations[conversation_id])
    }

@app.delete("/conversation/{conversation_id}")
def delete_conversation(conversation_id: str):
    """Deleta conversa"""
    if conversation_id not in conversations:
        return JSONResponse({"error": "conversa não encontrada"}, 404)
    
    del conversations[conversation_id]
    return {"status": "deletado"}

@app.get("/conversations")
def list_conversations():
    """Lista conversas"""
    return {
        "total": len(conversations),
        "conversations": [
            {"id": cid, "messages": len(msgs)}
            for cid, msgs in conversations.items()
        ]
    }

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)

