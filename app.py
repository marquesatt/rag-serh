#!/usr/bin/env python3
"""SERH RAG Chatbot com Vertex AI Agent Engine + LangGraph
Implementação oficial conforme: https://docs.cloud.google.com/agent-builder/agent-engine/develop/langgraph?hl=pt-br
"""

import os
import uuid
from typing import Optional
from dotenv import load_dotenv

# FastAPI setup
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Vertex AI
import vertexai
from vertexai import agent_engines, rag

load_dotenv()
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "marqu-443914")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")
PORT = int(os.getenv("PORT", 8000))

# Inicializa Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION)

# IDs do corpus SERH
CORPUS_ID = os.getenv("CORPUS_ID", "3527444408137940992")
CORPUS_DISPLAY_NAME = "serh-novo"

# ============================================================================
# FERRAMENTA: Busca em RAG SERH (conforme documentação oficial)
# ============================================================================

def search_serh_corpus(query: str) -> str:
    """Busca informações no corpus SERH usando Vertex AI RAG.
    
    Segue o padrão da documentação oficial. A ferramenta é uma simples função
    Python com docstring descrevendo parâmetros e retorno.
    
    Args:
        query: Pergunta ou termo para buscar no corpus de SERH.
            Exemplos: "Qual é o processo de admissão?", "Como solicitar férias?"
    
    Returns:
        str: Informações extraídas do corpus relevantes para a pergunta.
            Retorna mensagem se nenhum documento relevante for encontrado.
    """
    try:
        # Obtém o corpus pelo ID
        corpus = rag.get_corpus(
            name=f"projects/{PROJECT_ID}/locations/{LOCATION}/ragCorpora/{CORPUS_ID}"
        )
        
        if not corpus:
            return "Erro: Corpus SERH não encontrado. Verifique o ID."
        
        # Realiza retrieval query
        response = rag.retrieval_query(
            corpus_name=corpus.name,
            text=query,
            similarity_top_k=3,
        )
        
        # Formata resultados
        if response.responses:
            all_results = []
            for r in response.responses:
                if r.relevant_documents:
                    for doc in r.relevant_documents:
                        text = doc.chunk_data.text[:500]  # Limita tamanho
                        all_results.append(f"• {text}")
            
            if all_results:
                return "\n".join(all_results)
        
        return "Nenhum documento relevante encontrado para sua pergunta."
    
    except Exception as e:
        return f"Erro ao consultar corpus: {str(e)}"


# ============================================================================
# CRIAR AGENTE LANGGRAPH (conforme documentação oficial)
# ============================================================================

def create_serh_agent():
    """Cria agente SERH usando Vertex AI Agent Engine com LangGraph.
    
    Implementação exata conforme:
    https://docs.cloud.google.com/agent-builder/agent-engine/develop/langgraph?hl=pt-br
    
    Etapas:
    1. Define modelo: gemini-2.0-flash
    2. Define ferramenta: search_serh_corpus (função Python com docstring)
    3. Define instruções do sistema
    4. Cria LanggraphAgent com modelo, ferramenta e instruções
    """
    
    # Etapa 1: Configurar o modelo
    model = "gemini-2.0-flash"
    
    model_kwargs = {
        "temperature": 0.7,
        "max_output_tokens": 1024,
    }
    
    # Etapa 2: Instruções do sistema
    system_instruction = """Você é um assistente especializado em SERH.

Ajude usuários com dúvidas sobre:
- Processos e procedimentos de RH
- Auxílio e benefícios
- Férias, licenças e afastamentos  
- Frequência e controle de ponto
- Dados pessoais e documentação

INSTRUÇÕES:
1. Use a ferramenta 'search_serh_corpus' para consultar documentos quando apropriado
2. Seja sempre profissional, educado e transparente
3. Se não souber, recomende contato direto com setor de RH
4. Mantenha contexto da conversa anterior
5. Respostas devem ser claras e acessíveis

Tom: Profissional mas amigável, sempre pronto para ajudar."""
    
    # Etapa 3: Criar o agente
    # Padrão oficial: agent_engines.LanggraphAgent(model, tools, system_instruction)
    agent = agent_engines.LanggraphAgent(
        model=model,
        model_kwargs=model_kwargs,
        tools=[search_serh_corpus],  # Nossa ferramenta Python
        system_instruction=system_instruction,
    )
    
    return agent


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="SERH RAG Chatbot",
    description="Chat com Vertex AI Agent Engine + LangGraph",
    version="3.0-oficial"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Estado: armazena conversas em memória
# Chave: conversation_id
# Valor: lista de tuplas (role, content)
conversations: dict = {}

# Agente global - inicializado na startup
agent: Optional[object] = None

# ============================================================================
# MODELOS PYDANTIC
# ============================================================================

class Message(BaseModel):
    """Mensagem do usuário para o chat"""
    text: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    """Resposta do agente"""
    response: str
    conversation_id: str
    turn_count: int

# ============================================================================
# INICIALIZAÇÃO
# ============================================================================

@app.on_event("startup")
def startup():
    """Inicializa o agente na startup da aplicação"""
    global agent
    try:
        print(f"Iniciando agente...")
        print(f"  Project: {PROJECT_ID}")
        print(f"  Location: {LOCATION}")
        print(f"  Corpus ID: {CORPUS_ID}")
        
        agent = create_serh_agent()
        print("✓ Agente SERH LangGraph inicializado com sucesso")
        print(f"  Modelo: gemini-2.0-flash")
        print(f"  Ferramentas: search_serh_corpus")
        print(f"  Corpus: {CORPUS_DISPLAY_NAME} ({CORPUS_ID})")
    except Exception as e:
        print(f"✗ Erro ao inicializar agente: {e}")
        import traceback
        traceback.print_exc()
        print(f"\nDica: Verifique se as variáveis de ambiente estão configuradas:")
        print(f"  - GCP_PROJECT_ID={PROJECT_ID}")
        print(f"  - GCP_LOCATION={LOCATION}")
        print(f"  - CORPUS_ID={CORPUS_ID}")
        print(f"  - Google Cloud credentials configuradas")

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
def root():
    """Rota raiz - mostra status da API"""
    return {
        "name": "SERH RAG Chatbot",
        "version": "3.0-oficial",
        "status": "ok" if agent else "initializing",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "chat": "/chat",
            "conversation": "/conversation/{id}",
            "conversations": "/conversations"
        }
    }

@app.get("/health")
def health():
    """Health check do serviço"""
    return {
        "status": "ok" if agent else "initializing",
        "version": "3.0-oficial",
        "model": "gemini-2.0-flash",
        "framework": "Vertex AI Agent Engine + LangGraph",
        "corpus": CORPUS_DISPLAY_NAME,
    }

@app.post("/chat")
def chat(msg: Message) -> ChatResponse:
    """Chat com o agente SERH com suporte a multi-turn conversations.
    
    Segue padrão oficial de agent.query():
    - input: {"messages": [(role, content), ...]}
    - config: {"configurable": {"thread_id": conversation_id}}
    
    Params:
        msg.text: Mensagem do usuário
        msg.conversation_id: ID da conversa (gerado se não fornecido)
    
    Returns:
        ChatResponse com resposta, conversation_id e turn_count
    """
    
    if not agent:
        return JSONResponse(
            {"error": "Agente não inicializado. Aguarde startup..."},
            status_code=503
        )
    
    try:
        # Gera ou reutiliza conversation_id
        conversation_id = msg.conversation_id or str(uuid.uuid4())
        
        # Obtém ou cria histórico de conversa
        if conversation_id not in conversations:
            conversations[conversation_id] = []
        
        messages_list = conversations[conversation_id]
        
        # Adiciona mensagem do usuário
        messages_list.append(("user", msg.text))
        
        # Prepara input para o agente conforme documentação oficial
        # Format: lista de tuplas (role, content)
        agent_input = {
            "messages": messages_list
        }
        
        # Configura thread_id para persistência de conversa (Etapa 3 da doc)
        config = {
            "configurable": {
                "thread_id": conversation_id
            }
        }
        
        # Chama agente.query() - padrão oficial
        response = agent.query(
            input=agent_input,
            config=config
        )
        
        # Extrai resposta do dicionário retornado
        assistant_message = _extract_response(response)
        
        # Adiciona resposta ao histórico
        messages_list.append(("assistant", assistant_message))
        
        return ChatResponse(
            response=assistant_message,
            conversation_id=conversation_id,
            turn_count=len([m for m in messages_list if m[0] == "user"])
        )
    
    except Exception as e:
        print(f"✗ Erro no chat: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )

@app.get("/conversation/{conversation_id}")
def get_conversation(conversation_id: str):
    """Retorna o histórico completo de uma conversa"""
    
    if conversation_id not in conversations:
        return JSONResponse(
            {"error": "Conversa não encontrada"},
            status_code=404
        )
    
    messages = conversations[conversation_id]
    
    return {
        "conversation_id": conversation_id,
        "messages": [
            {"role": role, "content": content}
            for role, content in messages
        ],
        "message_count": len(messages),
        "user_turns": len([m for m in messages if m[0] == "user"]),
    }

@app.delete("/conversation/{conversation_id}")
def delete_conversation(conversation_id: str):
    """Deleta uma conversa do histórico"""
    
    if conversation_id not in conversations:
        return JSONResponse(
            {"error": "Conversa não encontrada"},
            status_code=404
        )
    
    del conversations[conversation_id]
    
    return {
        "status": "deleted",
        "conversation_id": conversation_id
    }

@app.get("/conversations")
def list_conversations():
    """Lista todas as conversas ativas"""
    
    return {
        "total_conversations": len(conversations),
        "conversations": [
            {
                "id": cid,
                "message_count": len(msgs),
                "user_turns": len([m for m in msgs if m[0] == "user"]),
            }
            for cid, msgs in conversations.items()
        ]
    }

# ============================================================================
# HELPERS
# ============================================================================

def _extract_response(response) -> str:
    """Extrai texto da resposta do agente.
    
    O agente retorna dicionário com possível estrutura:
    - {"output": "..."}
    - {"messages": [...]}
    - Outras estruturas
    """
    
    if isinstance(response, dict):
        if "output" in response:
            return response["output"]
        elif "messages" in response and response["messages"]:
            last_msg = response["messages"][-1]
            if isinstance(last_msg, dict) and "content" in last_msg:
                return last_msg["content"]
            elif hasattr(last_msg, "content"):
                return last_msg.content
    
    return str(response)

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)

