#!/usr/bin/env python3
"""SERH RAG Chatbot - Versão Simples (sem Vertex AI local)"""

import os
import uuid
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import requests

# Setup
load_dotenv()
PROJECT_ID = os.getenv("PROJECT_ID", "serhrag")
LOCATION = os.getenv("LOCATION", "europe-west4")
PORT = int(os.getenv("PORT", 8000))
VERTEXAI_API_KEY = os.getenv("VERTEXAI_API_KEY", "")

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

# Armazena conversas
conversations = {}

class Message(BaseModel):
    text: str
    conversation_id: str = None

class Response(BaseModel):
    response: str
    conversation_id: str
    history_length: int

@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "2.0-simple",
        "message": "App está funcionando! Use POST /chat para enviar mensagens"
    }

@app.post("/chat")
def chat(msg: Message):
    """Chat com histórico"""
    try:
        # Gera ID se não tiver
        conversation_id = msg.conversation_id or str(uuid.uuid4())
        
        # Pega ou cria conversa
        if conversation_id not in conversations:
            conversations[conversation_id] = []
        
        messages = conversations[conversation_id]
        
        # Adiciona mensagem do usuário
        messages.append({"role": "user", "content": msg.text})
        
        # Resposta simulada (demonstração)
        # Em produção, isso chamará a API Vertex AI
        assistant_message = f"Recebi sua mensagem: '{msg.text}'. Sistema em desenvolvimento."
        
        # Adiciona ao histórico
        messages.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        return Response(
            response=assistant_message,
            conversation_id=conversation_id,
            history_length=len(messages)
        )
    
    except Exception as e:
        import traceback
        print(f"Erro: {e}")
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
