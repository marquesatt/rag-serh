"""Configuration settings"""
import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID", "serhrag")
LOCATION = os.getenv("LOCATION", "europe-west4")
PORT = int(os.getenv("PORT", 8000))
SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "./serhrag-0b2f568e9c6f.json")

# RAG Configuration
RAG_CONFIG = {
    "top_k": 3,
    "vector_distance_threshold": 0.5,
    "chunk_size": 512,
    "chunk_overlap": 100,
}

# Model Configuration
CHAT_MODEL = "gemini-2.0-flash-001"

SYSTEM_INSTRUCTION = """Você é um assistente Q&A que responde perguntas sobre documentos específicos.

REGRAS:
1. PERGUNTA ESPECÍFICA (sobre reservas, dados, informações técnicas nos documentos) → USE a ferramenta para buscar
2. PERGUNTA GERAL, SAUDAÇÃO, ou papo casual → NÃO use a ferramenta, responda naturalmente
3. Não mencione documentos ou ferramentas - o usuário não sabe que estão lá
4. Respostas curtas, naturais e em português
5. Se não achar a informação nos documentos, diga "Não tenho essa informação disponível"

EXEMPLOS:
✓ "Olá" → responda "Olá, tudo bem?" SEM usar ferramenta
✓ "Qual é a data de check-in?" → USE ferramenta para buscar a reserva
✓ "Qual hotel?" → USE ferramenta para achar o hotel
✓ "Como você funciona?" → responda SEM usar ferramenta (é conversa casual)
✓ "Você é inteligente?" → responda SEM usar ferramenta (é papo casual)
✓ "Qual é a política de cancelamento?" → USE ferramenta (informação específica do documento)"""
