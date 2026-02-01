import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import json
import base64

from dotenv import load_dotenv
load_dotenv()

import vertexai
from vertexai import rag
from vertexai.generative_models import GenerativeModel, Tool
from google.oauth2 import service_account

PROJECT_ID = os.getenv("PROJECT_ID", "serhrag")
LOCATION = os.getenv("LOCATION", "europe-west4")
PORT = int(os.getenv("PORT", 8000))

CREDENTIALS = {
    "type": "service_account",
    "project_id": "serhrag",
    "private_key_id": os.getenv("PRIVATE_KEY_ID", "0b2f568e9c6f"),
    "private_key": os.getenv("PRIVATE_KEY", "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQDG...\n-----END PRIVATE KEY-----"),
    "client_email": os.getenv("CLIENT_EMAIL", "serhrag@serhrag.iam.gserviceaccount.com"),
    "client_id": os.getenv("CLIENT_ID", "123456789"),
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
}

corpus = None
model = None


def setup():
    global corpus, model
    try:
        creds = service_account.Credentials.from_service_account_info(CREDENTIALS)
        vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=creds)
        
        corpora = list(rag.list_corpora())
        if corpora:
            corpus = corpora[0]
            config = rag.RagRetrievalConfig(top_k=3, filter=rag.Filter(vector_distance_threshold=0.5))
            tool = Tool.from_retrieval(
                retrieval=rag.Retrieval(
                    source=rag.VertexRagStore(
                        rag_resources=[rag.RagResource(rag_corpus=corpus.name)],
                        rag_retrieval_config=config,
                    ),
                )
            )
            model = GenerativeModel(
                model_name="gemini-2.0-flash-001",
                tools=[tool],
                system_instruction="responda perguntas sobre documentos. use rag para queries especificas. responda naturalmente para perguntas gerais."
            )
            print(f"ok corpus: {corpus.display_name}")
        return True
    except Exception as e:
        print(f"erro: {e}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup()
    yield
    print("encerrando")


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Msg(BaseModel):
    message: str


@app.get("/")
def root():
    return {"api": "rag", "version": "1.0.0", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok" if model and corpus else "error"}


@app.post("/chat")
def chat(req: Msg):
    if not model or not corpus:
        return {"error": "nao inicializado"}, 503
    if not req.message.strip():
        return {"error": "mensagem vazia"}, 400
    
    try:
        response = model.generate_content(req.message)
        return {"response": response.text}
    except Exception as e:
        return {"error": str(e)}, 500


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
