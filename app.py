import os
import json
import uuid
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from typing import Optional
from vertexai.generative_models import (
    SafetySetting, HarmCategory, HarmBlockThreshold,
    Content, Part
)

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID", "serhrag")
LOCATION = os.getenv("LOCATION", "europe-west4")
PORT = int(os.getenv("PORT", 8000))

# Configura credenciais do Google Cloud de forma segura
def setup_google_credentials():
    """Setup Google Cloud credentials usando variável de ambiente"""
    import tempfile
    
    # 1. Tenta usar variável de ambiente (produção e desenvolvimento)
    creds_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if creds_json:
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                f.write(creds_json)
                temp_creds_path = f.name
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_creds_path
            print(f"✓ Credenciais carregadas da variável de ambiente")
            return
        except Exception as e:
            print(f"✗ Erro ao processar GOOGLE_APPLICATION_CREDENTIALS_JSON: {e}")
    
    # 2. Tenta arquivo local como fallback (desenvolvimento)
    import glob
    json_files = glob.glob("./serhrag*.json")
    if json_files:
        local_creds = json_files[0]
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = local_creds
        print(f"✓ Credenciais carregadas do arquivo local: {local_creds}")
        return
    
    print("✗ Nenhuma credencial encontrada")
    
    # Tenta arquivo local (desenvolvimento)
    local_creds = "./serhrag-d481c39ed083.json"
    if os.path.exists(local_creds):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = local_creds
        print(f"✓ Credenciais carregadas do arquivo local")
        return True
    
    print("✗ Nenhuma credencial encontrada")
    print("   Para Railway: configure GOOGLE_CREDENTIALS_JSON nas variáveis de ambiente")
    print("   Para desenvolvimento local: adicione serhrag-d481c39ed083.json")
    return False

setup_google_credentials()

corpus = None
model = None


def init_vertex_ai():
    global corpus, model
    try:
        # imports locais para evitar conflitos
        import vertexai
        from vertexai import rag
        from vertexai.generative_models import GenerativeModel, Tool
        
        # autentica com google cloud
        print(f"Inicializando Vertex AI com PROJECT_ID={PROJECT_ID}, LOCATION={LOCATION}")
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        print(f"✓ Vertex AI inicializado")
        
        # carrega corpus
        print(f"Listando corpora disponíveis...")
        corpora = list(rag.list_corpora())
        print(f"Corpora encontrados: {len(corpora)}")
        
        # Encontra corpus com arquivos (não vazio)
        corpus = None
        for corpus_item in corpora:
            files = list(rag.list_files(corpus_name=corpus_item.name))
            print(f"  - {corpus_item.display_name}: {len(files)} arquivo(s) (ID: {corpus_item.name})")
            if files and not corpus:  # Pega primeiro corpus com arquivos
                corpus = corpus_item
                print(f"     ✓ Usando este corpus (tem {len(files)} arquivo)")
        
        if corpus:
            print(f"\n✓ Corpus carregado: {corpus.display_name}")
            
            # configura rag retrieval
            config = rag.RagRetrievalConfig(
                top_k=3,
                filter=rag.Filter(vector_distance_threshold=0.5),
            )
            
            tool = Tool.from_retrieval(
                retrieval=rag.Retrieval(
                    source=rag.VertexRagStore(
                        rag_resources=[rag.RagResource(rag_corpus=corpus.name)],
                        rag_retrieval_config=config,
                    ),
                )
            )
            
            # inicializa modelo com RAG tool
            model = GenerativeModel(
                model_name="gemini-2.0-flash",
                tools=[tool],
                system_instruction="""Você é um assistente especializado em SERH (Sistema Eletrônico de Recursos Humanos).

INSTRUÇÕES PRINCIPAIS:
1. BUSQUE SEMPRE NO CORPUS PRIMEIRO usando a ferramenta de busca
2. FOCO NA PERGUNTA ATUAL: Se o assunto muda, concentre-se no novo tópico, não misture contextos antigos
3. RACIOCINE COM CONFIANÇA: 
   - Se não encontrar resposta exata, use lógica SERH para inferir (exemplo: "Em SERH, você provavelmente...")
   - Compare com módulos similares se apropriado
   - Forneça orientações úteis mesmo com informações parciais
4. SEJA TRANSPARENTE:
   - Quando tiver certeza: responda com confiança
   - Quando tiver dúvida: sugira "Para detalhes específicos, recomendo contatar o suporte"
   - NUNCA diga "não está disponível" sem tentar raciocinar antes

EVITE:
- Responder "não está disponível" ou "não sei" sem tentar ajudar
- Misturar contextos quando há mudança clara de assunto
- Termos técnicos como "RAG", "corpus", "buscar"
- Ser muito literal ou robótico

Tom: Especialista em RH/SERH, confiante mas honesto, sempre focado em ajudar."""
            )

            print(f"✓ Modelo Gemini pronto com RAG tool")
            return True
        else:
            print("✗ NENHUM CORPUS ENCONTRADO NO GOOGLE CLOUD")
            print(f"   Verifique:")
            print(f"   - Se PROJECT_ID está correto: {PROJECT_ID}")
            print(f"   - Se LOCATION está correta: {LOCATION}")
            print(f"   - Se o corpus foi criado no Google Cloud RAG")
            return False
    except Exception as e:
        print(f"✗ Erro ao inicializar: {e}")
        import traceback
        traceback.print_exc()
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("iniciando api...")
    init_vertex_ai()
    yield
    print("encerrando api...")


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Armazena ChatSession por conversation_id
# ChatSession gerencia histórico automaticamente
chat_sessions = {}


class Message(BaseModel):
    text: str
    conversation_id: Optional[str] = None  # Se None, cria uma nova conversa


class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    corpus: str
    asking_clarification: bool


@app.get("/")
def root():
    return {"api": "rag", "version": "1.0.0", "status": "running"}


@app.get("/health")
def health():
    status = "ok" if model and corpus else "initializing"
    return {"status": status, "corpus": corpus.display_name if corpus else None}


@app.get("/conversation/{conversation_id}")
def get_conversation(conversation_id: str):
    """Retorna histórico de uma conversa"""
    if conversation_id not in chat_sessions:
        return {"error": "conversa não encontrada"}, 404
    
    session = chat_sessions[conversation_id]
    history = [
        {
            "role": content.role,
            "text": content.parts[0].text if content.parts else ""
        }
        for content in session.history
    ]
    
    return {
        "conversation_id": conversation_id,
        "history": history,
        "message_count": len(history)
    }


@app.delete("/conversation/{conversation_id}")
def delete_conversation(conversation_id: str):
    """Deleta uma conversa"""
    if conversation_id not in chat_sessions:
        return {"error": "conversa não encontrada"}, 404
    
    del chat_sessions[conversation_id]
    return {"status": "conversa deletada"}


@app.get("/conversations")
def list_conversations():
    """Lista todas as conversas ativas"""
    return {
        "total_conversations": len(chat_sessions),
        "conversations": [
            {
                "conversation_id": cid,
                "message_count": len(session.history)
            }
            for cid, session in chat_sessions.items()
        ]
    }


@app.post("/chat")
def chat(msg: Message):
    if not model or not corpus:
        return {"error": "corpus nao carregado"}, 503
    
    if not msg.text.strip():
        return {"error": "mensagem vazia"}, 400
    
    try:
        conversation_id = msg.conversation_id or str(uuid.uuid4())
        
        # Cria ou obtém ChatSession
        if conversation_id not in chat_sessions:
            chat_sessions[conversation_id] = model.start_chat()
        
        session = chat_sessions[conversation_id]
        
        # Envia mensagem (ChatSession gerencia história automaticamente)
        response = session.send_message(
            msg.text,
            generation_config={
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 1024,
            },
            safety_settings=[
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                ),
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                ),
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                ),
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                ),
            ]
        )
        
        response_text = response.text
        
        # Detecciona se pediu clarificação
        is_asking_clarification = any(keyword in response_text.lower() for keyword in [
            "esclareça", "clarify", "qual é exatamente", "qual é o seu", "você quer dizer",
            "pode ser mais específico", "pode detalhar", "qual delas", "qual opção",
            "entendo melhor", "como assim", "quer dizer"
        ])
        
        return {
            "response": response_text, 
            "conversation_id": conversation_id,
            "corpus": corpus.display_name,
            "asking_clarification": is_asking_clarification,
            "history_length": len(session.history)
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "conversation_id": msg.conversation_id}
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)

