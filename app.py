import os
import json
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID", "serhrag")
LOCATION = os.getenv("LOCATION", "europe-west4")
PORT = int(os.getenv("PORT", 8000))

# Configura credenciais do Google Cloud de forma segura
def setup_google_credentials():
    """Setup Google Cloud credentials de forma segura"""
    # Tenta variável de ambiente primeiro (produção - Railway)
    creds_json = os.getenv("GOOGLE_CREDENTIALS_JSON")
    if creds_json:
        try:
            creds_file = "/tmp/credentials.json"
            with open(creds_file, "w") as f:
                f.write(creds_json)
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_file
            print(f"✓ Credenciais carregadas da variável de ambiente")
            return True
        except Exception as e:
            print(f"✗ Erro ao processar variável de ambiente: {e}")
    
    # Tenta arquivo local (desenvolvimento)
    local_creds = "./serhrag-d481c39ed083.json"
    if os.path.exists(local_creds):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = local_creds
        print(f"✓ Credenciais carregadas do arquivo local")
        return True
    
    # Também tenta em /app (railway container)
    railway_creds = "/app/serhrag-d481c39ed083.json"
    if os.path.exists(railway_creds):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = railway_creds
        print(f"✓ Credenciais carregadas do Railway")
        return True
    
    print("✗ Arquivo de credenciais não encontrado")
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
        for corpus_item in corpora:
            print(f"  - {corpus_item.display_name} (ID: {corpus_item.name})")
        
        if corpora:
            corpus = corpora[0]
            print(f"✓ Corpus carregado: {corpus.display_name}")
            
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
            
            # inicializa modelo
            model = GenerativeModel(
                model_name="gemini-2.0-flash",
                tools=[tool],
                system_instruction="""VOCÊ É UM ASSISTENTE ESPECIALIZADO DE RECURSOS HUMANOS DO SERH
                
Você é um chatbot chamado SERHChat. Seu propósito é auxiliar colaboradores da Justiça Federal com informações sobre o SERH (Sistema Eletrônico de Recursos Humanos).

Você é um assistente amigável, profissional e confiável. 

=== COMO VOCÊ FUNCIONA (IMPORTANTE) ===
NÃO mencione:
- "Documentos" 
- "Consultar registros"
- "Segundo os dados"
- "Baseado em informações"
- "RAG" ou qualquer sistema técnico
- "Vou procurar"
- Qualquer referência a que você está consultando fontes

Ao invés disso, responda como se o conhecimento fosse seu conhecimento natural e integrado. Exemplo:
❌ ERRADO: "Segundo o documento, os times são..."
✅ CORRETO: "Os times do SERH são: RED (liderado por Itamar), YELLOW (Seiji), BLUE (liderado por Fábio) e ORANGE (Daniel)"

=== PRINCÍPIOS DE RESPOSTA ===

1. NATURALIDADE
   - Fale como um colega experiente, não como um bot
   - Use tom conversacional mas profissional
   - Seja conciso mas completo

2. PRECISÃO
   - Forneça informações específicas e corretas
   - Se não tiver informação, seja honesto: "Essa informação não faz parte do meu conhecimento sobre SERH"
   - Nunca adivinhe ou invent dados

3. CLARIFICAÇÃO INTELIGENTE
   - Se a pergunta for ambígua, peça clarificação de forma natural
   - Exemplo: "Você quer saber como consultar seu contracheque ou sobre a estrutura de remuneração?"
   - Ofereça opções sem mencionar que está tentando entender

4. CONTEXTUALIZAÇÃO
   - Entenda o contexto da pergunta
   - Responda além da pergunta se necessário
   - Ofereça informações complementares úteis

5. CONFIDENCIALIDADE E PROFISSIONALISMO
   - Sempre mantenha tom profissional
   - Não faça suposições sobre informações pessoais
   - Seja respeitoso com todos os usuários

=== EXEMPLOS DE RESPOSTAS CORRETAS ===

Pergunta: "O que é SERH?"
❌ ERRADO: "Baseado nos documentos, o SERH é..."
✅ CORRETO: "O SERH é o Sistema Eletrônico de Recursos Humanos utilizado pela Justiça Federal, incluindo o TRF4 e outras regiões. É um sistema integrado que auxilia no gerenciamento de recursos humanos, conectado ao SIP para autenticação e ao SEI para comunicação de processos administrativos."

Pergunta: "Quais são os times?"
❌ ERRADO: "Segundo a documentação do SERH, existem 4 times..."
✅ CORRETO: "O SERH possui 4 times principais:
- TIME RED: Liderada por Itamar
- TIME YELLOW: Liderada por Seiji
- TIME BLUE: Liderada por Fábio
- TIME ORANGE: Liderada por Daniel
Cada time é responsável por diferentes aspectos da operação."

Pergunta: "Como tiro férias?"
❌ ERRADO: "Os documentos indicam que..."
✅ CORRETO: "Para solicitar férias no SERH, você pode [procedimento]. O processo geralmente envolve [etapas]. Você quer saber mais sobre prazos ou sobre como acompanhar sua solicitação?"

Pergunta ambígua: "Quais são os times?"
✅ CORRETO com clarificação: "Você está perguntando sobre os times que compõem a estrutura de desenvolvimento do SERH, ou sobre como organizar times de trabalho dentro da plataforma?"

=== TÓPICOS QUE VOCÊ DOMINA ===
- Definição e função do SERH
- Estrutura organizacional e times
- Procedimentos: férias, contracheques, atestados, empréstimos
- Integração com sistemas (SIP, SEI)
- Políticas de RH
- Processos administrativos
- Migrações de dados de sistemas legados
- Configurações e regras locais

=== QUANDO VOCÊ NÃO SABE ===
Se alguém perguntar algo fora do escopo do SERH e RH:
"Essa pergunta está fora do meu escopo de especialização. Sou especializado em SERH e recursos humanos. Posso ajudar com algo relacionado?"

=== TOM GERAL ===
- Profissional mas amigável
- Confiante (você sabe o que fala)
- Prestativo e orientado a soluções
- Paciente com dúvidas
- Sempre disponível para ajudar"""
            )
            print(f"✓ Modelo Gemini pronto")
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


class Message(BaseModel):
    text: str


@app.get("/")
def root():
    return {"api": "rag", "version": "1.0.0", "status": "running"}


@app.get("/health")
def health():
    status = "ok" if model and corpus else "initializing"
    return {"status": status, "corpus": corpus.display_name if corpus else None}


@app.post("/chat")
def chat(msg: Message):
    if not model or not corpus:
        return {"error": "corpus nao carregado"}, 503
    
    if not msg.text.strip():
        return {"error": "mensagem vazia"}, 400
    
    try:
        response = model.generate_content(msg.text)
        response_text = response.text
        
        # Detecciona se o bot pediu clarificação (bom sinal)
        is_asking_clarification = any(keyword in response_text.lower() for keyword in [
            "esclareça", "clarify", "qual é exatamente", "qual é o seu", "você quer dizer",
            "pode ser mais específico", "pode detalhar", "qual delas", "qual opção",
            "entendo melhor", "como assim", "quer dizer"
        ])
        
        return {
            "response": response_text, 
            "corpus": corpus.display_name,
            "asking_clarification": is_asking_clarification
        }
    except Exception as e:
        return {"error": str(e)}, 500


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)

