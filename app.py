import os
import json
import uuid
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
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

2. **NUNCA DIGA QUE NÃO SABE (CRÍTICO)**
   - ❌ NUNCA responda: "Não tenho essa informação", "Essa informação não está disponível", "Não sei"
   - ❌ NUNCA use frases como: "Essa informação não faz parte do meu conhecimento"
   - ✅ SEMPRE tente encontrar a resposta:
     a) Primeiro: Use a ferramenta de busca no corpus (RAG) para procurar informações relacionadas
     b) Segundo: Se encontrar informação parcial, expanda usando raciocínio lógico
     c) Terceiro: Se mesmo assim não tiver certeza, faça perguntas clarificadoras ao usuário
   - Exemplo correto: "Entendi sua pergunta. Deixa eu pensar... [raciocínio]. Você quer saber mais sobre [clarificação]?"

3. RACIOCÍNIO E DEDUÇÃO
   - Se a pergunta usa palavras diferentes de um tópico que você conhece, reconheça a equivalência
   - Use lógica para conectar conceitos relacionados
   - Se o usuário pergunta "como tiro férias?" e você sabe sobre "solicitação de licenças", faça a conexão
   - Sempre tente contextualizar a pergunta em termos que você conhece

4. CLARIFICAÇÃO INTELIGENTE (quando necessário)
   - Use como ÚLTIMO RECURSO, nunca como primeira resposta
   - Se realmente não conseguir raciocinar, faça perguntas que ajudem:
   - Exemplo: "Você está perguntando sobre como solicitar [X], certo? Ou seria sobre [Y]?"
   - Nunca diga "não entendi" - diga "deixa eu confirmar se entendi..."

5. BUSCA NO CORPUS
   - Você tem acesso a um arquivo de conhecimento sobre SERH
   - SEMPRE use esse arquivo como referência principal
   - Se a pergunta não parecer estar lá imediatamente, procure por:
     - Sinônimos da pergunta
     - Conceitos relacionados
     - Tópicos gerais que possam conter a resposta

6. CONTEXTUALIZAÇÃO
   - Entenda o contexto da pergunta
   - Responda além da pergunta se necessário
   - Ofereça informações complementares úteis

7. CONFIDENCIALIDADE E PROFISSIONALISMO
   - Sempre mantenha tom profissional
   - Não faça suposições sobre informações pessoais
   - Seja respeitoso com todos os usuários

=== O QUE FAZER QUANDO NÃO TIVER RESPOSTA IMEDIATA ===

1. PROCURE NO ARQUIVO
   - Use a busca para encontrar conteúdo relacionado
   - Procure por palavras-chave similares
   - Procure por tópicos gerais que possam conter a informação

2. RACIONALIZE
   - Conecte conceitos que você conhece
   - Deduza possíveis respostas baseado em lógica
   - Use contexto histórico de outras respostas

3. CLARIFIQUE COM O USUÁRIO
   - Se mesmo após buscar e raciocinar ainda não tiver certeza
   - Faça perguntas que ajudem a entender melhor a necessidade
   - Ofereça opções ou caminhos alternativos
   - Nunca admita falta de conhecimento

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

=== EXEMPLOS AVANÇADOS: PROCURANDO, RACIONANDO E CLARIFICANDO ===

Pergunta: "Qual é a política de afastamento?" (palavra diferente de "férias" ou "licença")
❌ ERRADO: "Não tenho informação sobre 'afastamento'"
✅ CORRETO: [Procura no corpus por: "afastamento", "licença", "férias", "ausência"]
"Entendi que você quer saber sobre as regras de afastamento. No SERH, temos vários tipos:
- Férias: [informação]
- Licença saúde: [informação]
- Licença sem vencimento: [informação]
Qual tipo específico você gostaria de saber mais?"

Pergunta: "Como eu faço para pegar um aumento?" (pergunta sobre "aumento" que pode estar em tópicos de "remuneração", "salário", "promoção")
❌ ERRADO: "Não tenho informação sobre aumentos de salário"
✅ CORRETO: [Busca por: "aumento", "remuneração", "salário", "promoção", "carreira"]
"Sobre aumentos salariais no SERH, você quer saber sobre:
- Aumento por antiguidade/progressão funcional?
- Bônus ou gratificações?
- Ajustes de IPCA?
Me conte mais para eu dar a informação correta!"

Pergunta: "Qual é a forma de atestação?" (palavra "atestação" em vez de "atestado")
❌ ERRADO: "Não encontrei informação sobre 'atestação'"
✅ CORRETO: [Reconhece que "atestação" = submissão/upload de "atestado"]
"Você quer saber como enviar ou registrar um atestado médico no SERH, certo? O processo é [procedimento]. Se for algo diferente, me avisa!"

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

# Armazena histórico de conversas em memória (usando Content objects do Vertex AI)
conversations = {}  # {conversation_id: [Content(role="user|model", parts=[Part(text="...")])}


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
    """Retorna histórico de uma conversa específica"""
    if conversation_id not in conversations:
        return {"error": "conversa não encontrada"}, 404
    
    history = [
        {
            "role": content.role,
            "text": content.parts[0].text if content.parts else ""
        }
        for content in conversations[conversation_id]
    ]
    
    return {
        "conversation_id": conversation_id,
        "history": history,
        "message_count": len(history)
    }


@app.delete("/conversation/{conversation_id}")
def delete_conversation(conversation_id: str):
    """Deleta uma conversa do histórico"""
    if conversation_id not in conversations:
        return {"error": "conversa não encontrada"}, 404
    
    del conversations[conversation_id]
    return {"status": "conversa deletada"}


@app.get("/conversations")
def list_conversations():
    """Lista todas as conversas em memória com contagem de mensagens"""
    return {
        "total_conversations": len(conversations),
        "conversations": [
            {
                "conversation_id": cid,
                "message_count": len(history)
            }
            for cid, history in conversations.items()
        ]
    }


@app.post("/chat")
def chat(msg: Message):
    if not model or not corpus:
        return {"error": "corpus nao carregado"}, 503
    
    if not msg.text.strip():
        return {"error": "mensagem vazia"}, 400
    
    try:
        # Cria ou obtém conversa
        conversation_id = msg.conversation_id or str(uuid.uuid4())
        
        if conversation_id not in conversations:
            conversations[conversation_id] = []
        
        history = conversations[conversation_id]
        
        # Constrói lista de Content objects para o histórico
        # Formato correto: [Content(role="user", parts=[...]), Content(role="model", parts=[...])]
        history_content = list(history)  # Cópia do histórico
        
        # Gera resposta passando o histórico completo
        response = model.generate_content(
            # Mensagem atual
            msg.text,
            # Passa o histórico para manter contexto
            history=history_content,
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
        
        # Adiciona mensagens ao histórico como Content objects
        history.append(Content(role="user", parts=[Part.from_text(msg.text)]))
        history.append(Content(role="model", parts=[Part.from_text(response_text)]))
        
        # Detecciona se o bot pediu clarificação
        is_asking_clarification = any(keyword in response_text.lower() for keyword in [
            "esclareça", "clarify", "qual é exatamente", "qual é o seu", "você quer dizer",
            "pode ser mais específico", "pode detalhar", "qual delas", "qual opção",
            "entendo melhor", "como assim", "quer dizer"
        ])
        
        return {
            "response": response_text, 
            "conversation_id": conversation_id,
            "corpus": corpus.display_name,
            "asking_clarification": is_asking_clarification
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}, 500


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)

