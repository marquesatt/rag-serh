rag engine api

uma api rest para processamento de perguntas e respostas que utiliza rag (retrieval augmented generation) com google vertex ai. integra gemini 2.0 flash para geração de respostas e embeddings com text-embedding-005 para recuperação semantica de documentos.

requisitos

- python 3.14+
- google cloud project com vertex ai habilitado
- service account json com credenciais de autenticacao
- 4gb ram minimo para inference
- conexao de rede para google cloud

instalacao

1. clonar repositorio
   git clone <repository-url>
   cd ragserh

2. criar virtual environment
   python -m venv venv
   source venv/bin/activate  # linux/mac
   venv\Scripts\activate     # windows

3. instalar dependencias
   pip install -r requirements.txt

4. configurar credenciais google cloud
   export GOOGLE_APPLICATION_CREDENTIALS="./serhrag-0b2f568e9c6f.json"

5. configurar variaveis de ambiente
   cp .env.example .env

6. executar api
   python main.py

arquitetura

app/
├── core/
│   ├── auth.py          - autenticacao service account google cloud
│   └── config.py        - carregamento de configuracoes e variaves
├── services/
│   └── rag_service.py   - logica principal rag: corpus load, inference
├── api/
│   ├── routes.py        - definicao de endpoints rest
│   └── schemas.py       - modelos pydantic para validacao
└── main.py              - instancia fastapi e lifespan management

endpoints

get /
  descricao: informacoes basicas da api
  response: {name, version, docs, health, chat}

get /health
  descricao: verifica status e corpus carregado
  response: {status: "ok", corpus: "corpus-serh"}
  http: 200 ok | 503 service unavailable

post /chat
  descricao: processa pergunta e retorna resposta com rag
  request: {message: string}
  response: {response: string, corpus: string}
  headers: content-type application/json
  http: 200 ok | 400 bad request | 500 internal error | 503 unavailable

get /corpus
  descricao: informacoes do corpus carregado
  response: {name: string, id: string}
  http: 200 ok | 404 not found

get /corpus/list
  descricao: lista todos corpus disponiveis no projeto
  response: {total: int, corpus: [{name, id}]}
  http: 200 ok | 500 error

get /docs
  descricao: documentacao interativa swagger ui

configuracao

arquivo .env:

project_id=serhrag
location=europe-west4
port=8000
google_application_credentials=./serhrag-0b2f568e9c6f.json

variaveis rag (app/core/config.py):

rag_config = {
    "top_k": 3,
    "vector_distance_threshold": 0.5,
    "chunk_size": 512,
    "chunk_overlap": 100,
}

modelos

chat_model: gemini-2.0-flash-001
embedding_model: text-embedding-005
dimensao: 768

comportamento rag

sistema instruction determina automaticamente quando usar rag:

- queries especificas sobre documentos → ativa rag retrieval
- perguntas gerais/casual → responde sem rag
- greetings → responde naturalmente sem ferramenta

desenvolvimento local

iniciar api:
  python main.py

testar endpoint:
  curl -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d '{"message":"sua pergunta aqui"}'

acessar swagger ui:
  http://localhost:8000/docs

producao (railway)

deployment automatico via procfile:
  web: python main.py

variaveis obrigatorias:
  project_id
  location
  google_application_credentials (secret)
  port (opcional, default 8000)

tratamento de erros

400 bad request: mensagem vazia ou malformada
403 forbidden: autenticacao falhou
404 not found: corpus nao encontrado
500 internal server error: erro durante inference
503 service unavailable: corpus nao inicializado

performance

benchmarks europa-oeste-4:

latencia media: 800-1500ms
throughput: ~10 qps
memoria: 2-3gb
cpu: 1-2 cores recomendado

seguranca

service account credenciais nao versionadas (.gitignore)
cors: allow all (configuravel em app/main.py)
validacao input via pydantic
rate limiting: nao implementado (usar gateway externo)

persistencia

nao ha persistencia de conversas ou logs
sessoes sao stateless
corpus carregado na inicializacao

dependencias

google-cloud-aiplatform>=1.55.0
google-auth>=2.30.0
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
pydantic>=2.8.0
python-dotenv>=1.0.0

referencias

google vertex ai: https://cloud.google.com/vertex-ai/docs
fastapi: https://fastapi.tiangolo.com
railway: https://docs.railway.app

licenca

proprietary

notas

lazy imports para contornar compatibilidade python 3.14 pathlib
deprecation warning do generative models sdk nao afeta funcionalidade
corpus deve estar pre-carregado no projeto google cloud
