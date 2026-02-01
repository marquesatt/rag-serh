# RAG SERH - Simples e Prático

## Setup Rápido

```bash
# 1. Ativar ambiente
.\venv\Scripts\activate

# 2. Executar script
python rag.py
```

## O que faz?

**Menu:**
- Criar novo corpus
- Fazer upload em corpus existente  
- Listar corpus

## Exemplo de Uso

```
python rag.py
→ Escolha [1] para criar corpus
→ Dê um nome: "meu-corpus"
→ Digite [s] para importar documento
→ Cole o caminho: gs://meu-bucket/documento.pdf
```

## Arquivos

- `rag.py` - Script principal
- `auth.py` - Autenticação
- `config.py` - Configurações
- `.env` - Seu PROJECT_ID
- `requirements.txt` - Dependências

## Documentação

- Python SDK: https://cloud.google.com/vertex-ai
- RAG Docs: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/rag-engine
