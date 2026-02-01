#!/usr/bin/env python3
"""Buscar conteudo no corpus"""
import os
import json
import tempfile
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID", "serhrag")
LOCATION = os.getenv("LOCATION", "europe-west4")

# Setup credentials
creds_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
if creds_json:
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write(creds_json)
        temp_creds_path = f.name
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_creds_path

import vertexai
from vertexai import rag
from vertexai.generative_models import GenerativeModel, Tool

vertexai.init(project=PROJECT_ID, location=LOCATION)

# Buscar corpus com arquivos
corpora = list(rag.list_corpora())
corpus_with_files = None

for corpus in corpora:
    files = list(rag.list_files(corpus_name=corpus.name))
    if files:
        corpus_with_files = corpus
        print(f"Corpus encontrado: {corpus.display_name}")
        print(f"ID: {corpus.name}")
        print(f"Arquivos: {[f.display_name for f in files]}")
        break

if not corpus_with_files:
    print("Nenhum corpus com arquivos encontrado")
    exit(1)

# Criar RAG tool
retrieval_tool = Tool.from_retrieval(
    retrieval=rag.Retrieval(
        source=rag.VertexRagStore(
            rag_resources=[rag.RagResource(rag_corpus=corpus_with_files.name)],
            rag_retrieval_config=rag.RagRetrievalConfig(
                top_k=10,
            ),
        ),
    )
)

# Testar buscas no corpus
model = GenerativeModel(
    model_name="gemini-2.0-flash",
    tools=[retrieval_tool],
)

# Busca 1: Importação
print("\n" + "="*70)
print("BUSCANDO: 'importacao de dados'")
print("="*70)
response = model.generate_content(
    "Quais sao os problemas na importacao de dados do SERH?",
)
print(response.text)

# Busca 2: Problemas
print("\n" + "="*70)
print("BUSCANDO: 'problemas inconsistentes'")
print("="*70)
response = model.generate_content(
    "Qual é a solução para dados errados ou inconsistentes?",
)
print(response.text)
