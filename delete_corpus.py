#!/usr/bin/env python3
"""Script para deletar um corpus RAG no Vertex AI"""

import os
from pathlib import Path
from dotenv import load_dotenv
import vertexai
from vertexai import rag

# Carregar variáveis de ambiente
load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID", "serhrag")
LOCATION = os.getenv("LOCATION", "europe-west4")

# Carregar credenciais
credentials_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
if credentials_json:
    print("✓ Usando credenciais do ambiente")
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write(credentials_json)
        creds_file = f.name
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_file
else:
    local_creds = list(Path(".").glob("serhrag*.json"))
    if local_creds:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(local_creds[0])
        print(f"✓ Usando credenciais locais: {local_creds[0]}")

# Inicializar Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION)

print(f"\n🗑️  Deletando corpus em {PROJECT_ID} ({LOCATION})...\n")

try:
    # Listar corpus para confirmar qual vamos deletar
    corpora = list(rag.list_corpora())
    
    if not corpora:
        print("❌ Nenhum corpus encontrado!")
    else:
        print(f"Corpus encontrados:\n")
        for i, corpus_item in enumerate(corpora, 1):
            print(f"{i}. Nome: {corpus_item.display_name}")
            print(f"   ID: {corpus_item.name}\n")
        
        # Deletar o primeiro (ou único) corpus
        corpus_to_delete = corpora[0]
        print(f"⚠️  Deletando: {corpus_to_delete.display_name}")
        print(f"ID: {corpus_to_delete.name}\n")
        
        # Deletar
        rag.delete_corpus(name=corpus_to_delete.name)
        print("✅ Corpus deletado com sucesso!")
        
except Exception as e:
    print(f"❌ Erro ao deletar corpus: {e}")
    import traceback
    traceback.print_exc()
