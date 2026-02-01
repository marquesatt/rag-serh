#!/usr/bin/env python3
"""Script para listar todos os corpus RAG no Vertex AI"""

import json
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
    # Salvar temporariamente em arquivo
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write(credentials_json)
        creds_file = f.name
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_file
else:
    # Procurar arquivo local
    local_creds = list(Path(".").glob("serhrag*.json"))
    if local_creds:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(local_creds[0])
        print(f"✓ Usando credenciais locais: {local_creds[0]}")

# Inicializar Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION)

print(f"\n📚 Listando corpus em {PROJECT_ID} ({LOCATION})...\n")

try:
    # Listar todos os corpus
    corpora = list(rag.list_corpora())
    
    if corpora:
        print(f"✅ Total de corpus encontrados: {len(corpora)}\n")
        for i, corpus_item in enumerate(corpora, 1):
            print(f"{i}. Nome: {corpus_item.display_name}")
            print(f"   ID: {corpus_item.name}")
            print()
    else:
        print("❌ Nenhum corpus encontrado!")
        
except Exception as e:
    print(f"❌ Erro ao listar corpus: {e}")
    import traceback
    traceback.print_exc()
    print(f"\nDebug info:")
    print(f"  PROJECT_ID: {PROJECT_ID}")
    print(f"  LOCATION: {LOCATION}")
    print(f"  Credenciais carregadas: {bool(os.getenv('GOOGLE_APPLICATION_CREDENTIALS'))}")
