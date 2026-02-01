#!/usr/bin/env python3
"""Script para verificar o conteúdo do corpus"""
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

vertexai.init(project=PROJECT_ID, location=LOCATION)

print("=" * 70)
print("VERIFICANDO CONTEÚDO DO CORPUS")
print("=" * 70)

# Lista corpora
corpora = list(rag.list_corpora())
print(f"\n📦 Corpora encontrados: {len(corpora)}")
for corpus in corpora:
    print(f"\n  Corpus: {corpus.display_name}")
    print(f"  ID: {corpus.name}")
    print(f"  Display Name: {corpus.display_name}")
    
    # Lista arquivos no corpus
    print(f"\n  📄 Arquivos no corpus:")
    files = list(rag.list_files(corpus_name=corpus.name))
    print(f"     Total: {len(files)} arquivo(s)")
    
    for file in files:
        print(f"\n     - {file.display_name}")
        print(f"       ID: {file.name}")
        if hasattr(file, 'size_bytes'):
            print(f"       Tamanho: {file.size_bytes} bytes")

print("\n" + "=" * 70)
