#!/usr/bin/env python3
"""Script para criar novo corpus e fazer import de arquivos do Google Drive"""

import os
import sys
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

# Configurações
CORPUS_NAME = "serh-novo"
GOOGLE_DRIVE_FOLDER_ID = "1sdWc44QZD-3sQYdpVYJ0vY5MbmlfdAg8"
GOOGLE_DRIVE_FOLDER_URL = f"https://drive.google.com/drive/folders/{GOOGLE_DRIVE_FOLDER_ID}"

print(f"\n📚 Criando corpus '{CORPUS_NAME}'...\n")

try:
    # 1. Criar o novo corpus
    print(f"1️⃣ Criando corpus '{CORPUS_NAME}'...")
    new_corpus = rag.create_corpus(
        display_name=CORPUS_NAME,
    )
    print(f"✅ Corpus criado: {new_corpus.display_name}")
    print(f"   ID: {new_corpus.name}\n")
    
    # 2. Importar arquivos do Google Drive usando paths (URLs)
    print(f"2️⃣ Importando arquivos do Google Drive...")
    print(f"   URL: {GOOGLE_DRIVE_FOLDER_URL}")
    
    # Usar paths com URL do Google Drive
    import_response = rag.import_files(
        corpus_name=new_corpus.name,
        paths=[GOOGLE_DRIVE_FOLDER_URL],
    )
    
    print(f"\n✅ Import concluído!")
    print(f"   Arquivos importados: {import_response.imported_rag_files_count}")
    print(f"   Arquivos pulados: {import_response.skipped_rag_files_count}\n")
    
    # 3. Listar arquivos do corpus
    print(f"3️⃣ Listando arquivos do corpus...")
    imported_files = list(rag.list_files(corpus_name=new_corpus.name))
    
    print(f"\n✅ Corpus '{CORPUS_NAME}' criado com sucesso!")
    print(f"\nDetalhes do corpus:")
    print(f"  Nome: {new_corpus.display_name}")
    print(f"  ID: {new_corpus.name}")
    print(f"  Total de arquivos: {len(imported_files)}")
    
    if imported_files:
        print(f"\nArquivos no corpus:")
        for f in imported_files:
            print(f"  - {f.display_name}")
    
except Exception as e:
    print(f"\n❌ Erro: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
