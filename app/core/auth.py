"""Authentication module for Vertex AI"""
import os
from google.auth import default
from google.auth.transport.requests import Request
from google.oauth2 import service_account


def authenticate_vertex_ai(project_id: str, location: str):
    """Authenticate with Google Cloud using service account"""
    service_account_file = os.getenv(
        "GOOGLE_APPLICATION_CREDENTIALS",
        "./serhrag-0b2f568e9c6f.json"
    )
    
    print("\n" + "="*66)
    print("AUTENTICAÇÃO VERTEX AI")
    print("="*66)
    
    print("\n[1/2] Procurando arquivo Service Account...")
    
    if not os.path.exists(service_account_file):
        raise FileNotFoundError(f"Service Account não encontrado: {service_account_file}")
    
    print(f"✓ Arquivo encontrado: {service_account_file}")
    
    # Set environment variable for authentication
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_file
    
    # Authenticate
    credentials, _ = default()
    print(f"✓ Autenticado via Service Account: {service_account_file}")
    
    print("="*66 + "\n")
    
    return credentials
