"""autenticacao google cloud vertex ai"""
import os
from google.auth import default


def authenticate_vertex_ai(project_id: str, location: str):
    """autentica com google cloud usando service account"""
    service_account_file = os.getenv(
        "GOOGLE_APPLICATION_CREDENTIALS",
        "./serhrag-0b2f568e9c6f.json"
    )
    
    print("\n" + "="*60)
    print("autenticacao vertex ai")
    print("="*60)
    print("\n[1/2] procurando arquivo service account...")
    
    if not os.path.exists(service_account_file):
        raise FileNotFoundError(f"service account nao encontrado: {service_account_file}")
    
    print(f"ok arquivo encontrado: {service_account_file}")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_file
    
    credentials, _ = default()
    print(f"ok autenticado via service account")
    print("="*60 + "\n")
    
    return credentials
