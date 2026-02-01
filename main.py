"""RAG Engine API - Entry Point"""
from app.main import app

if __name__ == "__main__":
    import uvicorn
    import os
    
    port = int(os.getenv("PORT", 8000))
    print(f"\n🌐 API rodando em http://0.0.0.0:{port}")
    print(f"📚 Docs em http://localhost:{port}/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
