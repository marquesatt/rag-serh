# Arquivo de entrada para Railway
# Railway procura por main.py ou wsgi.py por padrão
from app import app

if __name__ == "__main__":
    import uvicorn
    import os
    
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
