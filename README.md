rag api

api rest simples para perguntas e respostas com documentos usando google vertex ai e fastapi.

setup

1. python -m venv venv
2. source venv/bin/activate (linux/mac) ou venv\Scripts\activate (windows)
3. pip install -r requirements.txt
4. editar .env com suas credenciais google cloud
5. python app.py

endpoints

get / - info
get /health - status
post /chat - {message: string}

credenciais

adicione seu service account json no arquivo .env:
- PRIVATE_KEY_ID
- PRIVATE_KEY
- CLIENT_EMAIL
- CLIENT_ID
