# Deploy no Railway

## 1. Preparar o projeto
```bash
# Instalar dependências localmente (opcional)
pip install -r requirements.txt

# Testar a API localmente
python api.py
# Acessar em http://localhost:8000/docs
```

## 2. Deploy no Railway

### Opção A: Usando Railway CLI
```bash
# Instalar Railway CLI
npm i -g @railway/cli

# Login
railway login

# Deploy
railway up
```

### Opção B: Conectar GitHub
1. Fazer push deste repositório para GitHub
2. Ir para https://railway.app
3. Criar novo projeto
4. Conectar GitHub repository
5. Railway detecta automaticamente o Procfile e faz deploy

## 3. Configurar variáveis de ambiente no Railway

Adicione no painel do Railway:

```
# Project ID do Google Cloud
PROJECT_ID=serhrag

# Caminho do arquivo Service Account
GOOGLE_APPLICATION_CREDENTIALS=./serhrag-0b2f568e9c6f.json

# (Opcional) Porta
PORT=8000
```

## 4. Upload do arquivo Service Account

Opção 1: Incluir no git (NÃO RECOMENDADO - segurança)
- Adicione `serhrag-*.json` ao `.gitignore`

Opção 2: Usar Railway Secrets (RECOMENDADO)
1. No painel Railway, adicionar secret com o conteúdo do JSON
2. Criar variável `GOOGLE_APPLICATION_CREDENTIALS_JSON`
3. No arquivo `auth.py`, modificar para ler a variável

Opção 3: Usar Google Cloud Service Account directamente
1. No painel Railway: Settings → Add Domain
2. Gerar nova chave no Google Cloud Console
3. Adicionar como variável de ambiente

## 5. Testar a API

Após deploy, você terá uma URL como:
```
https://seu-projeto.railway.app
```

Testes:
```bash
# Health check
curl https://seu-projeto.railway.app/health

# Chat
curl -X POST https://seu-projeto.railway.app/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Olá!"}'

# Docs (Swagger UI)
https://seu-projeto.railway.app/docs
```

## Endpoints disponíveis

- `GET /` - Info sobre a API
- `GET /health` - Health check
- `GET /docs` - Documentação interativa (Swagger)
- `POST /chat` - Enviar pergunta e receber resposta
- `GET /corpus` - Info do corpus carregado
- `GET /corpus/list` - Listar todos os corpus

## Exemplo de uso

```python
import requests

url = "https://seu-projeto.railway.app/chat"
response = requests.post(url, json={"message": "Qual é a data da reserva?"})
print(response.json()["response"])
```

## Troubleshooting

### Erro: "Corpus not loaded"
- Certifique-se que existe pelo menos um corpus no projeto Google Cloud
- Verifique as variáveis de ambiente no Railway

### Erro: "Permission denied"
- Verifique que o Service Account tem a role `roles/aiplatform.user`
- Confirme que as variáveis de ambiente estão corretas

### Erro: "Quota exceeded"
- Espere alguns minutos ou aumente a quota no Google Cloud Console
- Vá para: Cloud Console → APIs & Services → Quotas

## Monitoramento

No painel Railway você pode:
- Ver logs em tempo real
- Monitorar uso de recursos (CPU, memória)
- Gerenciar deployments anteriores
- Configurar alertas
