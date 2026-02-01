FROM python:3.11-slim

WORKDIR /app

# Copia projeto
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Define variável de ambiente para credenciais
ENV GOOGLE_APPLICATION_CREDENTIALS=/tmp/credentials.json
ENV PORT=8080

# Expõe porta
EXPOSE 8080

# Roda a aplicação
CMD ["python", "app.py"]
