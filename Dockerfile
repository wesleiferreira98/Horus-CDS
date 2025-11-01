# Horus-CDS - Dockerfile para ambiente de produção
FROM python:3.12-slim

# Definir diretório de trabalho
WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar arquivo de dependências
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código da aplicação
COPY . .

# Expor portas
EXPOSE 5000 5001

# Variáveis de ambiente
ENV PYTHONUNBUFFERED=1

# Comando padrão (pode ser sobrescrito no docker-compose)
CMD ["python", "root/API/app.py"]
