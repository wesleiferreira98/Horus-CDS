# Guia de Implantação com Docker - Horus-CDS

## Visão Geral

Este guia fornece instruções detalhadas para implantar o Horus-CDS utilizando Docker e Docker Compose. A containerização oferece isolamento, portabilidade e facilita o deployment em diferentes ambientes.

## Pré-requisitos

### Software Necessário

- Docker Engine 20.10 ou superior
- Docker Compose 2.0 ou superior
- 4GB de RAM disponível (mínimo)
- 10GB de espaço em disco

### Instalação do Docker

**Linux (Ubuntu/Debian)**:

```bash
# Atualizar repositórios
sudo apt-get update

# Instalar dependências
sudo apt-get install ca-certificates curl gnupg lsb-release

# Adicionar chave GPG oficial do Docker
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Configurar repositório
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Instalar Docker Engine
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Verificar instalação
docker --version
docker compose version
```

**Linux (Fedora)**:

```bash
# Instalar Docker
sudo dnf install docker docker-compose

# Iniciar e habilitar serviço
sudo systemctl start docker
sudo systemctl enable docker

# Adicionar usuário ao grupo docker
sudo usermod -aG docker $USER
```

**Windows/macOS**:

Baixe e instale Docker Desktop em:
https://www.docker.com/products/docker-desktop

### Suporte a GPU (Opcional)

Para utilizar aceleração GPU dentro de containers Docker, é necessário instalar o NVIDIA Container Toolkit.

**Linux**:

```bash
# Adicionar repositório
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Instalar nvidia-container-toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Reiniciar Docker
sudo systemctl restart docker

# Testar GPU no container
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

### Arquivo Docker Compose com GPU

O repositório inclui um arquivo pré-configurado para uso com GPU:

```bash
# Usar configuração com GPU
docker-compose -f docker-compose-gpu.yml build
docker-compose -f docker-compose-gpu.yml up -d

# Verificar GPU no container
docker exec horus-cds-api-gpu nvidia-smi
```

O arquivo `docker-compose-gpu.yml` inclui:
- Configuração de recursos GPU
- Variáveis de ambiente NVIDIA
- Health checks
- Limites de memória
- Documentação de configurações avançadas

**Modificar docker-compose.yml manualmente**:

```yaml
services:
  horus-api:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
```

**Windows/macOS com Docker Desktop**:

Docker Desktop para Windows com WSL2 suporta GPU. Para macOS, suporte a GPU NVIDIA não está disponível.

Consulte o guia completo de instalação CUDA: [CUDA_INSTALLATION.md](CUDA_INSTALLATION.md)

## Estrutura dos Containers

O Horus-CDS utiliza dois containers principais:

1. **horus-cds-api**: Servidor backend Flask (porta 5000)
2. **horus-cds-web**: Dashboard web Flask (porta 5001)

## Configuração e Execução

### 1. Construir as Imagens

```bash
# Navegue até o diretório do projeto
cd Horus-CDS

# Construa as imagens Docker
docker-compose build
```

### 2. Iniciar os Containers

```bash
# Iniciar em modo daemon (background)
docker-compose up -d

# Ou iniciar em modo interativo (ver logs)
docker-compose up
```

### 3. Verificar Status

```bash
# Listar containers em execução
docker-compose ps

# Verificar saúde dos containers
docker ps -a
```

### 4. Acessar os Serviços

- **API**: http://localhost:5000
- **Dashboard Web**: http://localhost:5001

## Gerenciamento de Containers

### Visualizar Logs

```bash
# Logs da API
docker-compose logs horus-api

# Logs do Web
docker-compose logs horus-web

# Logs em tempo real (follow)
docker-compose logs -f

# Últimas 100 linhas
docker-compose logs --tail=100
```

### Reiniciar Containers

```bash
# Reiniciar todos os containers
docker-compose restart

# Reiniciar container específico
docker-compose restart horus-api
```

### Parar Containers

```bash
# Parar todos os containers
docker-compose stop

# Parar container específico
docker-compose stop horus-api
```

### Remover Containers

```bash
# Parar e remover containers
docker-compose down

# Remover containers, redes e volumes
docker-compose down -v

# Remover também imagens
docker-compose down --rmi all
```

## Acesso ao Shell do Container

```bash
# Acessar shell do container da API
docker exec -it horus-cds-api bash

# Executar comando específico
docker exec horus-cds-api python --version

# Acessar shell do container web
docker exec -it horus-cds-web bash
```

## Atualização de Código

Quando houver alterações no código:

```bash
# Reconstruir e reiniciar
docker-compose up -d --build

# Ou separadamente
docker-compose build
docker-compose up -d
```

## Volumes Persistentes

O docker-compose.yml está configurado com volumes para persistência de dados:

```yaml
volumes:
  - ./root/API:/app/root/API
  - ./root/web:/app/root/web
  - ./logs:/app/logs
```

Isso significa que alterações nos diretórios locais são refletidas nos containers.

## Variáveis de Ambiente

Para configurar variáveis de ambiente, crie um arquivo `.env`:

```bash
# Exemplo de arquivo .env
PYTHONUNBUFFERED=1
FLASK_ENV=production
API_PORT=5000
WEB_PORT=5001
```

Então modifique o docker-compose.yml:

```yaml
services:
  horus-api:
    env_file:
      - .env
```

## Troubleshooting

### Container não inicia

```bash
# Verificar logs de erro
docker-compose logs horus-api

# Verificar configuração
docker-compose config

# Remover e recriar
docker-compose down
docker-compose up -d --force-recreate
```

### Porta já em uso

```bash
# Verificar portas em uso
sudo netstat -tulpn | grep :5000
sudo netstat -tulpn | grep :5001

# Alterar portas no docker-compose.yml
ports:
  - "5002:5000"  # Mapeia porta local 5002 para container 5000
```

### Problema de permissões

```bash
# Adicionar usuário ao grupo docker
sudo usermod -aG docker $USER

# Relogar ou executar
newgrp docker
```

### Limpar cache do Docker

```bash
# Remover containers parados
docker container prune

# Remover imagens não utilizadas
docker image prune

# Limpeza completa (cuidado!)
docker system prune -a --volumes
```

## Monitoramento

### Uso de Recursos

```bash
# Estatísticas em tempo real
docker stats

# Uso específico de um container
docker stats horus-cds-api
```

### Health Check

Adicione health checks ao docker-compose.yml:

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:5000/status"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

## Deployment em Produção

### Considerações de Segurança

1. **Não expor portas desnecessárias**
2. **Usar secrets do Docker para informações sensíveis**
3. **Configurar firewall adequadamente**
4. **Manter imagens atualizadas**

### Usar Docker Swarm (Opcional)

Para ambientes de produção com alta disponibilidade:

```bash
# Inicializar swarm
docker swarm init

# Deploy do stack
docker stack deploy -c docker-compose.yml horus

# Listar serviços
docker service ls

# Escalar serviços
docker service scale horus_horus-api=3
```

## Backup e Restore

### Backup de Volumes

```bash
# Criar backup do volume de logs
docker run --rm -v horus-cds_logs:/data -v $(pwd):/backup ubuntu tar czf /backup/logs-backup.tar.gz /data
```

### Restore de Volumes

```bash
# Restaurar backup
docker run --rm -v horus-cds_logs:/data -v $(pwd):/backup ubuntu tar xzf /backup/logs-backup.tar.gz -C /
```

## Conclusão

Docker simplifica significativamente o deployment e gerenciamento do Horus-CDS. Para ambientes de produção, considere usar orquestradores como Kubernetes ou Docker Swarm para maior escalabilidade e resiliência.

## Recursos Adicionais

- Documentação oficial do Docker: https://docs.docker.com/
- Docker Compose: https://docs.docker.com/compose/
- Best Practices: https://docs.docker.com/develop/dev-best-practices/
