# Guia Rápido de Instalação - Horus-CDS

## Instalação Automática (Mais Fácil)

Execute o script de instalação interativo que guiará você pelo processo:

**Linux / macOS**:
```bash
git clone https://github.com/wesleiferreira98/Horus-CDS.git
cd Horus-CDS
./install.sh
```

**Windows**:
```cmd
git clone https://github.com/wesleiferreira98/Horus-CDS.git
cd Horus-CDS
install.bat
```

O instalador oferece três opções com verificação automática de dependências e GPU.

---

## Instalação Manual

Se preferir instalação manual, escolha uma das opções abaixo:

### Opção 1: Instalação Rápida com Docker (Recomendado)

**Para sistemas sem GPU**:
```bash
git clone https://github.com/wesleiferreira98/Horus-CDS.git
cd Horus-CDS
docker-compose build
docker-compose up -d
```

**Para sistemas com GPU NVIDIA**:
```bash
# Instalar NVIDIA Container Toolkit primeiro
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Executar com GPU
cd Horus-CDS
docker-compose -f docker-compose-gpu.yml build
docker-compose -f docker-compose-gpu.yml up -d
```

Acesse:
- API: http://localhost:5000
- Dashboard: http://localhost:5001

---

### Opção 2: Instalação com Ambiente Virtual Python

**Sem GPU** (apenas CPU):
```bash
git clone https://github.com/wesleiferreira98/Horus-CDS.git
cd Horus-CDS
python3.12 -m venv venv-Horus
source venv-Horus/bin/activate  # Linux/macOS
# venv-Horus\Scripts\activate   # Windows
pip install --upgrade pip
pip install -r requirements.txt
python test_gpu_setup.py  # Verificar instalação
```

**Com GPU** (aceleração CUDA):
```bash
# 1. Instalar CUDA e cuDNN primeiro
# Siga o guia completo: CUDA_INSTALLATION.md

# 2. Verificar driver NVIDIA
nvidia-smi

# 3. Instalar Python e dependências
git clone https://github.com/wesleiferreira98/Horus-CDS.git
cd Horus-CDS
python3.12 -m venv venv-Horus
source venv-Horus/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 4. Verificar GPU
python test_gpu_setup.py
```

---

## Verificação da Instalação

### Verificar se está funcionando

**Com Docker**:
```bash
# Verificar containers
docker-compose ps

# Testar API
curl http://localhost:5000/status
```

**Com venv**:
```bash
# Verificar GPU (se disponível)
python test_gpu_setup.py

# Verificar TensorFlow
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

---

## Executar Experimentos

### Experimento 1: Detecção em Tempo Real

**Com venv**:
```bash
source venv-Horus/bin/activate
cd root/API
python app.py
```

**Com Docker**:
```bash
# Já está rodando se usou docker-compose up -d
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [[[244], [570], [243], [226]]]}'
```

### Experimento 2: Interface de Treinamento

**Apenas com venv** (PyQt5 não funciona em Docker):
```bash
source venv-Horus/bin/activate
cd root/Linux
python main.py
```

### Experimento 3: Dashboard Web

**Com venv**:
```bash
source venv-Horus/bin/activate
cd root/web
python run_web.py
```

**Com Docker**:
```bash
# Já está rodando em http://localhost:5001
```

---

## Troubleshooting Rápido

### GPU não detectada
```bash
# Verificar driver
nvidia-smi

# Verificar CUDA
nvcc --version

# Executar diagnóstico completo
python test_gpu_setup.py
```

### Docker não inicia
```bash
# Verificar logs
docker-compose logs

# Reconstruir
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Erro de permissão (Scapy)
```bash
# Modo simulação é ativado automaticamente
# Para modo real:
sudo python root/API/app.py
```

### Dependências faltando
```bash
# Reinstalar
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

---

## Comandos Úteis

### Docker
```bash
# Iniciar
docker-compose up -d

# Parar
docker-compose down

# Logs em tempo real
docker-compose logs -f

# Acessar shell do container
docker exec -it horus-cds-api bash

# Reconstruir
docker-compose up -d --build
```

### Python venv
```bash
# Ativar
source venv-Horus/bin/activate

# Desativar
deactivate

# Listar pacotes
pip list

# Atualizar dependências
pip install --upgrade -r requirements.txt
```

### GPU
```bash
# Informações da GPU
nvidia-smi

# Monitorar uso
watch -n 1 nvidia-smi

# Teste rápido TensorFlow
python -c "import tensorflow as tf; print(len(tf.config.list_physical_devices('GPU')), 'GPU(s) disponível(is)')"
```

---

## Links Úteis

### Documentação Completa
- [README.md](README.md) - Documentação principal
- [CUDA_INSTALLATION.md](CUDA_INSTALLATION.md) - Instalação GPU
- [DOCKER.md](DOCKER.md) - Guia Docker detalhado

### Ferramentas
- [test_gpu_setup.py](test_gpu_setup.py) - Teste de configuração
- [docker-compose-gpu.yml](docker-compose-gpu.yml) - Config GPU

### Suporte
- Issues: https://github.com/wesleiferreira98/Horus-CDS/issues
- NVIDIA CUDA: https://docs.nvidia.com/cuda/
- TensorFlow GPU: https://www.tensorflow.org/install/gpu

---

## Próximos Passos

1. ✅ Escolher método de instalação
2. ✅ Instalar dependências
3. ✅ Verificar instalação com test_gpu_setup.py
4. ✅ Executar experimento 1 (API)
5. ✅ Executar experimento 2 (Treinamento)
6. ✅ Executar experimento 3 (Dashboard)
7. ✅ Consultar documentação completa para detalhes

**Tempo estimado de instalação**:
- Docker (CPU): 10-15 minutos
- Docker (GPU): 30-45 minutos
- venv (CPU): 15-20 minutos
- venv (GPU): 1-2 horas (incluindo CUDA)

**Boa sorte!**
