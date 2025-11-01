# Guia de Instalação do CUDA e cuDNN

## Índice

1. [Visão Geral](#visão-geral)
2. [Requisitos de Hardware](#requisitos-de-hardware)
3. [Instalação no Linux](#instalação-no-linux)
   - [Ubuntu 22.04 / 24.04](#ubuntu-2204--2404)
   - [Fedora 38 / 39 / 40](#fedora-38--39--40)
4. [Instalação no Windows](#instalação-no-windows)
5. [Instalação no macOS](#instalação-no-macos)
6. [Verificação da Instalação](#verificação-da-instalação)
7. [Resolução de Problemas](#resolução-de-problemas)
8. [Compatibilidade de Versões](#compatibilidade-de-versões)
9. [NVIDIA Container Toolkit (Para Docker)](#nvidia-container-toolkit-para-docker)
10. [Recursos Adicionais](#recursos-adicionais)

## Visão Geral

Este guia fornece instruções detalhadas para instalação dos drivers NVIDIA CUDA Toolkit e cuDNN, necessários para execução do Horus-CDS com aceleração GPU. O sistema foi testado com CUDA 12.6 e cuDNN 9.5.1.

**Importante**: A instalação de CUDA e cuDNN é **opcional**. O Horus-CDS pode ser executado utilizando apenas CPU, porém o treinamento dos modelos será significativamente mais lento. Recomenda-se GPU para:

* Treinamento de novos modelos (redução de horas para minutos)
* Experimentação com diferentes hiperparâmetros
* Processamento de grandes volumes de dados
* Ambiente de desenvolvimento intensivo

Para uso em produção apenas para detecção (sem treinamento), CPU é suficiente.

## Requisitos de Hardware

### GPU Compatível

- NVIDIA GeForce RTX série 20xx ou superior
- NVIDIA GeForce GTX série 16xx ou superior
- NVIDIA Tesla, Quadro ou A-series
- Compute Capability 3.5 ou superior

Para verificar compatibilidade da sua GPU:
https://developer.nvidia.com/cuda-gpus

### Requisitos Mínimos de Sistema

- 4GB de RAM (8GB recomendado)
- 10GB de espaço em disco livre
- Conexão com internet para download dos pacotes

## Instalação no Linux

### Ubuntu 22.04 / 24.04

#### 1. Remover Instalações Antigas (Opcional)

```bash
# Remover drivers NVIDIA antigos
sudo apt-get purge nvidia-*
sudo apt-get autoremove
sudo apt-get autoclean

# Remover CUDA antigo
sudo rm -rf /usr/local/cuda*
```

#### 2. Instalar Dependências

```bash
# Atualizar repositórios
sudo apt-get update

# Instalar dependências
sudo apt-get install -y build-essential dkms
sudo apt-get install -y freeglut3 freeglut3-dev libxi-dev libxmu-dev
```

#### 3. Instalar Driver NVIDIA

**Método 1: Repositório Ubuntu (Recomendado para iniciantes)**

```bash
# Adicionar repositório de drivers
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update

# Instalar driver recomendado
sudo ubuntu-drivers autoinstall

# Ou instalar versão específica
sudo apt-get install nvidia-driver-550

# Reiniciar sistema
sudo reboot
```

**Método 2: Driver oficial NVIDIA**

```bash
# Baixar driver em: https://www.nvidia.com/Download/index.aspx
# Exemplo para driver 550.120
wget https://us.download.nvidia.com/XFree86/Linux-x86_64/550.120/NVIDIA-Linux-x86_64-550.120.run

# Tornar executável
chmod +x NVIDIA-Linux-x86_64-550.120.run

# Instalar (modo texto, sem X server)
sudo systemctl isolate multi-user.target
sudo ./NVIDIA-Linux-x86_64-550.120.run
sudo systemctl start graphical.target
```

#### 4. Verificar Instalação do Driver

```bash
# Verificar driver instalado
nvidia-smi

# Saída esperada:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 550.120    Driver Version: 550.120    CUDA Version: 12.6        |
# +-----------------------------------------------------------------------------+
```

#### 5. Instalar CUDA Toolkit 12.6

**Método 1: Instalador Network (Recomendado)**

```bash
# Ubuntu 22.04
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-6

# Ubuntu 24.04
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-6
```

**Método 2: Instalador Runfile**

```bash
# Baixar CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda_12.6.0_560.28.03_linux.run

# Executar instalador
sudo sh cuda_12.6.0_560.28.03_linux.run

# Durante instalação:
# - Aceitar EULA
# - NÃO instalar driver (se já instalado no passo 3)
# - Instalar CUDA Toolkit
# - Instalar samples (opcional)
```

#### 6. Configurar Variáveis de Ambiente

```bash
# Adicionar ao ~/.bashrc ou ~/.zshrc
echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

# Recarregar configuração
source ~/.bashrc

# Verificar instalação
nvcc --version
```

#### 7. Instalar cuDNN 9.5.1

```bash
# Baixar cuDNN do site NVIDIA (requer conta):
# https://developer.nvidia.com/cudnn-downloads

# Exemplo para Ubuntu 22.04
wget https://developer.download.nvidia.com/compute/cudnn/9.5.1/local_installers/cudnn-local-repo-ubuntu2204-9.5.1_1.0-1_amd64.deb

# Instalar repositório local
sudo dpkg -i cudnn-local-repo-ubuntu2204-9.5.1_1.0-1_amd64.deb

# Copiar chave GPG
sudo cp /var/cudnn-local-repo-ubuntu2204-9.5.1/cudnn-*-keyring.gpg /usr/share/keyrings/

# Instalar cuDNN
sudo apt-get update
sudo apt-get install -y libcudnn9-cuda-12 libcudnn9-dev-cuda-12
```

**Método Alternativo (Manual)**

```bash
# Baixar arquivo tar.gz do cuDNN
tar -xvf cudnn-linux-x86_64-9.5.1.17_cuda12-archive.tar.xz

# Copiar arquivos
sudo cp cudnn-linux-x86_64-9.5.1.17_cuda12-archive/include/cudnn*.h /usr/local/cuda-12.6/include
sudo cp -P cudnn-linux-x86_64-9.5.1.17_cuda12-archive/lib/libcudnn* /usr/local/cuda-12.6/lib64

# Configurar permissões
sudo chmod a+r /usr/local/cuda-12.6/include/cudnn*.h /usr/local/cuda-12.6/lib64/libcudnn*
```

### Fedora 38 / 39 / 40

#### 1. Instalar Repositórios RPM Fusion

```bash
# Ativar RPM Fusion
sudo dnf install -y https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm
sudo dnf install -y https://download1.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm
```

#### 2. Instalar Driver NVIDIA

```bash
# Instalar driver
sudo dnf install -y akmod-nvidia
sudo dnf install -y xorg-x11-drv-nvidia-cuda

# Aguardar compilação do módulo
sudo akmods --force
sudo dracut --force

# Reiniciar
sudo reboot
```

#### 3. Instalar CUDA Toolkit

```bash
# Adicionar repositório CUDA
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/fedora39/x86_64/cuda-fedora39.repo

# Instalar CUDA
sudo dnf clean all
sudo dnf install -y cuda-toolkit-12-6
```

#### 4. Configurar Variáveis de Ambiente

```bash
# Adicionar ao ~/.bashrc
echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

#### 5. Instalar cuDNN

```bash
# Baixar e instalar RPM do cuDNN
sudo dnf install cudnn9-cuda-12 cudnn9-devel-cuda-12
```

## Instalação no Windows

### Windows 10 / 11

#### 1. Verificar Requisitos

- Windows 10 versão 1909 ou superior
- Windows 11 (qualquer versão)
- Visual Studio 2019 ou 2022 (Community Edition suficiente)

#### 2. Instalar Visual Studio

```powershell
# Baixar Visual Studio Community
# https://visualstudio.microsoft.com/downloads/

# Durante instalação, selecionar:
# - Desktop development with C++
# - Windows 10 SDK
```

#### 3. Instalar Driver NVIDIA

- Baixar GeForce Experience ou driver standalone:
  https://www.nvidia.com/Download/index.aspx

- Executar instalador e seguir assistente
- Selecionar "Custom Installation" para opções avançadas
- Reiniciar sistema após instalação

#### 4. Verificar Driver

```powershell
# Abrir PowerShell e executar
nvidia-smi

# Verificar versão do driver e CUDA compatível
```

#### 5. Instalar CUDA Toolkit 12.6

- Baixar instalador:
  https://developer.nvidia.com/cuda-downloads

- Executar instalador (exe ou network installer)
- Selecionar instalação "Custom" ou "Express"
- Componentes recomendados:
  - CUDA Toolkit
  - Visual Studio Integration
  - CUDA Samples (opcional)
  - CUDA Documentation (opcional)

#### 6. Configurar Variáveis de Ambiente

O instalador geralmente configura automaticamente. Verificar:

```powershell
# Verificar Path
echo $env:PATH

# Deve conter:
# C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin
# C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\libnvvp

# Verificar CUDA_PATH
echo $env:CUDA_PATH
# Deve retornar: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
```

#### 7. Instalar cuDNN 9.5.1

- Criar conta NVIDIA Developer (gratuito):
  https://developer.nvidia.com/cudnn-downloads

- Baixar cuDNN para Windows (ZIP)

- Extrair arquivo ZIP

- Copiar arquivos para diretório CUDA:

```powershell
# Executar como Administrador

# Copiar DLLs
Copy-Item "cudnn-windows-x86_64-9.5.1.17_cuda12-archive\bin\*.dll" -Destination "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"

# Copiar headers
Copy-Item "cudnn-windows-x86_64-9.5.1.17_cuda12-archive\include\*.h" -Destination "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\include"

# Copiar bibliotecas
Copy-Item "cudnn-windows-x86_64-9.5.1.17_cuda12-archive\lib\x64\*.lib" -Destination "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\lib\x64"
```

#### 8. Verificar Instalação

```powershell
# Verificar nvcc
nvcc --version

# Verificar cuDNN (no Python)
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Instalação no macOS

**Nota Importante**: A partir do macOS 10.14 (Mojave), a Apple descontinuou suporte para GPUs NVIDIA. CUDA não está disponível para Macs com chips Apple Silicon (M1/M2/M3) ou Macs recentes com GPUs AMD.

Para Macs antigos com GPUs NVIDIA (anterior a 2014):

### macOS High Sierra (10.13) ou anterior

#### 1. Verificar GPU

```bash
system_profiler SPDisplaysDataType | grep "Chipset Model"
```

#### 2. Instalar Xcode Command Line Tools

```bash
xcode-select --install
```

#### 3. Baixar CUDA para macOS

- Última versão compatível: CUDA 10.2
- Download: https://developer.nvidia.com/cuda-10.2-download-archive

#### 4. Instalar CUDA Toolkit

```bash
# Executar instalador DMG
# Seguir assistente de instalação
```

#### 5. Configurar Variáveis

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bash_profile
echo 'export DYLD_LIBRARY_PATH=/usr/local/cuda/lib:$DYLD_LIBRARY_PATH' >> ~/.bash_profile
source ~/.bash_profile
```

**Recomendação**: Para Macs modernos, utilize versão CPU do TensorFlow ou considere plataformas com GPUs NVIDIA (Linux/Windows).

## Verificação da Instalação

### Teste Automatizado (Recomendado)

O Horus-CDS inclui um script de teste completo que verifica toda a configuração:

```bash
cd Horus-CDS
python test_gpu_setup.py
```

Este script irá verificar automaticamente:
- Versão do Python
- Instalação do TensorFlow
- Suporte CUDA
- Dispositivos GPU disponíveis
- Compute Capability das GPUs
- Memória GPU
- Performance (benchmark CPU vs GPU)
- Todas as bibliotecas necessárias

### Teste Manual

Alternativamente, você pode criar manualmente um arquivo `test_cuda.py`:

```python
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("CUDA disponível:", tf.test.is_built_with_cuda())
print("GPU disponível:", tf.test.is_gpu_available(cuda_only=True))
print("\nDispositivos GPU:")
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    print(f"  - {gpu}")
    details = tf.config.experimental.get_device_details(gpu)
    print(f"    Compute Capability: {details.get('compute_capability', 'N/A')}")
```

Execute:

```bash
python test_cuda.py
```

Saída esperada:

```
TensorFlow version: 2.20.0
CUDA disponível: True
GPU disponível: True

Dispositivos GPU:
  - PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')
    Compute Capability: (8, 9)
```

### Benchmark de Performance

```python
import tensorflow as tf
import time

# Criar matriz grande
matrix_size = 10000
a = tf.random.normal([matrix_size, matrix_size])
b = tf.random.normal([matrix_size, matrix_size])

# Teste CPU
with tf.device('/CPU:0'):
    start = time.time()
    c = tf.matmul(a, b)
    cpu_time = time.time() - start
    print(f"CPU time: {cpu_time:.4f}s")

# Teste GPU
with tf.device('/GPU:0'):
    start = time.time()
    c = tf.matmul(a, b)
    gpu_time = time.time() - start
    print(f"GPU time: {gpu_time:.4f}s")

print(f"Speedup: {cpu_time/gpu_time:.2f}x")
```

## Resolução de Problemas

### GPU não detectada

```bash
# Linux: Verificar driver
nvidia-smi

# Verificar versão CUDA
nvcc --version

# Verificar bibliotecas
ldconfig -p | grep cuda
ldconfig -p | grep cudnn

# Verificar TensorFlow
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Conflito de versões

```bash
# Verificar versão do driver
cat /proc/driver/nvidia/version

# Verificar compatibilidade CUDA/Driver
# CUDA 12.6 requer driver >= 525.60.13
```

### Erro de memória GPU

Adicione ao início do script Python:

```python
import tensorflow as tf

# Permitir crescimento gradual de memória
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Ou limitar memória
tf.config.set_logical_device_configuration(
    gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]
)
```

### Bibliotecas não encontradas (Linux)

```bash
# Atualizar cache do linker
sudo ldconfig

# Adicionar permanentemente ao /etc/ld.so.conf.d/
echo '/usr/local/cuda-12.6/lib64' | sudo tee /etc/ld.so.conf.d/cuda.conf
sudo ldconfig
```

### Erro de permissões (Linux)

```bash
# Adicionar usuário ao grupo video
sudo usermod -a -G video $USER

# Relogar para aplicar mudanças
```

## Compatibilidade de Versões

### TensorFlow 2.20.0

| Componente | Versão Testada | Versão Mínima |
|------------|----------------|---------------|
| CUDA       | 12.6           | 12.3          |
| cuDNN      | 9.5.1          | 9.0           |
| Driver     | 550.120        | 525.60.13     |
| Python     | 3.12.12        | 3.9           |

### Compute Capability

| Série GPU        | Compute Capability |
|------------------|--------------------|
| RTX 40xx         | 8.9                |
| RTX 30xx         | 8.6                |
| RTX 20xx / GTX 16xx | 7.5             |
| GTX 10xx         | 6.1                |
| Tesla V100       | 7.0                |
| Tesla T4         | 7.5                |

## Recursos Adicionais

### Documentação Oficial

- CUDA Toolkit: https://docs.nvidia.com/cuda/
- cuDNN: https://docs.nvidia.com/deeplearning/cudnn/
- TensorFlow GPU: https://www.tensorflow.org/install/gpu

### Downloads

- CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
- cuDNN: https://developer.nvidia.com/cudnn-downloads
- Drivers NVIDIA: https://www.nvidia.com/download/index.aspx

### Suporte

- Fóruns NVIDIA: https://forums.developer.nvidia.com/
- TensorFlow GitHub: https://github.com/tensorflow/tensorflow/issues
- Stack Overflow: https://stackoverflow.com/questions/tagged/cuda

## NVIDIA Container Toolkit (Para Docker)

Para utilizar GPU dentro de containers Docker, é necessário instalar o NVIDIA Container Toolkit. Este componente permite que containers Docker acessem as GPUs do host.

### Instalação no Linux

#### Ubuntu / Debian

```bash
# Configurar repositório
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Instalar pacote
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configurar Docker runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Reiniciar Docker
sudo systemctl restart docker
```

#### Fedora

```bash
# Configurar repositório
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
  sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo

# Instalar pacote
sudo dnf install -y nvidia-container-toolkit

# Configurar Docker runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Reiniciar Docker
sudo systemctl restart docker
```

### Verificar Instalação

```bash
# Testar acesso à GPU no container
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi

# Saída esperada: informações da GPU similar ao comando nvidia-smi no host
```

### Configurar docker-compose.yml

Para utilizar GPU com Docker Compose, modifique o arquivo:

```yaml
version: '3.8'

services:
  horus-api:
    build: .
    ports:
      - "5000:5000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
```

### Configurar Dockerfile

Adicione no Dockerfile para usar imagem base com CUDA:

```dockerfile
# Use imagem base com CUDA
FROM nvidia/cuda:12.6.0-cudnn9-runtime-ubuntu22.04

# Instalar Python
RUN apt-get update && apt-get install -y python3.12 python3-pip

# Resto da configuração...
```

### Limitar Uso de GPU

Para limitar quais GPUs o container pode acessar:

```bash
# Usar apenas GPU 0
docker run --rm --gpus '"device=0"' nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi

# Usar GPUs 0 e 1
docker run --rm --gpus '"device=0,1"' nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi

# Usar todas as GPUs
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

### Configurar Limites de Memória

No docker-compose.yml:

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
        limits:
          memory: 8G
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
```

### Windows com WSL2

Para Windows, use Docker Desktop com WSL2 backend:

1. Instalar WSL2:
```powershell
wsl --install
wsl --set-default-version 2
```

2. Instalar Docker Desktop para Windows

3. Habilitar integração WSL2 nas configurações do Docker Desktop

4. Instalar driver NVIDIA para WSL2:
   - Download: https://developer.nvidia.com/cuda/wsl

5. Verificar GPU no WSL2:
```bash
# No terminal WSL2
nvidia-smi
```

6. Testar container:
```bash
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

### Troubleshooting Container Toolkit

#### Erro: "could not select device driver"

```bash
# Verificar instalação do toolkit
nvidia-ctk --version

# Reconfigurar runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

#### Erro: "unknown runtime specified nvidia"

```bash
# Verificar configuração do Docker
cat /etc/docker/daemon.json

# Deve conter:
# {
#   "runtimes": {
#     "nvidia": {
#       "path": "nvidia-container-runtime",
#       "runtimeArgs": []
#     }
#   }
# }

# Reiniciar Docker
sudo systemctl restart docker
```

#### GPU não detectada no container

```bash
# Verificar variáveis de ambiente
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 env | grep NVIDIA

# Deve mostrar:
# NVIDIA_VISIBLE_DEVICES=all
# NVIDIA_DRIVER_CAPABILITIES=compute,utility
```

### Documentação Adicional

- NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/
- Docker GPU support: https://docs.docker.com/config/containers/resource_constraints/#gpu
- CUDA Docker Hub: https://hub.docker.com/r/nvidia/cuda

## Conclusão

A instalação correta do CUDA e cuDNN é essencial para aproveitar a aceleração GPU no Horus-CDS. Para deployment com Docker, o NVIDIA Container Toolkit permite usar GPUs dentro de containers de forma transparente. Em caso de dúvidas, consulte a documentação oficial da NVIDIA e do TensorFlow para sua plataforma específica.
