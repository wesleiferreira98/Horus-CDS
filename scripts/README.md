# Scripts - Horus-CDS

Esta pasta contém todos os scripts executáveis do projeto Horus-CDS.

## Scripts Disponíveis

### Scripts de Instalação

#### Linux/macOS - install.sh

Script interativo de instalação para sistemas Unix-like.

**Características:**
- Detecção automática de Python 3.12+
- Verificação de dependências
- 3 métodos de instalação:
  1. Docker (Recomendado)
  2. Ambiente Virtual (Python venv)
  3. Instalação Global
- Detecção e suporte para GPU NVIDIA
- Instalação opcional do NVIDIA Container Toolkit
- Menu interativo com cores

**Uso:**
```bash
chmod +x scripts/install.sh
./scripts/install.sh
```

**Requisitos:**
- Sistema Linux ou macOS
- Bash 4.0+
- Python 3.12+ (para métodos 2 e 3)
- Docker 20.10+ (para método 1)

---

#### Windows - install.bat

Script interativo de instalação para Windows.

**Características:**
- Detecção automática de Python 3.12+
- Verificação de dependências
- 3 métodos de instalação:
  1. Docker (Recomendado)
  2. Ambiente Virtual (Python venv)
  3. Instalação Global
- Suporte para ANSI colors (Windows 10+)
- Menu interativo colorido
- Codificação UTF-8

**Uso:**
```cmd
scripts\install.bat
```

**Requisitos:**
- Windows 10 ou superior (para cores)
- Python 3.12+ (para métodos 2 e 3)
- Docker Desktop (para método 1)

---

### Scripts de Teste

#### test_gpu_setup.py

Script automatizado de verificação de configuração GPU/CUDA.

**Funcionalidades:**
- Verificação de 7 componentes críticos:
  1. Versão do Python (>= 3.12)
  2. TensorFlow instalado
  3. Suporte CUDA do TensorFlow
  4. Dispositivos GPU disponíveis
  5. Compute Capability (>= 3.5)
  6. Memória GPU disponível
  7. Bibliotecas CUDA e cuDNN
- Benchmark CPU vs GPU
- Relatório detalhado e colorido
- Exit codes para CI/CD:
  - 0: Sucesso total
  - 1: Falhas críticas
  - 2: Avisos (GPU não disponível)

**Uso:**

Verificação completa:
```bash
python scripts/test_gpu_setup.py
```

Verificação silenciosa (CI/CD):
```bash
python scripts/test_gpu_setup.py --silent
echo $?  # Verifica exit code
```

**Saída Exemplo:**
```
╔══════════════════════════════════════╗
║   VERIFICAÇÃO DE CONFIGURAÇÃO GPU   ║
╚══════════════════════════════════════╝

[✓] Python 3.12.12 detectado
[✓] TensorFlow 2.20.0 instalado
[✓] CUDA disponível no TensorFlow
[✓] 1 GPU(s) detectada(s):
    - GPU 0: NVIDIA GeForce RTX 3080
[✓] Compute Capability: 8.6
[✓] Memória GPU disponível: 10240 MB
[✓] cuDNN 9.5.1 carregado

Benchmark CPU vs GPU:
- CPU: 2.34 segundos
- GPU: 0.15 segundos
- Aceleração: 15.6x
```

---

## Como Usar os Scripts

### Instalação Rápida (Recomendado)

**Linux/macOS:**
```bash
./scripts/install.sh
# Selecione opção 1 (Docker) no menu
```

**Windows:**
```cmd
scripts\install.bat
REM Selecione opção 1 (Docker) no menu
```

### Verificação Pós-Instalação

Após instalar, verifique a configuração GPU:

```bash
python scripts/test_gpu_setup.py
```

### Instalação com GPU

Se você tem GPU NVIDIA:

1. Instale drivers NVIDIA primeiro
2. Instale CUDA e cuDNN (veja [docs/CUDA_INSTALLATION.md](../docs/CUDA_INSTALLATION.md))
3. Execute `install.sh` ou `install.bat`
4. Selecione a opção de instalar NVIDIA Container Toolkit (se usar Docker)
5. Verifique com `test_gpu_setup.py`

---

## Troubleshooting

### install.sh/install.bat

**Problema**: "Python não encontrado"
- **Solução**: Instale Python 3.12+ e adicione ao PATH

**Problema**: "Docker não encontrado"
- **Solução**: Instale Docker e inicie o serviço
  - Linux: `sudo systemctl start docker`
  - Windows: Inicie Docker Desktop

**Problema**: "Permissão negada" (Linux)
- **Solução**: `chmod +x scripts/install.sh`

**Problema**: Cores não aparecem (Windows)
- **Solução**: Use Windows 10+ ou PowerShell

### test_gpu_setup.py

**Problema**: "TensorFlow não encontrado"
- **Solução**: `pip install tensorflow==2.20.0`

**Problema**: "GPU não detectada"
- **Solução**: 
  1. Verifique drivers NVIDIA: `nvidia-smi`
  2. Reinstale CUDA/cuDNN
  3. Consulte [docs/CUDA_INSTALLATION.md](../docs/CUDA_INSTALLATION.md)

**Problema**: "CUDA version mismatch"
- **Solução**: Alinhe versões CUDA, cuDNN e TensorFlow
  - TensorFlow 2.20.0 → CUDA 12.3+ / cuDNN 9.0+

---

## Integração CI/CD

### GitHub Actions Exemplo

```yaml
name: Test GPU Setup
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: pip install tensorflow==2.20.0
      - name: Run GPU test
        run: python scripts/test_gpu_setup.py --silent
```

### Exit Codes

- `0`: Todos os testes passaram (GPU funcional)
- `1`: Falhas críticas (TensorFlow, Python, etc.)
- `2`: Avisos (GPU não disponível, mas sistema OK)

---

## Documentação Adicional

- **Guia completo dos scripts**: [../docs/INSTALL_SCRIPTS.md](../docs/INSTALL_SCRIPTS.md)
- **Instalação CUDA**: [../docs/CUDA_INSTALLATION.md](../docs/CUDA_INSTALLATION.md)
- **Docker deployment**: [../docs/DOCKER.md](../docs/DOCKER.md)
- **Quick start**: [../docs/QUICKSTART.md](../docs/QUICKSTART.md)

---

## Contribuindo

Para adicionar novos scripts:

1. Mantenha compatibilidade cross-platform quando possível
2. Use cores ANSI para feedback visual
3. Adicione verificação de requisitos
4. Documente o script neste README
5. Inclua exit codes apropriados
6. Adicione exemplos de uso

---

## Suporte

Para problemas com os scripts:
- Consulte a seção Troubleshooting acima
- Leia [../docs/INSTALL_SCRIPTS.md](../docs/INSTALL_SCRIPTS.md)
- Abra uma issue no GitHub
- Verifique os logs de erro

---

**Última atualização**: Novembro 2025
