# Scripts de Instalação Automática - Horus-CDS

## Visão Geral

Os scripts `install.sh` (Linux/macOS) e `install.bat` (Windows) fornecem instalação guiada e interativa do Horus-CDS com verificação automática de dependências e configuração do ambiente.

## Características

### Interface Visual
- **ASCII Art**: Banner do Horus-CDS em arte ASCII
- **Cores ANSI**: Mensagens coloridas para melhor legibilidade
  - Verde: Sucesso
  - Vermelho: Erro
  - Amarelo: Aviso
  - Ciano: Informação
  - Azul: Separadores
- **Menu Interativo**: Navegação simples por números
- **Feedback Visual**: Indicadores de progresso e status

### Funcionalidades

#### Verificações Automáticas
1. **Python**: Verifica versão 3.9+ (recomenda 3.12)
2. **Docker**: Detecta Docker Engine e Docker Compose
3. **GPU NVIDIA**: Identifica GPU, driver e CUDA Toolkit
4. **Dependências**: Valida instalação de bibliotecas

#### Três Métodos de Instalação

**Método 1: Instalação Global**
- Instala dependências no Python do sistema
- Não recomendado (conflitos possíveis)
- Requer confirmação explícita

**Método 2: Ambiente Virtual (Recomendado)**
- Cria `venv-Horus` isolado
- Instala dependências no venv
- Executa test_gpu_setup.py automaticamente
- Fornece comandos para ativação/desativação

**Método 3: Docker**
- Detecta suporte GPU
- Escolhe docker-compose.yml apropriado (CPU/GPU)
- Instala NVIDIA Container Toolkit (Linux)
- Constrói e inicia containers automaticamente

## Uso

### Linux / macOS

```bash
# Clonar repositório
git clone https://github.com/wesleiferreira98/Horus-CDS.git
cd Horus-CDS

# Tornar executável (se necessário)
chmod +x install.sh

# Executar instalador
./install.sh
```

### Windows

```cmd
# Clonar repositório
git clone https://github.com/wesleiferreira98/Horus-CDS.git
cd Horus-CDS

# Executar instalador
install.bat
```

**Nota**: Requer Windows 10+ para cores ANSI.

## Fluxo de Instalação

### 1. Verificação Inicial
- Confirma presença de `requirements.txt`
- Exibe banner do Horus-CDS
- Apresenta menu de opções

### 2. Seleção de Método
```
Escolha o método de instalação:

  1 - Instalação Global (sem venv, sem Docker)
      Não recomendado - instala dependências globalmente

  2 - Instalação com Ambiente Virtual (RECOMENDADO)
      Cria ambiente Python isolado (venv-Horus)

  3 - Instalação com Docker
      Usa containers para isolamento completo

  0 - Sair
```

### 3. Execução
- Verifica pré-requisitos
- Instala/configura ambiente
- Valida instalação
- Exibe instruções de uso

### 4. Finalização
- Resumo da instalação
- Comandos para executar aplicação
- Informações de acesso (URLs)

## Saída de Exemplo

### Banner Inicial
```
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║   ██╗  ██╗ ██████╗ ██████╗ ██╗   ██╗███████╗      ██████╗██████╗███╗ ║
║   ██║  ██║██╔═══██╗██╔══██╗██║   ██║██╔════╝     ██╔════╝██╔══██╗████║ ║
║   ███████║██║   ██║██████╔╝██║   ██║███████╗█████╗██║     ██║  ██║╚██║ ║
║   ██╔══██║██║   ██║██╔══██╗██║   ██║╚════██║╚════╝██║     ██║  ██║ ██║ ║
║   ██║  ██║╚██████╔╝██║  ██║╚██████╔╝███████║     ╚██████╗██████╔╝ ██║ ║
║   ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝      ╚═════╝╚═════╝  ╚═╝ ║
║                                                                       ║
║          Cyber Detection System for Smart Grids                      ║
║                 Sistema de Detecção de Ataques                       ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
```

### Verificações
```
[i] Verificando instalação do Python...
[✓] Python 3.12.12 encontrado (compatível)
[i] Verificando GPU NVIDIA...
[✓] GPU detectada: NVIDIA GeForce RTX 4050 Laptop GPU
[✓] CUDA Toolkit instalado: 12.6
```

### Instalação Completa (venv)
```
=== INSTALAÇÃO CONCLUÍDA ===
[✓] Horus-CDS instalado com sucesso no ambiente virtual

[i] Para ativar o ambiente virtual:
  source venv-Horus/bin/activate

[i] Comandos para execução:
  API:        cd root/API && python app.py
  Treinamento: cd root/Linux && python main.py
  Dashboard:   cd root/web && python run_web.py

[i] Para desativar o ambiente virtual:
  deactivate
```

### Instalação Completa (Docker)
```
=== INSTALAÇÃO CONCLUÍDA ===
[✓] Horus-CDS instalado e executando em containers Docker

[i] Serviços disponíveis:
  API:       http://localhost:5000
  Dashboard: http://localhost:5001

[i] Comandos úteis:
  Ver logs:      docker compose -f docker-compose.yml logs -f
  Parar:         docker compose -f docker-compose.yml down
  Reiniciar:     docker compose -f docker-compose.yml restart
  Status:        docker compose -f docker-compose.yml ps
```

## Detalhes Técnicos

### install.sh (Linux/macOS)

**Tamanho**: ~15KB
**Linhas**: ~600 linhas
**Shell**: Bash

**Funções principais**:
- `print_banner()`: Exibe ASCII art
- `check_python()`: Valida versão Python
- `check_docker()`: Verifica Docker/Compose
- `check_gpu()`: Detecta GPU e CUDA
- `install_method_X()`: Executa instalação

**Recursos**:
- Variáveis de ambiente preservadas
- Exit codes apropriados
- Tratamento de erros
- Prompts interativos
- Cores ANSI

### install.bat (Windows)

**Tamanho**: ~12KB
**Linhas**: ~450 linhas
**Shell**: Batch (CMD)

**Características**:
- Codepage UTF-8 (chcp 65001)
- ANSI escape codes (Windows 10+)
- Registro VirtualTerminalLevel
- Funções simuladas com `:labels`
- Delayed expansion habilitada

**Limitações**:
- Requer Windows 10+ para cores
- Fallback para texto simples em versões antigas

## Requisitos

### Linux / macOS
- Bash 4.0+
- Git
- Permissões de usuário normal (não root)

### Windows
- Windows 10 ou superior
- Git para Windows
- PowerShell ou CMD
- Permissões de administrador (opcional, para Docker)

## Resolução de Problemas

### Script não executa (Linux)
```bash
# Tornar executável
chmod +x install.sh

# Verificar shebang
head -1 install.sh  # Deve ser #!/bin/bash
```

### Cores não aparecem (Windows)
```cmd
# Habilitar ANSI manualmente
reg add HKCU\Console /v VirtualTerminalLevel /t REG_DWORD /d 1 /f
```

### Erro de permissões (Linux)
```bash
# Não executar como root
./install.sh  # Correto
sudo ./install.sh  # Errado - script detecta e aborta
```

### Docker não encontrado
```bash
# Instalar Docker
# Linux: https://docs.docker.com/engine/install/
# Windows: https://www.docker.com/products/docker-desktop
```

## Integração com CI/CD

Os scripts podem ser usados em pipelines CI/CD com modo não-interativo:

```bash
# Instalação automática (opção 2 - venv)
echo "2" | ./install.sh

# Ou com variável de ambiente
export HORUS_INSTALL_METHOD=2
./install.sh < /dev/null
```

## Contribuindo

Para modificar os scripts:

1. Mantenha formatação consistente
2. Teste em sistemas limpos
3. Preserve compatibilidade
4. Documente alterações

## Changelog

### Versão 1.0.0 (Novembro 2025)
- Criação inicial dos scripts
- Suporte Linux, macOS e Windows
- Três métodos de instalação
- Verificação automática de GPU
- Interface colorida com ASCII art
- Integração com test_gpu_setup.py

## Licença

Mesma licença do Horus-CDS (veja LICENSE).

## Suporte

Para problemas com os scripts:
- Abra issue no GitHub: https://github.com/wesleiferreira98/Horus-CDS/issues
- Consulte documentação: README.md, QUICKSTART.md
- Verifique logs de erro

## Recursos Relacionados

- [README.md](README.md) - Documentação principal
- [QUICKSTART.md](QUICKSTART.md) - Guia rápido
- [CUDA_INSTALLATION.md](CUDA_INSTALLATION.md) - Instalação GPU
- [DOCKER.md](DOCKER.md) - Guia Docker
- [test_gpu_setup.py](test_gpu_setup.py) - Validação GPU
