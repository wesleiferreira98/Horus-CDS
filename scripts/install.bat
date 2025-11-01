@echo off
setlocal enabledelayedexpansion

REM ################################################################################
REM # Script de Instalacao Rapida - Horus-CDS
REM # Sistema de Deteccao de Ataques Ciberneticos para Smart Grids
REM #
REM # Autor: Horus-CDS Team
REM # Data: Novembro 2025
REM # Descricao: Instalador interativo com multiplas opcoes de deployment
REM ################################################################################

REM Configurar codepage UTF-8
chcp 65001 > nul

REM Cores (usando ANSI escape codes - requer Windows 10+)
set "RED=[91m"
set "GREEN=[92m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "MAGENTA=[95m"
set "CYAN=[96m"
set "WHITE=[97m"
set "NC=[0m"
set "BOLD=[1m"

REM Habilitar cores ANSI no Windows
reg add HKCU\Console /v VirtualTerminalLevel /t REG_DWORD /d 1 /f > nul 2>&1

REM ============================================================================
REM FUNCOES
REM ============================================================================

:print_banner
cls
echo %CYAN%
echo     ╔═════════════════════════════════════════════════════════════════════════════╗
echo     ║                                                                             ║
echo     ║   ██╗  ██╗ ██████╗ ██████╗ ██╗   ██╗███████╗      ██████╗██████╗  ███████╗  ║
echo     ║   ██║  ██║██╔═══██╗██╔══██╗██║   ██║██╔════╝      ██╔════╝██╔══██╗██╔════╝  ║
echo     ║   ███████║██║   ██║██████╔╝██║   ██║███████╗█████╗██║     ██║  ██║███████╗  ║
echo     ║   ██╔══██║██║   ██║██╔══██╗██║   ██║╚════██║╚════╝██║     ██║  ██║╚════██║  ║
echo     ║   ██║  ██║╚██████╔╝██║  ██║╚██████╔╝███████║     ╚██████╗██████╔╝ ███████║  ║
echo     ║   ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝      ╚═════╝╚═════╝ ╚══════╝   ║
echo     ║                                                                             ║
echo     ║          Cyber Detection System for Smart Grids                             ║
echo     ║                 Sistema de Deteccao de Ataques                              ║
echo     ║                                                                             ║
echo     ╚═════════════════════════════════════════════════════════════════════════════╝
echo %NC%
goto :eof

:print_separator
echo %BLUE%═══════════════════════════════════════════════════════════════════════%NC%
goto :eof

:print_success
echo %GREEN%[✓]%NC% %~1
goto :eof

:print_error
echo %RED%[✗]%NC% %~1
goto :eof

:print_warning
echo %YELLOW%[!]%NC% %~1
goto :eof

:print_info
echo %CYAN%[i]%NC% %~1
goto :eof

:print_section
echo.
echo %BOLD%%WHITE%=== %~1 ===%NC%
call :print_separator
goto :eof

REM ============================================================================
REM VERIFICACOES
REM ============================================================================

:check_python
call :print_info "Verificando instalacao do Python..."

REM Verificar Python 3.12
python --version > nul 2>&1
if !errorlevel! equ 0 (
    for /f "tokens=2" %%V in ('python --version 2^>^&1') do set PYTHON_VERSION=%%V
    
    REM Extrair versao major.minor
    for /f "tokens=1,2 delims=." %%a in ("!PYTHON_VERSION!") do (
        set PYTHON_MAJOR=%%a
        set PYTHON_MINOR=%%b
    )
    
    if !PYTHON_MAJOR! geq 3 (
        if !PYTHON_MINOR! geq 9 (
            set PYTHON_CMD=python
            call :print_success "Python !PYTHON_VERSION! encontrado (compativel)"
            exit /b 0
        )
    )
    
    call :print_error "Python 3.9+ necessario. Versao encontrada: !PYTHON_VERSION!"
    call :print_info "Baixe Python 3.12 em: https://www.python.org/downloads/"
    exit /b 1
) else (
    call :print_error "Python nao encontrado"
    call :print_info "Instale Python 3.12 de: https://www.python.org/downloads/"
    exit /b 1
)

:check_docker
call :print_info "Verificando instalacao do Docker..."

docker --version > nul 2>&1
if !errorlevel! equ 0 (
    for /f "tokens=3" %%V in ('docker --version') do set DOCKER_VERSION=%%V
    call :print_success "Docker encontrado (versao !DOCKER_VERSION!)"
    
    docker compose version > nul 2>&1
    if !errorlevel! equ 0 (
        call :print_success "Docker Compose encontrado"
        exit /b 0
    ) else (
        call :print_warning "Docker Compose nao encontrado"
        call :print_info "Instale Docker Desktop: https://www.docker.com/products/docker-desktop"
        exit /b 1
    )
) else (
    call :print_error "Docker nao encontrado"
    call :print_info "Instale Docker Desktop: https://www.docker.com/products/docker-desktop"
    exit /b 1
)

:check_gpu
call :print_info "Verificando GPU NVIDIA..."

nvidia-smi > nul 2>&1
if !errorlevel! equ 0 (
    for /f "tokens=*" %%G in ('nvidia-smi --query-gpu^=name --format^=csv^,noheader 2^>nul') do (
        call :print_success "GPU detectada: %%G"
    )
    
    nvcc --version > nul 2>&1
    if !errorlevel! equ 0 (
        for /f "tokens=5" %%V in ('nvcc --version ^| findstr "release"') do (
            set CUDA_VERSION=%%V
            call :print_success "CUDA Toolkit instalado: !CUDA_VERSION!"
        )
    ) else (
        call :print_warning "CUDA Toolkit nao instalado"
        call :print_info "Consulte: CUDA_INSTALLATION.md"
    )
    exit /b 0
) else (
    call :print_warning "GPU NVIDIA nao detectada ou drivers nao instalados"
    call :print_info "Sistema funcionara apenas com CPU"
    exit /b 1
)

REM ============================================================================
REM METODOS DE INSTALACAO
REM ============================================================================

:install_method_1
call :print_banner
call :print_section "METODO 1: Instalacao Global (Nao Recomendado)"

call :print_warning "Este metodo instala as dependencias globalmente no sistema"
call :print_warning "Pode causar conflitos com outras instalacoes Python"
echo.
set /p confirm="Deseja continuar? (s/N): "
if /i not "!confirm!"=="s" (
    call :print_info "Instalacao cancelada"
    goto :menu
)

call :print_section "Instalando Dependencias"

call :check_python
if !errorlevel! neq 0 goto :menu

call :print_info "Atualizando pip..."
!PYTHON_CMD! -m pip install --upgrade pip

call :print_info "Instalando dependencias do requirements.txt..."
!PYTHON_CMD! -m pip install -r requirements.txt

if !errorlevel! equ 0 (
    call :print_success "Dependencias instaladas com sucesso"
    
    call :print_section "Verificando Instalacao"
    !PYTHON_CMD! test_gpu_setup.py
    
    call :print_section "INSTALACAO CONCLUIDA"
    call :print_success "Horus-CDS instalado globalmente"
    call :print_info "Execute a API com: cd root\API && python app.py"
    call :print_info "Execute o treinamento com: cd root\Linux && python main.py"
    call :print_info "Execute o dashboard com: cd root\web && python run_web.py"
) else (
    call :print_error "Erro ao instalar dependencias"
)
goto :menu

:install_method_2
call :print_banner
call :print_section "METODO 2: Instalacao com Ambiente Virtual (Recomendado)"

call :print_info "Este metodo cria um ambiente isolado para o projeto"
call :print_success "Metodo recomendado para desenvolvimento"
echo.

call :check_python
if !errorlevel! neq 0 goto :menu

call :print_section "Criando Ambiente Virtual"

if exist "venv-Horus\" (
    call :print_warning "Ambiente virtual 'venv-Horus' ja existe"
    set /p confirm="Deseja recria-lo? (s/N): "
    if /i "!confirm!"=="s" (
        call :print_info "Removendo ambiente antigo..."
        rmdir /s /q venv-Horus
    ) else (
        call :print_info "Usando ambiente existente"
    )
)

if not exist "venv-Horus\" (
    call :print_info "Criando venv-Horus..."
    !PYTHON_CMD! -m venv venv-Horus
    
    if !errorlevel! neq 0 (
        call :print_error "Erro ao criar ambiente virtual"
        goto :menu
    )
    call :print_success "Ambiente virtual criado"
)

call :print_section "Instalando Dependencias"

call :print_info "Ativando ambiente virtual..."
call venv-Horus\Scripts\activate.bat

call :print_info "Atualizando pip..."
python -m pip install --upgrade pip

call :print_info "Instalando dependencias do requirements.txt..."
pip install -r requirements.txt

if !errorlevel! equ 0 (
    call :print_success "Dependencias instaladas com sucesso"
    
    call :print_section "Verificando Instalacao"
    call :check_gpu
    python test_gpu_setup.py
    
    call :print_section "INSTALACAO CONCLUIDA"
    call :print_success "Horus-CDS instalado com sucesso no ambiente virtual"
    echo.
    call :print_info "Para ativar o ambiente virtual:"
    echo   %YELLOW%venv-Horus\Scripts\activate%NC%
    echo.
    call :print_info "Comandos para execucao:"
    echo   %CYAN%API:%NC%        cd root\API ^&^& python app.py
    echo   %CYAN%Treinamento:%NC% cd root\Linux ^&^& python main.py
    echo   %CYAN%Dashboard:%NC%   cd root\web ^&^& python run_web.py
    echo.
    call :print_info "Para desativar o ambiente virtual:"
    echo   %YELLOW%deactivate%NC%
    
    call venv-Horus\Scripts\deactivate.bat
) else (
    call :print_error "Erro ao instalar dependencias"
    call venv-Horus\Scripts\deactivate.bat
)
goto :menu

:install_method_3
call :print_banner
call :print_section "METODO 3: Instalacao com Docker"

call :print_info "Este metodo usa containers Docker para isolamento completo"
call :print_success "Recomendado para producao"
echo.

call :check_docker
if !errorlevel! neq 0 goto :menu

REM Verificar GPU
set HAS_GPU=false
call :check_gpu
if !errorlevel! equ 0 (
    echo.
    set /p gpu_confirm="Deseja habilitar suporte GPU? (S/n): "
    if /i not "!gpu_confirm!"=="n" (
        set HAS_GPU=true
        call :print_warning "Suporte GPU no Docker requer Docker Desktop com WSL2"
        call :print_info "Consulte: DOCKER.md para configuracao"
    )
)

call :print_section "Construindo Imagens Docker"

if "!HAS_GPU!"=="true" (
    call :print_info "Usando configuracao com GPU (docker-compose-gpu.yml)..."
    set COMPOSE_FILE=docker-compose-gpu.yml
) else (
    call :print_info "Usando configuracao CPU-only (docker-compose.yml)..."
    set COMPOSE_FILE=docker-compose.yml
)

call :print_info "Construindo imagens... (isso pode levar alguns minutos)"

docker compose -f !COMPOSE_FILE! build
if !errorlevel! equ 0 (
    call :print_success "Imagens construidas com sucesso"
    
    call :print_section "Iniciando Containers"
    
    docker compose -f !COMPOSE_FILE! up -d
    if !errorlevel! equ 0 (
        call :print_success "Containers iniciados com sucesso"
        
        timeout /t 3 /nobreak > nul
        
        call :print_section "Status dos Containers"
        docker compose -f !COMPOSE_FILE! ps
        
        call :print_section "INSTALACAO CONCLUIDA"
        call :print_success "Horus-CDS instalado e executando em containers Docker"
        echo.
        call :print_info "Servicos disponiveis:"
        echo   %GREEN%API:%NC%       http://localhost:5000
        echo   %GREEN%Dashboard:%NC% http://localhost:5001
        echo.
        call :print_info "Comandos uteis:"
        echo   %CYAN%Ver logs:%NC%      docker compose -f !COMPOSE_FILE! logs -f
        echo   %CYAN%Parar:%NC%         docker compose -f !COMPOSE_FILE! down
        echo   %CYAN%Reiniciar:%NC%     docker compose -f !COMPOSE_FILE! restart
        echo   %CYAN%Status:%NC%        docker compose -f !COMPOSE_FILE! ps
        
        if "!HAS_GPU!"=="true" (
            echo.
            call :print_info "Para verificar GPU no container:"
            echo   %YELLOW%docker exec horus-cds-api-gpu nvidia-smi%NC%
        )
    ) else (
        call :print_error "Erro ao iniciar containers"
    )
) else (
    call :print_error "Erro ao construir imagens Docker"
)
goto :menu

REM ============================================================================
REM MENU PRINCIPAL
REM ============================================================================

:menu
call :print_banner

echo %BOLD%%WHITE%Bem-vindo ao Instalador do Horus-CDS%NC%
echo Sistema de Deteccao de Ataques Ciberneticos para Smart Grids
echo.
call :print_separator
echo.
echo %BOLD%Escolha o metodo de instalacao:%NC%
echo.
echo   %YELLOW%1%NC% - Instalacao Global (sem venv, sem Docker)
echo       %WHITE%Nao recomendado - instala dependencias globalmente%NC%
echo.
echo   %GREEN%2%NC% - Instalacao com Ambiente Virtual %BOLD%%GREEN%(RECOMENDADO)%NC%
echo       %WHITE%Cria ambiente Python isolado (venv-Horus)%NC%
echo.
echo   %BLUE%3%NC% - Instalacao com Docker
echo       %WHITE%Usa containers para isolamento completo%NC%
echo.
echo   %RED%0%NC% - Sair
echo.
call :print_separator
echo.

set /p option="Digite sua opcao: "

if "!option!"=="1" goto :install_method_1
if "!option!"=="2" goto :install_method_2
if "!option!"=="3" goto :install_method_3
if "!option!"=="0" goto :exit_script

call :print_error "Opcao invalida"
timeout /t 2 /nobreak > nul
goto :menu

:exit_script
call :print_banner
call :print_info "Instalacao cancelada pelo usuario"
echo.
pause
exit /b 0

REM ============================================================================
REM MAIN
REM ============================================================================

:main
REM Verificar se esta no diretorio correto
if not exist "requirements.txt" (
    call :print_banner
    call :print_error "Arquivo requirements.txt nao encontrado"
    call :print_info "Execute este script no diretorio raiz do Horus-CDS"
    pause
    exit /b 1
)

goto :menu
