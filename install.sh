#!/bin/bash

################################################################################
# Script de Instalação Rápida - Horus-CDS
# Sistema de Detecção de Ataques Cibernéticos para Smart Grids
#
# Autor: Horus-CDS Team
# Data: Novembro 2025
# Descrição: Instalador interativo com múltiplas opções de deployment
################################################################################

# Cores ANSI
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Função para limpar tela
clear_screen() {
    clear
}

# Função para imprimir banner ASCII
print_banner() {
    echo -e "${CYAN}"
    cat << "EOF"
    ╔════════════════════════════════════════════════════════════════════════════════╗
    ║                                                                                ║
    ║   ██╗  ██╗ ██████╗ ██████╗ ██╗   ██╗███████╗      ██████╗██████╗  ███████╗     ║
    ║   ██║  ██║██╔═══██╗██╔══██╗██║   ██║██╔════╝      ██╔════╝██╔══██╗██╔════╝     ║
    ║   ███████║██║   ██║██████╔╝██║   ██║███████╗█████╗██║     ██║  ██║███████╗     ║
    ║   ██╔══██║██║   ██║██╔══██╗██║   ██║╚════██║╚════╝██║     ██║  ██║╚════██║     ║
    ║   ██║  ██║╚██████╔╝██║  ██║╚██████╔╝███████║     ╚██████╗██████╔╝ ███████║     ║
    ║   ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝      ╚═════╝╚═════╝  ╚══════╝     ║
    ║                                                                                ║
    ║          Cyber Detection System for Smart Grids                                ║
    ║                 Sistema de Detecção de Ataques                                 ║
    ║                                                                                ║
    ╚════════════════════════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

# Função para imprimir linha separadora
print_separator() {
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
}

# Função para imprimir mensagem de sucesso
print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

# Função para imprimir mensagem de erro
print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Função para imprimir mensagem de aviso
print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Função para imprimir mensagem de informação
print_info() {
    echo -e "${CYAN}[i]${NC} $1"
}

# Função para imprimir título de seção
print_section() {
    echo -e "\n${BOLD}${WHITE}$1${NC}"
    print_separator
}

# Função para verificar se comando existe
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Função para verificar Python 3.12
check_python() {
    print_info "Verificando instalação do Python..."
    
    if command_exists python3.12; then
        PYTHON_CMD="python3.12"
        print_success "Python 3.12 encontrado"
        return 0
    elif command_exists python3; then
        VERSION=$(python3 --version 2>&1 | awk '{print $2}')
        MAJOR=$(echo $VERSION | cut -d. -f1)
        MINOR=$(echo $VERSION | cut -d. -f2)
        
        if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 9 ]; then
            PYTHON_CMD="python3"
            print_success "Python $VERSION encontrado (compatível)"
            return 0
        else
            print_error "Python 3.9+ necessário. Versão encontrada: $VERSION"
            return 1
        fi
    else
        print_error "Python 3 não encontrado"
        print_info "Instale Python 3.12 com: sudo apt install python3.12 python3.12-venv"
        return 1
    fi
}

# Função para verificar Docker
check_docker() {
    print_info "Verificando instalação do Docker..."
    
    if command_exists docker; then
        DOCKER_VERSION=$(docker --version | awk '{print $3}' | tr -d ',')
        print_success "Docker encontrado (versão $DOCKER_VERSION)"
        
        if command_exists docker-compose || docker compose version >/dev/null 2>&1; then
            print_success "Docker Compose encontrado"
            return 0
        else
            print_warning "Docker Compose não encontrado"
            print_info "Instale com: sudo apt install docker-compose-plugin"
            return 1
        fi
    else
        print_error "Docker não encontrado"
        print_info "Instale Docker seguindo: https://docs.docker.com/engine/install/"
        return 1
    fi
}

# Função para verificar GPU NVIDIA
check_gpu() {
    print_info "Verificando GPU NVIDIA..."
    
    if command_exists nvidia-smi; then
        GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        if [ -n "$GPU_INFO" ]; then
            print_success "GPU detectada: $GPU_INFO"
            
            # Verificar CUDA
            if command_exists nvcc; then
                CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | tr -d ',')
                print_success "CUDA Toolkit instalado: $CUDA_VERSION"
            else
                print_warning "CUDA Toolkit não instalado"
                print_info "Consulte: CUDA_INSTALLATION.md"
            fi
            return 0
        fi
    fi
    
    print_warning "GPU NVIDIA não detectada ou drivers não instalados"
    print_info "Sistema funcionará apenas com CPU"
    return 1
}

# Instalação Método 1: Sem venv e sem Docker
install_method_1() {
    clear_screen
    print_banner
    print_section "MÉTODO 1: Instalação Global (Não Recomendado)"
    
    print_warning "Este método instala as dependências globalmente no sistema"
    print_warning "Pode causar conflitos com outras instalações Python"
    echo ""
    read -p "Deseja continuar? (s/N): " confirm
    
    if [[ ! "$confirm" =~ ^[Ss]$ ]]; then
        print_info "Instalação cancelada"
        return 1
    fi
    
    print_section "Instalando Dependências"
    
    if ! check_python; then
        return 1
    fi
    
    print_info "Atualizando pip..."
    $PYTHON_CMD -m pip install --upgrade pip
    
    print_info "Instalando dependências do requirements.txt..."
    $PYTHON_CMD -m pip install -r requirements.txt
    
    if [ $? -eq 0 ]; then
        print_success "Dependências instaladas com sucesso"
        
        print_section "Verificando Instalação"
        $PYTHON_CMD test_gpu_setup.py
        
        print_section "INSTALAÇÃO CONCLUÍDA"
        print_success "Horus-CDS instalado globalmente"
        print_info "Execute a API com: cd root/API && python app.py"
        print_info "Execute o treinamento com: cd root/Linux && python main.py"
        print_info "Execute o dashboard com: cd root/web && python run_web.py"
    else
        print_error "Erro ao instalar dependências"
        return 1
    fi
}

# Instalação Método 2: Com venv (Recomendado)
install_method_2() {
    clear_screen
    print_banner
    print_section "MÉTODO 2: Instalação com Ambiente Virtual (Recomendado)"
    
    print_info "Este método cria um ambiente isolado para o projeto"
    print_success "Método recomendado para desenvolvimento"
    echo ""
    
    if ! check_python; then
        return 1
    fi
    
    print_section "Criando Ambiente Virtual"
    
    if [ -d "venv-Horus" ]; then
        print_warning "Ambiente virtual 'venv-Horus' já existe"
        read -p "Deseja recriá-lo? (s/N): " confirm
        if [[ "$confirm" =~ ^[Ss]$ ]]; then
            print_info "Removendo ambiente antigo..."
            rm -rf venv-Horus
        else
            print_info "Usando ambiente existente"
        fi
    fi
    
    if [ ! -d "venv-Horus" ]; then
        print_info "Criando venv-Horus..."
        $PYTHON_CMD -m venv venv-Horus
        
        if [ $? -ne 0 ]; then
            print_error "Erro ao criar ambiente virtual"
            print_info "Instale o módulo venv: sudo apt install python3-venv"
            return 1
        fi
        print_success "Ambiente virtual criado"
    fi
    
    print_section "Instalando Dependências"
    
    print_info "Ativando ambiente virtual..."
    source venv-Horus/bin/activate
    
    print_info "Atualizando pip..."
    pip install --upgrade pip
    
    print_info "Instalando dependências do requirements.txt..."
    pip install -r requirements.txt
    
    if [ $? -eq 0 ]; then
        print_success "Dependências instaladas com sucesso"
        
        print_section "Verificando Instalação"
        check_gpu
        python test_gpu_setup.py
        
        print_section "INSTALAÇÃO CONCLUÍDA"
        print_success "Horus-CDS instalado com sucesso no ambiente virtual"
        echo ""
        print_info "Para ativar o ambiente virtual:"
        echo -e "  ${YELLOW}source venv-Horus/bin/activate${NC}"
        echo ""
        print_info "Comandos para execução:"
        echo -e "  ${CYAN}API:${NC}        cd root/API && python app.py"
        echo -e "  ${CYAN}Treinamento:${NC} cd root/Linux && python main.py"
        echo -e "  ${CYAN}Dashboard:${NC}   cd root/web && python run_web.py"
        echo ""
        print_info "Para desativar o ambiente virtual:"
        echo -e "  ${YELLOW}deactivate${NC}"
    else
        print_error "Erro ao instalar dependências"
        deactivate
        return 1
    fi
}

# Instalação Método 3: Com Docker
install_method_3() {
    clear_screen
    print_banner
    print_section "MÉTODO 3: Instalação com Docker"
    
    print_info "Este método usa containers Docker para isolamento completo"
    print_success "Recomendado para produção"
    echo ""
    
    if ! check_docker; then
        return 1
    fi
    
    # Verificar GPU
    HAS_GPU=false
    if check_gpu; then
        echo ""
        read -p "Deseja habilitar suporte GPU? (S/n): " gpu_confirm
        if [[ ! "$gpu_confirm" =~ ^[Nn]$ ]]; then
            HAS_GPU=true
            
            # Verificar NVIDIA Container Toolkit
            print_info "Verificando NVIDIA Container Toolkit..."
            if docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
                print_success "NVIDIA Container Toolkit configurado"
            else
                print_warning "NVIDIA Container Toolkit não instalado"
                print_info "Instalando NVIDIA Container Toolkit..."
                
                distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
                curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
                curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
                    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
                
                sudo apt-get update
                sudo apt-get install -y nvidia-container-toolkit
                sudo systemctl restart docker
                
                print_success "NVIDIA Container Toolkit instalado"
            fi
        fi
    fi
    
    print_section "Construindo Imagens Docker"
    
    if [ "$HAS_GPU" = true ]; then
        print_info "Usando configuração com GPU (docker-compose-gpu.yml)..."
        COMPOSE_FILE="docker-compose-gpu.yml"
    else
        print_info "Usando configuração CPU-only (docker-compose.yml)..."
        COMPOSE_FILE="docker-compose.yml"
    fi
    
    print_info "Construindo imagens... (isso pode levar alguns minutos)"
    
    if docker compose -f "$COMPOSE_FILE" build; then
        print_success "Imagens construídas com sucesso"
        
        print_section "Iniciando Containers"
        
        if docker compose -f "$COMPOSE_FILE" up -d; then
            print_success "Containers iniciados com sucesso"
            
            sleep 3
            
            print_section "Status dos Containers"
            docker compose -f "$COMPOSE_FILE" ps
            
            print_section "INSTALAÇÃO CONCLUÍDA"
            print_success "Horus-CDS instalado e executando em containers Docker"
            echo ""
            print_info "Serviços disponíveis:"
            echo -e "  ${GREEN}API:${NC}       http://localhost:5000"
            echo -e "  ${GREEN}Dashboard:${NC} http://localhost:5001"
            echo ""
            print_info "Comandos úteis:"
            echo -e "  ${CYAN}Ver logs:${NC}      docker compose -f $COMPOSE_FILE logs -f"
            echo -e "  ${CYAN}Parar:${NC}         docker compose -f $COMPOSE_FILE down"
            echo -e "  ${CYAN}Reiniciar:${NC}     docker compose -f $COMPOSE_FILE restart"
            echo -e "  ${CYAN}Status:${NC}        docker compose -f $COMPOSE_FILE ps"
            
            if [ "$HAS_GPU" = true ]; then
                echo ""
                print_info "Para verificar GPU no container:"
                echo -e "  ${YELLOW}docker exec horus-cds-api-gpu nvidia-smi${NC}"
            fi
        else
            print_error "Erro ao iniciar containers"
            return 1
        fi
    else
        print_error "Erro ao construir imagens Docker"
        return 1
    fi
}

# Menu principal
show_menu() {
    clear_screen
    print_banner
    
    echo -e "${BOLD}${WHITE}Bem-vindo ao Instalador do Horus-CDS${NC}"
    echo -e "Sistema de Detecção de Ataques Cibernéticos para Smart Grids"
    echo ""
    print_separator
    echo ""
    echo -e "${BOLD}Escolha o método de instalação:${NC}"
    echo ""
    echo -e "  ${YELLOW}1${NC} - Instalação Global (sem venv, sem Docker)"
    echo -e "      ${WHITE}Não recomendado - instala dependências globalmente${NC}"
    echo ""
    echo -e "  ${GREEN}2${NC} - Instalação com Ambiente Virtual ${BOLD}${GREEN}(RECOMENDADO)${NC}"
    echo -e "      ${WHITE}Cria ambiente Python isolado (venv-Horus)${NC}"
    echo ""
    echo -e "  ${BLUE}3${NC} - Instalação com Docker"
    echo -e "      ${WHITE}Usa containers para isolamento completo${NC}"
    echo ""
    echo -e "  ${RED}0${NC} - Sair"
    echo ""
    print_separator
    echo ""
}

# Função principal
main() {
    # Verificar se está no diretório correto
    if [ ! -f "requirements.txt" ]; then
        clear_screen
        print_banner
        print_error "Arquivo requirements.txt não encontrado"
        print_info "Execute este script no diretório raiz do Horus-CDS"
        exit 1
    fi
    
    while true; do
        show_menu
        read -p "Digite sua opção: " option
        
        case $option in
            1)
                install_method_1
                echo ""
                read -p "Pressione ENTER para continuar..."
                ;;
            2)
                install_method_2
                echo ""
                read -p "Pressione ENTER para continuar..."
                ;;
            3)
                install_method_3
                echo ""
                read -p "Pressione ENTER para continuar..."
                ;;
            0)
                clear_screen
                print_banner
                print_info "Instalação cancelada pelo usuário"
                echo ""
                exit 0
                ;;
            *)
                print_error "Opção inválida"
                sleep 2
                ;;
        esac
    done
}

# Verificar se está executando como root (não recomendado)
if [ "$EUID" -eq 0 ]; then
    clear_screen
    print_banner
    print_warning "Este script não deve ser executado como root"
    print_info "Execute como usuário normal: ./install.sh"
    exit 1
fi

# Executar função principal
main
