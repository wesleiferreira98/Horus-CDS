#!/bin/bash

################################################################################
# start.sh - Inicializador do Horus-CDS
# Sobe a API (porta 5000) e o Dashboard Web (porta 5001) automaticamente.
#
# Uso:
#   ./start.sh           → inicia os serviços e aguarda (Ctrl+C para parar)
#   ./start.sh --logs    → mesmo, mas exibe os logs ao vivo no terminal
#   ./start.sh --sudo    → inicia a API com sudo (captura real de pacotes)
################################################################################

# ── Cores ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m'
BOLD='\033[1m'

# ── Caminhos ─────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
API_LOG="$LOG_DIR/api_service.log"
WEB_LOG="$LOG_DIR/web_service.log"

# ── PIDs dos processos ────────────────────────────────────────────────────────
API_PID=""
WEB_PID=""
TAIL_API_PID=""
TAIL_WEB_PID=""

# ── Flags de execução ─────────────────────────────────────────────────────────
SHOW_LOGS=false
USE_SUDO=false

for arg in "$@"; do
    case "$arg" in
        --logs|-l) SHOW_LOGS=true ;;
        --sudo|-s) USE_SUDO=true ;;
    esac
done

# ── Funções de log ────────────────────────────────────────────────────────────
print_success() { echo -e "${GREEN}[✓]${NC} $1"; }
print_error()   { echo -e "${RED}[✗]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[!]${NC} $1"; }
print_info()    { echo -e "${CYAN}[i]${NC} $1"; }
print_sep()     { echo -e "${BLUE}────────────────────────────────────────────────────────${NC}"; }

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

# ── Limpeza ao sair ───────────────────────────────────────────────────────────
cleanup() {
    echo ""
    print_warning "Encerrando serviços Horus-CDS..."

    [ -n "$TAIL_API_PID" ] && kill "$TAIL_API_PID" 2>/dev/null
    [ -n "$TAIL_WEB_PID" ] && kill "$TAIL_WEB_PID" 2>/dev/null
    [ -n "$API_PID" ]      && kill "$API_PID"      2>/dev/null && print_success "API encerrada (PID $API_PID)"
    [ -n "$WEB_PID" ]      && kill "$WEB_PID"      2>/dev/null && print_success "Web Dashboard encerrado (PID $WEB_PID)"

    wait 2>/dev/null
    echo ""
    print_info "Horus-CDS parado. Até logo!"
    exit 0
}

trap cleanup SIGINT SIGTERM

# ── Detecção do Python ────────────────────────────────────────────────────────
# PYTHON_CMD  → usado para processos normais (herda o PATH do venv)
# PYTHON_ABS  → caminho absoluto; necessário para sudo, que limpa o PATH
detect_python() {
    if [ -f "$SCRIPT_DIR/venv-Horus/bin/activate" ]; then
        source "$SCRIPT_DIR/venv-Horus/bin/activate"
        PYTHON_CMD="python"
        PYTHON_ABS="$SCRIPT_DIR/venv-Horus/bin/python"
        print_success "Ambiente virtual venv-Horus ativado"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_CMD="python3"
        PYTHON_ABS="$(command -v python3)"
        print_warning "Usando Python do sistema (venv-Horus não encontrado)"
    else
        print_error "Python 3 não encontrado. Instale Python 3.9+ para continuar."
        exit 1
    fi
}

# ── Verificar build do dashboard ──────────────────────────────────────────────
ensure_web_build() {
    local dist="$SCRIPT_DIR/root/web/dist/index.html"

    if [ -f "$dist" ]; then
        return 0
    fi

    print_warning "Build do dashboard React não encontrado (root/web/dist/)"

    if ! command -v npm >/dev/null 2>&1; then
        print_error "npm não encontrado. Instale Node.js e execute:"
        echo -e "  ${YELLOW}cd root/web && npm install && npm run build${NC}"
        exit 1
    fi

    read -p "$(echo -e "${CYAN}[i]${NC} Construir o dashboard agora? (S/n): ")" build_confirm
    if [[ "$build_confirm" =~ ^[Nn]$ ]]; then
        print_error "Dashboard não construído. Execute: cd root/web && npm run build"
        exit 1
    fi

    print_info "Instalando dependências npm..."
    (cd "$SCRIPT_DIR/root/web" && npm install) || {
        print_error "Falha ao instalar dependências npm"
        exit 1
    }

    print_info "Construindo bundle de produção..."
    (cd "$SCRIPT_DIR/root/web" && npm run build) || {
        print_error "Falha no build. Verifique os erros acima."
        exit 1
    }

    print_success "Dashboard construído com sucesso"
}

# ── Verificar se porta já está em uso ─────────────────────────────────────────
check_port() {
    local port=$1
    if ss -tlnp 2>/dev/null | grep -q ":${port} " || \
       lsof -ti ":${port}" >/dev/null 2>&1; then
        return 1  # porta ocupada
    fi
    return 0
}

# ── INÍCIO ────────────────────────────────────────────────────────────────────
clear
print_banner

mkdir -p "$LOG_DIR"

print_info "Verificando ambiente..."
detect_python

print_info "Verificando dashboard web..."
ensure_web_build

echo ""
print_sep

# Checar portas
if ! check_port 5000; then
    print_warning "Porta 5000 já está em uso. A API pode não iniciar corretamente."
fi
if ! check_port 5001; then
    print_warning "Porta 5001 já está em uso. O Web Dashboard pode não iniciar corretamente."
fi

# ── Iniciar API ───────────────────────────────────────────────────────────────
print_info "Iniciando API na porta ${BOLD}5000${NC}..."

if [ "$USE_SUDO" = true ]; then
    print_warning "Modo --sudo ativo: captura real de pacotes habilitada"
    echo ""
    print_info "Autenticando com sudo (necessário para captura real de pacotes)..."
    # Autentica no foreground — prompt visível no terminal — antes de subir em background
    if ! sudo -v </dev/tty; then
        print_error "Falha na autenticação sudo. Abortando."
        exit 1
    fi
    print_success "Autenticação sudo concluída"
    echo ""
    # Usa o caminho absoluto do Python do venv — sudo limpa o PATH e ignoraria o venv
    sudo "$PYTHON_ABS" "$SCRIPT_DIR/root/API/app.py" > "$API_LOG" 2>&1 &
else
    "$PYTHON_CMD" "$SCRIPT_DIR/root/API/app.py" > "$API_LOG" 2>&1 &
fi
API_PID=$!

# Aguarda inicialização
sleep 2

if ! kill -0 "$API_PID" 2>/dev/null; then
    print_error "Falha ao iniciar a API. Verifique o log:"
    echo -e "  ${YELLOW}cat $API_LOG${NC}"
    exit 1
fi
print_success "API iniciada (PID: ${BOLD}$API_PID${NC})"

# ── Iniciar Web Dashboard ─────────────────────────────────────────────────────
print_info "Iniciando Web Dashboard na porta ${BOLD}5001${NC}..."

"$PYTHON_CMD" "$SCRIPT_DIR/root/web/run_web.py" > "$WEB_LOG" 2>&1 &
WEB_PID=$!

sleep 1

if ! kill -0 "$WEB_PID" 2>/dev/null; then
    print_error "Falha ao iniciar o Web Dashboard. Verifique o log:"
    echo -e "  ${YELLOW}cat $WEB_LOG${NC}"
    kill "$API_PID" 2>/dev/null
    exit 1
fi
print_success "Web Dashboard iniciado (PID: ${BOLD}$WEB_PID${NC})"

# ── Status final ──────────────────────────────────────────────────────────────
echo ""
print_sep
echo ""
echo -e "  ${BOLD}${GREEN}▶ Horus-CDS está em execução${NC}"
echo ""
echo -e "  ${GREEN}API:${NC}       http://localhost:5000"
echo -e "  ${GREEN}Dashboard:${NC} http://localhost:5001"
echo ""

if [ "$USE_SUDO" = true ]; then
    echo -e "  ${YELLOW}Modo:${NC}      Captura Real (sudo)"
else
    echo -e "  ${CYAN}Modo:${NC}      Simulação (use --sudo para captura real)"
fi

echo ""
echo -e "  ${BLUE}Logs:${NC}"
echo -e "    API: $API_LOG"
echo -e "    Web: $WEB_LOG"
echo ""
print_sep

if [ "$SHOW_LOGS" = true ]; then
    echo ""
    print_info "Exibindo logs ao vivo (Ctrl+C para encerrar)..."
    echo ""

    tail -n 5 -f "$API_LOG" 2>/dev/null | awk '{print "\033[0;34m[API]\033[0m " $0; fflush()}' &
    TAIL_API_PID=$!

    tail -n 5 -f "$WEB_LOG" 2>/dev/null | awk '{print "\033[0;32m[WEB]\033[0m " $0; fflush()}' &
    TAIL_WEB_PID=$!
else
    echo ""
    echo -e "  ${YELLOW}Pressione Ctrl+C para encerrar os serviços${NC}"
    echo -e "  (Use ${CYAN}./start.sh --logs${NC} para ver os logs ao vivo)"
    echo ""
fi

# ── Monitorar processos ───────────────────────────────────────────────────────
while true; do
    sleep 5

    if ! kill -0 "$API_PID" 2>/dev/null; then
        echo ""
        print_error "A API encerrou inesperadamente!"
        echo -e "  Últimas linhas do log:"
        tail -n 10 "$API_LOG" | sed 's/^/    /'
        break
    fi

    if ! kill -0 "$WEB_PID" 2>/dev/null; then
        echo ""
        print_error "O Web Dashboard encerrou inesperadamente!"
        echo -e "  Últimas linhas do log:"
        tail -n 10 "$WEB_LOG" | sed 's/^/    /'
        break
    fi
done

cleanup
