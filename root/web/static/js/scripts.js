let monitoring = false;  // Flag para ativar/desativar monitoramento
let intervalId;  // Armazena o ID do intervalo de monitoramento
let tipoGraficoLogs = 'bar'; 

let currentPage = 1;  // Página inicial
const rowsPerPage = 10;  // Número de linhas por página


function showLoadingSpinner() {
    document.getElementById('loadingSpinner').style.display = 'flex';
}

function hideLoadingSpinner() {
    document.getElementById('loadingSpinner').style.display = 'none';
}


// Função para exibir a tabela com paginação
function atualizarTabelaPaginada(data) {
    const tabela = document.getElementById('dadosTabela');
    tabela.innerHTML = ''; // Limpa a tabela

    const logs = data.packet_logs.logs;
    const start = (currentPage - 1) * rowsPerPage;
    const end = start + rowsPerPage;
    const paginatedLogs = logs.slice(start, end);

    paginatedLogs.forEach(log => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${log.timestamp}</td>
            <td>${log.source_ip}</td>
            <td>${log.destination_ip}</td>
            <td>${log.tipo}</td>
        `;
        tabela.appendChild(tr);
    });

    // Atualizar o número da página
    document.getElementById('pageNumber').textContent = `Página ${currentPage}`;
}

// Função para navegar entre as páginas
document.getElementById('nextPage').addEventListener('click', () => {
    currentPage++;
    fetchData();
});

document.getElementById('prevPage').addEventListener('click', () => {
    if (currentPage > 1) {
        currentPage--;
        fetchData();
    }
});

const syncBtn = document.getElementById("syncBtn");
const dataContainer = document.getElementById("dataContainer");

syncBtn.addEventListener("click", async () => {
    fetchData();
});

// Função para buscar dados da API
async function fetchData() {
    const serverInfo = document.getElementById('serverInfo').value;
    const filter = document.getElementById('filter').value;

    if (!serverInfo) {
        alert('Digite o IP e a Porta do servidor!');
        return;
    }

    const [ip, port] = serverInfo.split(':');
    if (!ip || !port) {
        alert('Formato inválido! Digite no formato IP:PORTA.');
        return;
    }

    const url = `http://${ip}:${port}/gerar_dados?filter=${filter}`;

    showLoadingSpinner();
    syncBtn.disabled = true;
    syncBtn.innerHTML = `<i class="fas fa-spinner fa-spin"></i> Atualizando...`;

    try {
        const response = await fetch(url);
        if (!response.ok) {
            const errorMessage = `Erro ${response.status}: ${response.statusText}`;
            throw new Error(errorMessage);
        }

        const data = await response.json();
        atualizarGraficos(data);
        atualizarTabelaPaginada(data);
        atualizarTotais(data); // Atualizar totais
    } catch (error) {
        console.error(error);
        alert(`Erro ao buscar os dados da API: ${error.message}`);
    } finally {
        hideLoadingSpinner();
        syncBtn.disabled = false;
        syncBtn.innerHTML = `<i class="fas fa-sync-alt"></i> Atualizar`;
    }
}

// Função para iniciar/pausar monitoramento em tempo real
document.getElementById('toggleMonitor').addEventListener('click', () => {
    monitoring = !monitoring; // Alterna o estado de monitoramento
    const button = document.getElementById('toggleMonitor');
    const icon = button.querySelector('i'); // Seleciona o ícone dentro do botão

    if (monitoring) {
        icon.classList.remove('fa-play');
        icon.classList.add('fa-pause');
        button.querySelector('span').textContent = ' Parar';
        intervalId = setInterval(fetchData, 10000);  // Atualiza a cada 10 segundos
    } else {
        icon.classList.remove('fa-pause');
        icon.classList.add('fa-play');
        button.querySelector('span').textContent = ' Iniciar';
        clearInterval(intervalId);
    }
});
let previousData = {
    ataques: 0,
    permitidas: 0,
    inconclusivos: 0,
};

function atualizarTotais(data) {
    // Certifique-se de que os dados existem
    const totalAtaques = data.packet_logs?.ataques_detectados ?? 0;
    const totalPermitidas = data.packet_logs?.requisicoes_permitidas ?? 0;
    const totalInconclusivos = data.packet_logs?.inconclusivos ?? 0;

    // Apenas atualize os valores e setas se os dados forem diferentes
    if (
        totalAtaques !== previousData.ataques ||
        totalPermitidas !== previousData.permitidas ||
        totalInconclusivos !== previousData.inconclusivos
    ) {
        // Atualizar os elementos com os valores e ícones de tendência
        document.getElementById('totalAtaques').innerHTML = `
            ${totalAtaques} Ataques Detectados 
            <span style="
                display: inline-block; 
                width: 24px; 
                height: 24px; 
                background-color: ${totalAtaques > previousData.ataques ? '#ffcccc' : '#ccffcc'}; 
                color: black; 
                border-radius: 50%; 
                text-align: center; 
                line-height: 24px; 
                font-size: 1.2em;
            ">
                ${totalAtaques > previousData.ataques ? '↑' : '↓'}
            </span>
        `;
        document.getElementById('totalPermitidas').innerHTML = `
            ${totalPermitidas} Requisições Permitidas 
            <span style="
                display: inline-block; 
                width: 24px; 
                height: 24px; 
                background-color: ${totalPermitidas > previousData.permitidas ? '#ccffcc' : '#ffcccc'}; 
                color: black; 
                border-radius: 50%; 
                text-align: center; 
                line-height: 24px; 
                font-size: 1.2em;
            ">
                ${totalPermitidas > previousData.permitidas ? '↑' : '↓'}
            </span>
        `;
        document.getElementById('totalInconclusivos').innerHTML = `
            ${totalInconclusivos} Inconclusivos 
            <span style="
                display: inline-block; 
                width: 24px; 
                height: 24px; 
                background-color: ${totalInconclusivos > previousData.inconclusivos ? '#ffe5b4' : '#b3e6ff'}; 
                color: black; 
                border-radius: 50%; 
                text-align: center; 
                line-height: 24px; 
                font-size: 1.2em;
            ">
                ${totalInconclusivos > previousData.inconclusivos ? '↑' : '↓'}
            </span>
        `;



        // Atualizar os dados anteriores
        previousData.ataques = totalAtaques;
        previousData.permitidas = totalPermitidas;
        previousData.inconclusivos = totalInconclusivos;
    }
}



let packetChartInstance;  // Armazena a instância do gráfico de logs de pacotes
let predictionChartInstance;  // Armazena a instância do gráfico de predições
let predictionChartTimeInstance;  // Armazena a instância do gráfico de predições desnormalizadas

const maxPoints = 50;
// Função para atualizar os gráficos
function atualizarGraficos(data) {
    // Atualizando o gráfico de logs de rede
    const packetData = {
        labels: ['Ataques Detectados', 'Requisições Permitidas', 'Inconclusivos'],
        datasets: [{
            label: 'Logs de Rede',
            data: [
                data.packet_logs.ataques_detectados,
                data.packet_logs.requisicoes_permitidas,
                data.packet_logs.inconclusivos
            ],
            backgroundColor: ['red', 'green', 'gray'],
        }]
    };

    // Atualizando o gráfico de logs de rede
    if (packetChartInstance) {
        packetChartInstance.destroy();
    }

    const ctxPacket = document.getElementById('packetChart').getContext('2d');
    packetChartInstance = new Chart(ctxPacket, {
        type: tipoGraficoLogs,
        data: packetData,
        options: {
            scales: {
                y: { beginAtZero: true }
            }
        }
    });

    // Limitar o número de pontos exibidos no gráfico de predição normalizada
    const normalizedLabels = data.predictions_log.resultados
        .map((_, index) => `Pacote ${index + 1}`)
        .slice(-maxPoints);  // Mantém apenas os últimos maxPoints
    const normalizedPredictions = data.predictions_log.normalized_predictions.slice(-maxPoints);

    // Criando o gráfico para predição normalizada
    const normalizedData = {
        labels: normalizedLabels,
        datasets: [{
            label: 'Predição Normalizada',
            data: normalizedPredictions,
            backgroundColor: 'blue',
            fill: false,
            borderColor: 'blue',
            type: 'line'
        }]
    };

    // Verificar e destruir o gráfico de predições, se ele já existir
    if (predictionChartInstance) {
        predictionChartInstance.destroy();
    }

    const ctxPredictionNormalized = document.getElementById('predictionChartNormalized').getContext('2d');
    predictionChartInstance = new Chart(ctxPredictionNormalized, {
        type: 'line',
        data: normalizedData,
        options: {
            scales: {
                y: { beginAtZero: true }
            }
        }
    });

    // Limitar o número de pontos exibidos no gráfico de tempo desnormalizado
    const timeLabels = data.predictions_log.resultados
        .map((_, index) => `Pacote ${index + 1}`)
        .slice(-maxPoints);  // Mantém apenas os últimos maxPoints
    const desnormalizedPredictions = data.predictions_log.desnormalized_predictions.slice(-maxPoints);

    // Criando o gráfico para tempo desnormalizado
    const timeData = {
        labels: timeLabels,
        datasets: [{
            label: 'Tempo Desnormalizado',
            data: desnormalizedPredictions,
            backgroundColor: 'green',
            fill: false,
            borderColor: 'green',
            type: 'line'
        }]
    };

    if (predictionChartTimeInstance) {
        predictionChartTimeInstance.destroy();
    }

    const ctxPredictionTime = document.getElementById('predictionChartTime').getContext('2d');
    predictionChartTimeInstance = new Chart(ctxPredictionTime, {
        type: 'line',
        data: timeData,
        options: {
            scales: {
                y: { beginAtZero: true }
            }
        }
    });
}


// Função para buscar os dados imediatamente ao mudar o filtro
document.getElementById('filter').addEventListener('change', fetchData);

// Função para salvar as configurações
function salvarConfiguracoes() {
    tipoGraficoLogs = document.getElementById('chartType').value;
    fetchData();  // Atualizar os gráficos com o novo tipo
}

// Função para atualizar a tabela com os logs
function atualizarTabela(data) {
    const tabela = document.getElementById('dadosTabela');
    tabela.innerHTML = ''; // Limpa a tabela

    // Adiciona as linhas com os dados dos logs
    data.packet_logs.logs.forEach(log => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${log.timestamp}</td>
            <td>${log.source_ip}</td>
            <td>${log.destination_ip}</td>
            <td>${log.tipo}</td>
        `;
        tabela.appendChild(tr);
    });
}

// Chamar a função fetchData quando a página carregar
window.onload = fetchData;