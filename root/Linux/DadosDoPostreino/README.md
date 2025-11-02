# Dados de Pós-Treino - Horus-CDS

Esta pasta contém todos os dados gerados após o treinamento dos modelos de aprendizado profundo.

## Estrutura de Diretórios

```
DadosDoPostreino/
├── ModelosOlds/                              # Resultados dos modelos originais
│   ├── ModelosComplilados/                   # Modelos treinados (.h5 e .keras)
│   ├── RelatorioDosModelos(CSV)/            # Relatórios em CSV
│   ├── RelatorioDosModelos(PDF)/            # Relatórios em PDF
│   ├── RelatorioDosModelos(TXT)/            # Relatórios em TXT
│   ├── PrevisoesDosModelos/                 # Gráficos de previsões
│   ├── PrevisoesDosModelos(Block)/          # Gráficos de previsões em blocos
│   ├── PrevisoesDosModelos(BoxPlot)/        # Gráficos de previsões boxplot
│   ├── PrevisaoDosModelos(Diferenca)/       # Gráficos de diferenças
│   ├── MetricaDosModelos/                   # Gráficos de métricas
│   ├── ComparacaoMetricas(BoxPlot)/         # Comparação de métricas
│   └── MatrizConfusao/                      # Matrizes de confusão (se aplicável)
│
└── ModelosNew/                               # Resultados dos modelos corrigidos
    ├── ModelosCorrigidos/                   # Modelos treinados (.keras)
    ├── RelatorioDosModelos(CSV)/            # Relatórios em CSV
    ├── RelatorioDosModelos(PDF)/            # Relatórios em PDF
    ├── RelatorioDosModelos(TXT)/            # Relatórios em TXT
    ├── PrevisoesDosModelos/                 # Gráficos de previsões
    ├── PrevisoesDosModelos(Block)/          # Gráficos de previsões em blocos
    ├── PrevisoesDosModelos(BoxPlot)/        # Gráficos de previsões boxplot
    ├── PrevisaoDosModelos(Diferenca)/       # Gráficos de diferenças
    ├── MetricaDosModelos/                   # Gráficos de métricas
    ├── ComparacaoMetricas(BoxPlot)/         # Comparação de métricas
    └── MatrizConfusao/                      # Matrizes de confusão
```

---

## ModelosOlds

Contém os resultados dos **modelos originais** (versões antes das correções dos revisores):

### Modelos Incluídos:
- **GRU** (Gated Recurrent Unit)
- **LSTM** (Long Short-Term Memory)
- **RNN** (Recurrent Neural Network)
- **TCN** (Temporal Convolutional Network)

### Scripts de Treinamento:
- `ThreadTrain/TrainingThreadGRU.py`
- `ThreadTrain/TrainingThreadLSTM.py`
- `ThreadTrain/TrainingThreadRNN.py`
- `ThreadTrain/TrainingThreadTCN.py`

### Estrutura de Dados:

#### ModelosComplilados/
Modelos treinados salvos em formato Keras:
- `gru_model.h5`
- `lstm_model.h5`
- `rnn_model.h5`
- `tcn_model.keras`

#### RelatorioDosModelos(CSV)/
Contém arquivos CSV com resultados das previsões e métricas:
- `{modelo}_model_results.csv` - Dados: Real vs Previsto com timestamps
- `shared_model_metrics.csv` - Métricas consolidadas de todos os modelos
- `shared_model_metrics_list.csv` - Listas de MSE/RMSE por época
- `shared_model_difference_list.csv` - Diferenças entre previsões e valores reais

#### RelatorioDosModelos(PDF)/
Documentos PDF com visualizações e sumários:
- `{modelo}_model_summary.pdf` - Arquitetura e resumo do modelo
- `{modelo}_model_metrics.pdf` - Métricas de desempenho (MSE, RMSE, R²)

#### RelatorioDosModelos(TXT)/
Arquivos texto com métricas em formato legível:
- `{modelo}_model_metrics.txt` - MSE, RMSE e R² do modelo

#### PrevisoesDosModelos/
Gráficos de comparação entre valores reais e previstos:
- `{modelo}_prediction.jpg` - Gráfico de linha temporal

#### PrevisoesDosModelos(Block)/
Gráficos de previsões em formato de blocos:
- `{modelo}_prediction_block.jpg` - Visualização em blocos

#### PrevisoesDosModelos(BoxPlot)/
Gráficos de distribuição das previsões:
- `{modelo}_prediction_boxplot.jpg` - Boxplot das previsões

#### PrevisaoDosModelos(Diferenca)/
Gráficos de diferenças entre previsões e valores reais:
- `boxplot_difference_comparison.jpg` - Comparação de diferenças

#### MetricaDosModelos/
Gráficos de métricas individuais por modelo:
- `{modelo}_metrics.jpg` - Visualização de todas as métricas

#### ComparacaoMetricas(BoxPlot)/
Comparação visual de métricas entre modelos:
- `boxplot_metrics_comparison.jpg` - Comparação de métricas
- `boxplot_mse_progression.jpg` - Progressão do MSE

---

## ModelosNew

Contém os resultados dos **modelos corrigidos** (após implementação das sugestões dos revisores):

### Modelos Incluídos:
- **GRU_corrigido**
- **LSTM_corrigido**
- **RNN_corrigido**
- **TCN_corrigido**

### Scripts de Treinamento:
- `ThreadTrain/GRU_corrigido.py`
- `ThreadTrain/LSTM_corrigido.py`
- `ThreadTrain/RNN_corrigido.py`
- `ThreadTrain/TCN_corrigido.py`

### Correções Aplicadas:
1. **Divisão Temporal Adequada** - Sem vazamento de dados (data leakage)
2. **Features Calculadas Separadamente** - Normalização e features temporais corretas
3. **Balanceamento de Classes** - SMOTE e RandomUnderSampler
4. **Dataset Normalizado** - Uso de `dados_normalizados_smartgrid.csv`
5. **Ruído Gaussiano** - GaussianNoise e Dropout para regularização
6. **Batch Normalization** - Estabilização do treinamento

### Estrutura de Dados:

#### ModelosCorrigidos/
Modelos treinados salvos em formato Keras:
- `gru_model_corrigido.keras`
- `lstm_model_corrigido.keras`
- `rnn_model_corrigido.keras`
- `tcn_model_corrigido.keras`

#### RelatorioDosModelos(CSV)/
Arquivos CSV com resultados das previsões:
- Estrutura idêntica aos ModelosOlds

#### RelatorioDosModelos(PDF)/
Documentos PDF com análises:
- Estrutura idêntica aos ModelosOlds

#### RelatorioDosModelos(TXT)/
Métricas em texto:
- Estrutura idêntica aos ModelosOlds

#### PrevisoesDosModelos/
Gráficos de comparação entre valores reais e previstos:
- Estrutura idêntica aos ModelosOlds

#### PrevisoesDosModelos(Block)/
Gráficos de previsões em formato de blocos:
- Estrutura idêntica aos ModelosOlds

#### PrevisoesDosModelos(BoxPlot)/
Gráficos de distribuição das previsões:
- Estrutura idêntica aos ModelosOlds

#### PrevisaoDosModelos(Diferenca)/
Gráficos de diferenças entre previsões e valores reais:
- Estrutura idêntica aos ModelosOlds

#### MetricaDosModelos/
Gráficos de métricas individuais por modelo:
- Estrutura idêntica aos ModelosOlds

#### ComparacaoMetricas(BoxPlot)/
Comparação visual de métricas entre modelos:
- Estrutura idêntica aos ModelosOlds

---

## Comparação: Olds vs New

| Aspecto | ModelosOlds | ModelosNew |
|---------|-------------|------------|
| **Divisão de Dados** | Aleatória (train_test_split) | Temporal (split por data) |
| **Features** | Calculadas em todo dataset | Calculadas separadamente |
| **Balanceamento** | Sem balanceamento | SMOTE + RandomUnderSampler |
| **Normalização** | StandardScaler básico | Dataset pré-normalizado |
| **Regularização** | Dropout simples | GaussianNoise + BatchNorm + Dropout |
| **Augmentação** | Ruído aleatório | Ruído gaussiano controlado |
| **Relatórios** | Completos (CSV, PDF, TXT) | Completos (CSV, PDF, TXT) |
| **Gráficos** | Todos os tipos de visualização | Todos os tipos de visualização |
| **Interface** | Botão VERMELHO | Botão VERDE |

---

## Como Usar

### Treinar Modelos Originais (Olds)

```python
from ThreadTrain.TrainingThreadGRU import TrainingThreadGRU
import pandas as pd

# Carregar dados
data = pd.read_csv('DadosReais/dados_normalizados_smartgrid.csv')

# Treinar modelo
thread = TrainingThreadGRU(data)
thread.start()
# Resultados salvos automaticamente em DadosDoPostreino/ModelosOlds/
```

### Treinar Modelos Corrigidos (New)

```python
from ThreadTrain.GRU_corrigido import TrainingThreadGRUCorrected

# Treinar modelo corrigido
thread = TrainingThreadGRUCorrected(model_type="GRU")
thread.start()
# Modelo salvo automaticamente em DadosDoPostreino/ModelosNew/ModelosCorrigidos/
```

---

## Análise de Resultados

### Métricas Principais

- **MSE (Mean Squared Error)**: Erro quadrático médio
- **RMSE (Root Mean Squared Error)**: Raiz do erro quadrático médio
- **R² (Coefficient of Determination)**: Coeficiente de determinação (0 a 1, quanto maior melhor)

### Interpretação

- **MSE/RMSE baixos**: Modelo faz previsões próximas aos valores reais
- **R² próximo de 1**: Modelo explica bem a variabilidade dos dados
- **Difference**: Analise a distribuição dos erros (deve estar centrada em 0)

### Comparação entre Modelos

Consulte `shared_model_metrics.csv` em cada pasta para comparar:

```bash
# Ver métricas dos modelos originais
cat DadosDoPostreino/ModelosOlds/RelatorioDosModelos(CSV)/shared_model_metrics.csv

# Ver métricas dos modelos corrigidos (se disponível)
cat DadosDoPostreino/ModelosNew/RelatorioDosModelos(CSV)/shared_model_metrics.csv
```

---

## Troubleshooting

### Problema: Pasta vazia

**Solução**: Execute os scripts de treinamento correspondentes. As pastas são criadas automaticamente durante o primeiro treino.

### Problema: Modelos Olds não geram PDFs

**Solução**: Verifique se as dependências estão instaladas:
```bash
pip install reportlab fpdf pandas matplotlib seaborn
```

### Problema: Modelos New não salvam relatórios completos

**Observação**: Por design, os modelos corrigidos focam em salvar apenas os modelos treinados. Para relatórios completos, adapte o código para usar `RelatorioDosModelos` com `model_type="New"`.

---

## Manutenção

### Limpeza de Dados Antigos

Para limpar resultados antigos:

```bash
# Limpar ModelosOlds
rm -rf DadosDoPostreino/ModelosOlds/RelatorioDosModelos*

# Limpar ModelosNew
rm -rf DadosDoPostreino/ModelosNew/RelatorioDosModelos*
rm -rf DadosDoPostreino/ModelosNew/ModelosCorrigidos/*
```

### Backup de Resultados

Recomenda-se fazer backup regular:

```bash
# Compactar tudo
tar -czf backup_postreino_$(date +%Y%m%d).tar.gz DadosDoPostreino/

# Backup apenas dos modelos
tar -czf backup_modelos_$(date +%Y%m%d).tar.gz \
  DadosDoPostreino/ModelosNew/ModelosCorrigidos/ \
  ModelosComplilados/
```

---

## Referências

- **Documentação do Projeto**: [../../README.md](../../README.md)
- **Scripts de Treinamento**: [../ThreadTrain/](../ThreadTrain/)
- **Modelos Compilados**: [../ModelosComplilados/](../ModelosComplilados/)
- **Dados Originais**: [../DadosReais/](../DadosReais/)

---

**Última atualização**: Novembro 2025
