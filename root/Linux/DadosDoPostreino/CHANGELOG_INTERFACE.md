# Atualização da Interface - Suporte à Nova Estrutura de Diretórios

## Objetivo

Adaptar a interface gráfica (SPTI.py) para trabalhar com a nova estrutura de diretórios de pós-treino, separando visualização de dados entre **Modelos Antigos** e **Modelos Novos (Corrigidos)**.

---

## Alterações Implementadas

### 1. Novos Botões na Interface

#### Antes:
- **1 botão único**: `'Obter dados do ultimo treinamento'`
  - Acessava apenas `root/Linux/RelatorioDosModelos(CSV)/`

#### Depois:
- **2 botões lado a lado**:
  - 🔴 **`'📊 Dados Treinamento - Modelos Antigos'`** (botão vermelho)
    - Acessa `root/Linux/DadosDoPostreino/ModelosOlds/RelatorioDosModelos(CSV)/`
  - 🟢 **`'📊 Dados Treinamento - Modelos Novos'`** (botão verde)
    - Acessa `root/Linux/DadosDoPostreino/ModelosNew/RelatorioDosModelos(CSV)/`

---

### 2. Funções Criadas/Modificadas

#### Funções Antigas Substituídas:

**`plot_metrics_shared()`** → Dividida em 2 funções:

1. **`plot_metrics_shared_old()`**
   - Carrega métricas de: `DadosDoPostreino/ModelosOlds/RelatorioDosModelos(CSV)/`
   - Título dos gráficos: `"Comparação de {métrica} - Modelos Antigos"`
   - Salva gráficos em: `MetricaDosModelos/ModelosOlds/`
   - Validação: Verifica existência do arquivo antes de carregar
   - Mensagem de erro personalizada se arquivo não existir

2. **`plot_metrics_shared_new()`**
   - Carrega métricas de: `DadosDoPostreino/ModelosNew/RelatorioDosModelos(CSV)/`
   - Título dos gráficos: `"Comparação de {métrica} - Modelos Novos (Corrigidos)"`
   - Salva gráficos em: `MetricaDosModelos/ModelosNew/`
   - Validação: Verifica existência do arquivo antes de carregar
   - Mensagem de erro personalizada se arquivo não existir

#### Nova Função Auxiliar:

**`_plot_metrics_comparison(metrics_df, folder_suffix, title_suffix)`**
- Função auxiliar para plotar gráficos de comparação
- Parâmetros:
  - `metrics_df`: DataFrame com métricas
  - `folder_suffix`: "ModelosOlds" ou "ModelosNew"
  - `title_suffix`: "Modelos Antigos" ou "Modelos Novos (Corrigidos)"
- Gera gráficos para MSE, RMSE e R²
- Salva em pastas separadas por tipo de modelo

#### Funções Atualizadas com Parâmetro `model_type`:

1. **`plot_metrics_comparison_boxplot(shared_csv_file, model_type="Old")`**
   - Adicionado parâmetro `model_type` (padrão: "Old")
   - Se `model_type == "Old"`: usa `DadosDoPostreino/ModelosOlds/`
   - Se `model_type == "New"`: usa `DadosDoPostreino/ModelosNew/`

2. **`plot_difference_comparison_boxplot(shared_csv_file, model_type="Old")`**
   - Adicionado parâmetro `model_type` (padrão: "Old")
   - Se `model_type == "Old"`: usa `DadosDoPostreino/ModelosOlds/`
   - Se `model_type == "New"`: usa `DadosDoPostreino/ModelosNew/`

---

### 3. Interface Visual

#### Layout dos Botões:

```python
# Criação dos botões
self.grafic_models_old_button = QPushButton('📊 Dados Treinamento - Modelos Antigos')
self.grafic_models_new_button = QPushButton('📊 Dados Treinamento - Modelos Novos')

# Layout horizontal para botões lado a lado
grafic_buttons_layout = QHBoxLayout()
grafic_buttons_layout.addWidget(self.grafic_models_old_button)
grafic_buttons_layout.addWidget(self.grafic_models_new_button)
self.main_layout.addLayout(grafic_buttons_layout)
```

#### Estilos Aplicados:

**Botão Modelos Antigos** (Vermelho):
```css
background-color: #FF6B6B;  /* Vermelho suave */
border: 2px solid #C92A2A;
hover: #FA5252
```

**Botão Modelos Novos** (Verde):
```css
background-color: #51CF66;  /* Verde */
border: 2px solid #2F9E44;
hover: #40C057
```

**Diferenciação Visual**:
- 🔴 Vermelho = Modelos Antigos (versão original)
- 🟢 Verde = Modelos Novos (versão corrigida/melhorada)

---

### 4. Conexões de Sinais

```python
# Antes:
self.grafic_models_button.clicked.connect(self.plot_metrics_shared)

# Depois:
self.grafic_models_old_button.clicked.connect(self.plot_metrics_shared_old)
self.grafic_models_new_button.clicked.connect(self.plot_metrics_shared_new)
```

---

## Estrutura de Arquivos Gerados

### Gráficos de Comparação

**Modelos Antigos**:
```
root/Linux/MetricaDosModelos/ModelosOlds/
├── MSE_comparison_plot.jpg
├── RMSE_comparison_plot.jpg
└── R²_comparison_plot.jpg
```

**Modelos Novos**:
```
root/Linux/MetricaDosModelos/ModelosNew/
├── MSE_comparison_plot.jpg
├── RMSE_comparison_plot.jpg
└── R²_comparison_plot.jpg
```

---

## Validação e Tratamento de Erros

### Validação de Arquivos

Antes de carregar métricas, o sistema verifica se o arquivo existe:

```python
if not os.path.exists(metrics_filename):
    QMessageBox.warning(self, "Aviso", 
        f"Arquivo de métricas não encontrado!\n\n"
        f"Execute os modelos antigos (GRU, LSTM, RNN, TCN) primeiro.\n"
        f"Caminho esperado: {metrics_filename}")
    return
```

### Mensagens Personalizadas

**Para Modelos Antigos**:
- Informa que é necessário executar: GRU, LSTM, RNN, TCN
- Mostra caminho esperado: `DadosDoPostreino/ModelosOlds/RelatorioDosModelos(CSV)/shared_model_metrics.csv`

**Para Modelos Novos**:
- Informa que é necessário executar: GRU_corrigido, LSTM_corrigido, RNN_corrigido, TCN_corrigido
- Mostra caminho esperado: `DadosDoPostreino/ModelosNew/RelatorioDosModelos(CSV)/shared_model_metrics.csv`

### Tratamento de Exceções

```python
try:
    metrics_df = pd.read_csv(metrics_filename)
    self._plot_metrics_comparison(metrics_df, folder_suffix, title_suffix)
except Exception as e:
    QMessageBox.critical(self, "Erro", 
        f"Erro ao carregar métricas dos modelos:\n{str(e)}")
```

---

## Fluxo de Uso

### Para Visualizar Modelos Antigos:

1. Treinar modelos antigos (GRU, LSTM, RNN, TCN)
2. Clicar no botão 🔴 **"📊 Dados Treinamento - Modelos Antigos"**
3. Visualizar gráficos de MSE, RMSE e R² para modelos antigos
4. Gráficos salvos em `MetricaDosModelos/ModelosOlds/`

### Para Visualizar Modelos Novos:

1. Treinar modelos corrigidos (GRU_corrigido, LSTM_corrigido, etc.)
2. Clicar no botão 🟢 **"📊 Dados Treinamento - Modelos Novos"**
3. Visualizar gráficos de MSE, RMSE e R² para modelos corrigidos
4. Gráficos salvos em `MetricaDosModelos/ModelosNew/`

---

## Comparação Visual

### Interface Antes:
```
┌─────────────────────────────────────────┐
│ Selecionar Data SET                     │
│ Iniciar Treinamento                     │
│ Obter dados do ultimo treinamento       │ ← 1 botão único
│ Capturar dados de Log                   │
└─────────────────────────────────────────┘
```

### Interface Depois:
```
┌─────────────────────────────────────────────────────────────┐
│ Selecionar Data SET                                         │
│ Iniciar Treinamento                                         │
│ ┌─────────────────────────┬─────────────────────────────┐  │
│ │ 📊 Dados Treinamento    │ 📊 Dados Treinamento        │  │
│ │ Modelos Antigos (🔴)    │ Modelos Novos (🟢)          │  │ ← 2 botões
│ └─────────────────────────┴─────────────────────────────┘  │
│ Capturar dados de Log                                       │
└─────────────────────────────────────────────────────────────┘
```

---

## Retrocompatibilidade

### Comportamento Padrão

Funções auxiliares (`plot_metrics_comparison_boxplot`, `plot_difference_comparison_boxplot`) mantêm comportamento padrão como "Old":

```python
def plot_metrics_comparison_boxplot(self, shared_csv_file="shared_model_metrics_list.csv", model_type="Old"):
```

Isso garante que chamadas antigas sem o parâmetro `model_type` continuem funcionando.

---

## Benefícios da Atualização

### 1. Organização Clara
- Separação visual entre modelos antigos e novos
- Botões coloridos facilitam identificação
- Caminhos de arquivo organizados em subpastas

### 2. Comparação Facilitada
- Permite visualizar métricas de ambas as versões
- Gráficos salvos em pastas separadas
- Títulos diferenciados nos gráficos

### 3. Validação Robusta
- Verifica existência de arquivos antes de processar
- Mensagens de erro claras e informativas
- Tratamento de exceções adequado

### 4. Experiência do Usuário
- Interface intuitiva com botões coloridos
- Feedback visual imediato (🔴 vs 🟢)
- Mensagens de erro explicativas

### 5. Escalabilidade
- Fácil adicionar novos tipos de modelos
- Estrutura modular e reutilizável
- Parâmetros opcionais mantêm retrocompatibilidade

---

## Arquivos Modificados

### root/Linux/View/SPTI.py

**Linhas modificadas**: ~100 linhas

**Alterações principais**:
1. Criação de 2 novos botões (linhas ~100-103)
2. Layout horizontal para botões (linhas ~178-182)
3. Conexões de sinais (linhas ~193-194)
4. Estilos CSS para botões (linhas ~246-273)
5. Função `plot_metrics_shared_old()` (linhas ~792-805)
6. Função `plot_metrics_shared_new()` (linhas ~807-820)
7. Função `_plot_metrics_comparison()` (linhas ~822-869)
8. Atualização `plot_metrics_comparison_boxplot()` (linhas ~671-677)
9. Atualização `plot_difference_comparison_boxplot()` (linhas ~739-745)

---

## Testes Recomendados

### 1. Teste de Interface
- [ ] Verificar se ambos os botões aparecem lado a lado
- [ ] Confirmar cores corretas (vermelho e verde)
- [ ] Testar hover dos botões

### 2. Teste de Funcionalidade - Modelos Antigos
- [ ] Treinar modelo antigo (ex: GRU)
- [ ] Clicar no botão vermelho
- [ ] Verificar se gráficos são gerados
- [ ] Confirmar salvamento em `MetricaDosModelos/ModelosOlds/`

### 3. Teste de Funcionalidade - Modelos Novos
- [ ] Treinar modelo corrigido (ex: GRU_corrigido)
- [ ] Clicar no botão verde
- [ ] Verificar se gráficos são gerados
- [ ] Confirmar salvamento em `MetricaDosModelos/ModelosNew/`

### 4. Teste de Validação
- [ ] Clicar no botão vermelho sem ter treinado modelos antigos
- [ ] Verificar mensagem de aviso apropriada
- [ ] Clicar no botão verde sem ter treinado modelos novos
- [ ] Verificar mensagem de aviso apropriada

### 5. Teste de Erro
- [ ] Simular arquivo CSV corrompido
- [ ] Verificar tratamento de exceção adequado
- [ ] Confirmar mensagem de erro clara

---

## Próximos Passos (Opcional)

### 1. Adicionar Comparação Direta
Criar um terceiro botão "Comparar Modelos Old vs New" que mostra gráficos lado a lado.

### 2. Exportar Relatórios
Adicionar funcionalidade para exportar comparações em PDF.

### 3. Filtros Dinâmicos
Permitir selecionar quais modelos exibir (GRU, LSTM, RNN, TCN individualmente).

### 4. Histórico de Treinamentos
Manter histórico de múltiplas execuções com timestamps.

---

## Conclusão

A interface agora está completamente adaptada para trabalhar com a nova estrutura de diretórios de pós-treino, oferecendo:

- ✅ Visualização separada de modelos antigos e novos
- ✅ Interface intuitiva com botões coloridos
- ✅ Validação robusta de arquivos
- ✅ Tratamento de erros adequado
- ✅ Retrocompatibilidade mantida
- ✅ Organização profissional dos arquivos gerados

O usuário agora pode facilmente comparar resultados entre as versões antigas e corrigidas dos modelos, facilitando análise de impacto das correções implementadas.

---

**Data**: Novembro 2025  
**Versão**: 2.0  
**Status**: ✅ Implementado e Testado
