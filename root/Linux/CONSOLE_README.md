# Console de Treinamento Integrado

## Descrição

Widget de console integrado à interface PyQt5 que exibe em tempo real todas as mensagens de log durante o treinamento dos modelos.

## Arquivos Criados/Modificados

### 1. `/root/Linux/View/TrainingConsole.py` (NOVO)

Widget customizado que implementa:
- Console com syntax highlighting por tipo de mensagem
- Auto-scroll para acompanhar logs em tempo real
- Botão de limpar console
- Captura de stdout/stderr

**Cores por tipo de mensagem:**
- ERRO/ERROR: Vermelho (#ff6b6b)
- AVISO/WARNING: Amarelo (#ffd93d)
- SUCESSO/SUCCESS: Verde (#6bcf7f)
- GPU: Ciano (#4ecdc4)
- Informações de treino/teste: Verde claro (#95e1d3)
- Barras de progresso Keras: Azul claro (#a8dadc)
- Mensagens normais: Cinza (#d4d4d4)

### 2. `/root/Linux/View/SPTI.py` (MODIFICADO)

**Alterações:**
- Import do `TrainingConsole` e `ConsoleCapture`
- Instância do console adicionada à interface (após barra de progresso)
- Console inicialmente oculto
- Ativação automática ao iniciar treinamento
- Captura de stdout/stderr redirecionada para o console
- Restauração automática quando treinamento atinge 100%

**Métodos modificados:**
- `__init__`: Adicionado `self.console_capture = None`
- `initUI`: Adicionado widget `self.training_console`
- `start_training`: Ativa console e redireciona stdout/stderr
- `update_progress`: Restaura stdout/stderr quando chega a 100%

## Como Funciona

1. Usuário seleciona modelo e clica em "Iniciar Treinamento"
2. Console aparece automaticamente
3. Todos os prints dos scripts de treinamento são capturados
4. Mensagens aparecem em tempo real com cores
5. Quando treinamento finaliza (100%), stdout/stderr são restaurados

## Mensagens Capturadas

O console captura automaticamente:
- Mensagens de preparação de dados
- Avisos do TensorFlow/Keras
- Barras de progresso do treinamento
- Informações de GPU
- Métricas de treino (loss, mse, etc.)
- Mensagens de erro
- Confirmações de sucesso

## Exemplo de Saída

```
=== Iniciando treinamento ===

Features temporais adicionadas ao conjunto de Treino: 17160 amostras restantes
Features temporais adicionadas ao conjunto de Teste: 4288 amostras restantes
Aplicando normalização (StandardScaler apenas no treino)...
Dados preparados com sucesso!
  Treino: 17160 amostras
  Teste: 4288 amostras

Realizando busca de hiperparâmetros...
Fitting 3 folds for each of 5 candidates, totalling 15 fits
358/358 ━━━━━━━━━━━━━━━━━━━━ 5s 6ms/step - loss: 0.3400 - mse: 0.2220
179/179 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step

=== Treinamento finalizado ===
```

## Teste

Para testar o console isoladamente:

```bash
cd /home/wesleiferreira/Documentos/GitHub/APP2/Horus-CDS/root/Linux
source ../../venv-Horus/bin/activate
python test_console.py
```

## Notas Técnicas

- O console usa QTextEdit com fonte monoespaçada (Courier New)
- Background escuro (#1e1e1e) estilo terminal
- Altura ajustável (min: 250px, max: 400px)
- Não bloqueia a thread principal (mensagens via signals)
- Mantém stdout original funcionando em paralelo
