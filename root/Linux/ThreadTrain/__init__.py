# Import tolerante a falhas: os modelos Keras (V1-V4) precisam de TF, os
# modelos PyTorch (V5) precisam de torch. Se um backend nao estiver
# disponivel no ambiente atual (ex: V5 rodando em venv sem TF), nao
# bloqueamos os outros. Quem falhar fica fora de __all__ e quem tentar
# usar o thread inativo vai receber o erro real na hora do uso.

import importlib as _importlib

_MODULES = [
    'TrainingThreadGRU', 'TrainingThreadLSTM', 'TrainingThreadARIMA',
    'TrainingThreadTCN', 'TrainingThreadKNN', 'TrainingThreadRNN',
    'TrainingThreadRandomForest', 'TrainingThreadMLP',
    'GRU_corrigido', 'LSTM_corrigido', 'RNN_corrigido', 'TCN_corrigido',
    # V5 (PyTorch)
    'Transformer_PatchTST', 'TrainingThreadTransformer',
]

__all__ = []
_failed = []

for _name in _MODULES:
    try:
        _mod = _importlib.import_module(f'.{_name}', package=__name__)
        globals()[_name] = _mod
        __all__.append(_name)
    except Exception as _err:
        _failed.append((_name, type(_err).__name__, str(_err)[:120]))

if _failed:
    import sys as _sys
    _sys.stderr.write(
        "[ThreadTrain] Avisos de carregamento (modulos pulados):\n"
    )
    for _n, _et, _em in _failed:
        _sys.stderr.write(f"  - {_n}: {_et}: {_em}\n")
