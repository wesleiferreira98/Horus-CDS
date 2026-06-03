#!/usr/bin/env python3
"""Thread de treinamento do PatchTST V5 com protocolo de N execuções.

Protocolo: seeds fixas {0..4}, pré-processamento único fora do loop,
early stopping por val_loss, CSV de métricas por seed + summary agregado.

Ref: Pineau et al. (2020), Bouthillier et al. (2021) — reproducibilidade em ML.
"""

import os
import sys
import time
import random
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score,
    average_precision_score, roc_auc_score,
)
from PyQt5.QtCore import QThread, pyqtSignal

from TratamentoDeDados.temporal_split import (
    temporal_split,
    add_temporal_features_safe,
    analyze_class_distribution,
)
from TratamentoDeDados.windowing import (
    make_windows,
    DEFAULT_SEQ_LEN,
    DEFAULT_TRANSFORMER_FEATURES,
    CATEGORY_TO_INT,
    INT_TO_CATEGORY,
)
from ThreadTrain.Transformer_PatchTST import PatchTST


# ── Constantes ────────────────────────────────────────────────────────────────

DEFAULT_SEEDS = [0, 1, 2, 3, 4]


# ── Utilitários ───────────────────────────────────────────────────────────────

def set_seed(seed):
    """Fixa random, numpy e torch (CPU/CUDA/MPS)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)
        except AttributeError:
            pass


def get_device():
    """Retorna MPS, CUDA ou CPU — nessa ordem de preferência."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Treino de uma execução ─────────────────────────────────────────────────────

def train_one_run(
    seed, X_train, y_train, X_val, y_val, X_test, y_test,
    class_weights, model_config, train_config, device,
    progress_callback=None, log_callback=None,
):
    """Treina o modelo para uma seed e avalia no conjunto de teste.

    Args:
        seed (int): seed para reproducibilidade.
        X_train, y_train: arrays de treino (float32, int64).
        X_val, y_val: arrays de validação para early stopping.
        X_test, y_test: arrays de teste para avaliação final.
        class_weights (dict): pesos por classe, ex: {'ilegal': 1.04, ...}.
        model_config (dict): kwargs para PatchTST.__init__.
        train_config (dict): lr, batch_size, n_epochs, patience, weight_decay.
        device (torch.device): dispositivo de execução.
        progress_callback (callable | None): cb(epoch, total).
        log_callback (callable | None): cb(msg).

    Returns:
        tuple: (metrics dict, best_state dict, history list,
                test_preds ndarray, y_true ndarray)
    """
    set_seed(seed)

    def log(msg):
        if log_callback:
            log_callback(msg)

    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)
    Xv = torch.tensor(X_val,   dtype=torch.float32)
    yv = torch.tensor(y_val,   dtype=torch.long)
    Xs = torch.tensor(X_test,  dtype=torch.float32)
    ys = torch.tensor(y_test,  dtype=torch.long)

    # Generator próprio garante shuffle reproduzível
    gen = torch.Generator().manual_seed(seed)
    train_ds     = TensorDataset(Xt, yt)
    train_loader = DataLoader(
        train_ds, batch_size=train_config['batch_size'],
        shuffle=True, generator=gen, drop_last=False,
    )

    model   = PatchTST(**model_config).to(device)
    cw_list = [float(class_weights.get(INT_TO_CATEGORY[i], 1.0)) for i in range(3)]
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(cw_list, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config['lr'],
        weight_decay=train_config['weight_decay'],
    )

    n_epochs        = train_config['n_epochs']
    patience        = train_config['patience']
    best_val_loss   = float('inf')
    best_epoch      = -1
    best_state      = None
    patience_counter= 0
    history         = []
    Xv_dev, yv_dev  = Xv.to(device), yv.to(device)

    t_start = time.time()
    for epoch in range(n_epochs):
        model.train()
        train_loss_sum = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * xb.size(0)
        train_loss = train_loss_sum / len(train_ds)

        model.eval()
        with torch.no_grad():
            val_logits = model(Xv_dev)
            val_loss   = loss_fn(val_logits, yv_dev).item()
            val_preds  = val_logits.argmax(dim=1).cpu().numpy()
            val_acc    = accuracy_score(yv.numpy(), val_preds)

        history.append({
            'epoch': epoch, 'train_loss': train_loss,
            'val_loss': val_loss, 'val_acc': val_acc,
        })

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_epoch       = epoch
            best_state       = {k: v.detach().cpu().clone()
                                for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        log(f"[seed={seed}] epoch={epoch+1}/{n_epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.4f}")

        if progress_callback:
            progress_callback(epoch + 1, n_epochs)

        if patience_counter >= patience:
            log(f"[seed={seed}] early stopping epoch {epoch+1} "
                f"(best={best_epoch+1})")
            break

    train_elapsed = time.time() - t_start

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_logits = model(Xs.to(device))
        test_probs  = torch.softmax(test_logits, dim=-1).cpu().numpy()
        test_preds  = test_logits.argmax(dim=-1).cpu().numpy()
    y_true = ys.numpy()

    acc = accuracy_score(y_true, test_preds)
    f1m = f1_score(y_true, test_preds, average='macro',    zero_division=0)
    f1w = f1_score(y_true, test_preds, average='weighted', zero_division=0)

    try:
        auc_pr = average_precision_score(np.eye(3)[y_true], test_probs, average='macro')
    except Exception:
        auc_pr = float('nan')

    try:
        auc_roc = roc_auc_score(y_true, test_probs, multi_class='ovr', average='macro')
    except Exception:
        auc_roc = float('nan')

    metrics = {
        'seed':             seed,
        'accuracy':         acc,
        'f1_macro':         f1m,
        'f1_weighted':      f1w,
        'auc_pr':           auc_pr,
        'auc_roc':          auc_roc,
        'epoch_best':       best_epoch + 1,
        'n_epochs_trained': len(history),
        'train_time_s':     train_elapsed,
        'n_params':         sum(p.numel() for p in model.parameters() if p.requires_grad),
        'timestamp':        datetime.now().isoformat(timespec='seconds'),
    }

    return metrics, best_state, history, test_preds, y_true


# ── Agregação ─────────────────────────────────────────────────────────────────

def summarize_runs(runs_df, metrics_cols=None):
    """Retorna DataFrame com média ± desvio (ddof=1) por métrica.

    Args:
        runs_df (pd.DataFrame): uma linha por seed.
        metrics_cols (list | None): colunas a agregar; usa default se None.

    Returns:
        pd.DataFrame: colunas metric, mean, std, min, max, n_runs.
    """
    if metrics_cols is None:
        metrics_cols = ['accuracy', 'f1_macro', 'f1_weighted',
                        'auc_pr', 'auc_roc', 'train_time_s']
    rows = []
    for m in metrics_cols:
        vals = runs_df[m].dropna().values
        rows.append({
            'metric': m,
            'mean':   float(np.mean(vals)),
            'std':    float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            'min':    float(np.min(vals)),
            'max':    float(np.max(vals)),
            'n_runs': int(len(vals)),
        })
    return pd.DataFrame(rows)


# ── QThread ───────────────────────────────────────────────────────────────────

class TrainingThreadTransformer(QThread):
    """Thread de treinamento do V5 — orquestra N execuções com seeds fixas.

    Sinais emitidos espelham TrainingThreadTCN para integração na UI.
    """

    show_test_accuracy           = pyqtSignal(float)
    update_progress              = pyqtSignal(float)
    update_metrics_chart         = pyqtSignal(float, float, str)
    update_metrics_chart_boxplot = pyqtSignal(np.ndarray, np.ndarray, str)
    log_message                  = pyqtSignal(str)

    def __init__(
        self,
        data_set,
        seeds=None,
        seq_len=DEFAULT_SEQ_LEN,
        val_ratio=0.15,
        feature_cols=None,
        d_model=128, n_heads=4, n_layers=3,
        patch_size=4, dropout=0.1,
        pos_embed_type='learned',
        batch_size=64, n_epochs=50,
        lr=1e-3, weight_decay=1e-4, patience=10,
    ):
        super().__init__()
        self.data_set     = data_set
        self.seeds        = seeds if seeds is not None else DEFAULT_SEEDS
        self.seq_len      = seq_len
        self.val_ratio    = val_ratio
        self.feature_cols = (feature_cols if feature_cols is not None
                             else list(DEFAULT_TRANSFORMER_FEATURES))
        self.model_config = dict(
            seq_len=seq_len,
            num_features=len(self.feature_cols),
            num_classes=3,
            patch_size=patch_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            pos_embed_type=pos_embed_type,
        )
        self.train_config = dict(
            batch_size=batch_size,
            n_epochs=n_epochs,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
        )
        self.base_dir   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.output_dir = os.path.join(
            self.base_dir, "DadosDoPostreino", "ModelosNew", "Transformer"
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def _log(self, msg):
        print(msg)
        self.log_message.emit(msg)

    def _split_train_val(self, train_df):
        n = len(train_df)
        idx = int(n * (1 - self.val_ratio))
        return train_df.iloc[:idx].copy(), train_df.iloc[idx:].copy()

    def run(self):
        """Executa o protocolo completo: pré-proc → windowing → N treinos → CSV."""
        self._log("=" * 70)
        self._log(f"V5 PatchTST — {len(self.seeds)} execucoes")
        self._log("=" * 70)

        device = get_device()
        self._log(f"Device: {device}  features: {self.feature_cols}  "
                  f"seq_len={self.seq_len}  lr={self.train_config['lr']}")

        # Pré-processamento executado uma vez — mesmo pipeline do TCN
        self._log("\n[1/4] Pre-processamento")
        train_df, test_df = temporal_split(self.data_set, model_name="Transformer-V5")
        class_weights     = analyze_class_distribution(train_df, model_name="Transformer-V5")
        train_df = add_temporal_features_safe(train_df, is_train=True,  model_name="Transformer-V5")
        test_df  = add_temporal_features_safe(test_df,  is_train=False, model_name="Transformer-V5")

        train_part, val_part = self._split_train_val(train_df)
        self._log(f"Treino: {len(train_part)}  Val: {len(val_part)}  Teste: {len(test_df)}")

        self._log("\n[2/4] Windowing")
        kw = dict(feature_cols=self.feature_cols, seq_len=self.seq_len)
        X_train, y_train = make_windows(train_part, **kw, model_name="train")
        X_val,   y_val   = make_windows(val_part,   **kw, model_name="val")
        X_test,  y_test  = make_windows(test_df,    **kw, model_name="test")

        self._log(f"\n[3/4] Loop de {len(self.seeds)} execucoes")
        runs               = []
        total_epoch_budget = len(self.seeds) * self.train_config['n_epochs']

        def make_progress_cb(seed_idx):
            def cb(epoch_done, _):
                approx = seed_idx * self.train_config['n_epochs'] + epoch_done
                self.update_progress.emit(approx / total_epoch_budget)
            return cb

        for i, seed in enumerate(self.seeds):
            self._log(f"\n--- Execucao {i+1}/{len(self.seeds)} (seed={seed}) ---")
            metrics, best_state, history, _, _ = train_one_run(
                seed=seed,
                X_train=X_train, y_train=y_train,
                X_val=X_val,     y_val=y_val,
                X_test=X_test,   y_test=y_test,
                class_weights=class_weights,
                model_config=self.model_config,
                train_config=self.train_config,
                device=device,
                progress_callback=make_progress_cb(i),
                log_callback=lambda m: None,
            )
            runs.append(metrics)

            torch.save({
                'state_dict':   best_state,
                'model_config': self.model_config,
                'train_config': self.train_config,
                'seed':         seed,
                'metrics':      metrics,
            }, os.path.join(self.output_dir, f"patchtst_v5_seed{seed}.pt"))

            hist_df = pd.DataFrame(history)
            hist_df['seed'] = seed
            hist_df.to_csv(
                os.path.join(self.output_dir, f"patchtst_v5_seed{seed}_history.csv"),
                index=False,
            )

            self._log(f"[seed={seed}] acc={metrics['accuracy']:.4f}  "
                      f"f1_macro={metrics['f1_macro']:.4f}  "
                      f"auc_pr={metrics['auc_pr']:.4f}  "
                      f"auc_roc={metrics['auc_roc']:.4f}  "
                      f"t={metrics['train_time_s']:.1f}s")

        self._log("\n[4/4] Agregacao")
        runs_df = pd.DataFrame(runs)
        runs_df.to_csv(os.path.join(self.output_dir, "patchtst_v5_runs.csv"), index=False)

        summary_df = summarize_runs(runs_df)
        summary_df.to_csv(os.path.join(self.output_dir, "patchtst_v5_summary.csv"), index=False)

        for _, row in summary_df.iterrows():
            self._log(f"  {row['metric']:14s}: {row['mean']:.4f} +/- {row['std']:.4f}"
                      f"  (min={row['min']:.4f}, max={row['max']:.4f})")

        mean_acc = float(summary_df.loc[summary_df['metric'] == 'accuracy', 'mean'].iloc[0])
        std_acc  = float(summary_df.loc[summary_df['metric'] == 'accuracy', 'std'].iloc[0])
        self.show_test_accuracy.emit(mean_acc)
        self.update_metrics_chart.emit(mean_acc, std_acc, "PatchTST V5")
        self.update_metrics_chart_boxplot.emit(
            runs_df['accuracy'].values, runs_df['auc_pr'].values, "PatchTST V5",
        )
        self.update_progress.emit(1.0)
        self._log("\nProtocolo completo.\n" + "=" * 70)


# ── Smoke test ────────────────────────────────────────────────────────────────

def _smoke_test():
    csv_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "DadosReais", "dados_normalizados_smartgrid.csv",
    )
    if not os.path.exists(csv_path):
        print(f"dataset não encontrado: {csv_path}")
        return
    thread = TrainingThreadTransformer(
        data_set=pd.read_csv(csv_path), seeds=[0], n_epochs=3, patience=2,
    )
    thread.run()


if __name__ == "__main__":
    _smoke_test()
