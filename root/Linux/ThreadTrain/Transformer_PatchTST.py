#!/usr/bin/env python3
"""PatchTST adaptado para classificação supervisionada de tráfego SCADA.

Ref: Nie et al., "A Time Series is Worth 64 Words", ICLR 2023.
     https://arxiv.org/abs/2211.14730

Adaptações em relação ao paper original (forecasting → classificação):
  - channel-mixing (vs channel-independence do paper)
  - cabeça softmax de 3 classes no lugar da cabeça de projeção temporal
  - pos_embed_type parametrizável: 'learned' (default) ou 'sinusoidal'
"""

import math

import torch
import torch.nn as nn


# ── Embedding ─────────────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """Divide a série em patches e projeta cada um para d_model.

    Args:
        seq_len (int): comprimento da janela de entrada.
        num_features (int): número de variáveis (canais).
        patch_size (int): tamanho de cada patch; deve dividir seq_len.
        d_model (int): dimensão do espaço de embedding.
        pos_embed_type (str): 'learned' ou 'sinusoidal'.

    Shape:
        input:  (batch, seq_len, num_features)
        output: (batch, num_patches, d_model)
    """

    def __init__(self, seq_len, num_features, patch_size, d_model,
                 pos_embed_type='learned'):
        super().__init__()
        if seq_len % patch_size != 0:
            raise ValueError(
                f"seq_len ({seq_len}) deve ser divisível por patch_size ({patch_size})."
            )
        self.patch_size  = patch_size
        self.num_patches = seq_len // patch_size
        self.pos_embed_type = pos_embed_type

        self.proj = nn.Linear(patch_size * num_features, d_model)

        if pos_embed_type == 'learned':
            self.pos_embed = nn.Embedding(self.num_patches, d_model)
        elif pos_embed_type == 'sinusoidal':
            self.register_buffer('pos_embed',
                                 self._sinusoidal_pe(self.num_patches, d_model))
        else:
            raise ValueError(f"pos_embed_type inválido: '{pos_embed_type}'")

    @staticmethod
    def _sinusoidal_pe(n_pos, d_model):
        """Positional encoding senoidal de Vaswani et al. (2017)."""
        pe       = torch.zeros(n_pos, d_model)
        position = torch.arange(n_pos, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        batch, seq_len, num_features = x.shape
        x      = x.reshape(batch, self.num_patches, self.patch_size * num_features)
        tokens = self.proj(x)

        if self.pos_embed_type == 'learned':
            pos = self.pos_embed(
                torch.arange(self.num_patches, device=x.device, dtype=torch.long)
            ).unsqueeze(0)
        else:
            pos = self.pos_embed.unsqueeze(0)

        return tokens + pos


# ── Modelo ────────────────────────────────────────────────────────────────────

class PatchTST(nn.Module):
    """PatchTST para classificação supervisionada.

    Pipeline: PatchEmbedding → TransformerEncoder → mean pool → Linear.

    Args:
        seq_len (int): comprimento da janela. Default: 32.
        num_features (int): número de variáveis de entrada. Default: 3.
        num_classes (int): número de classes de saída. Default: 3.
        patch_size (int): tamanho de cada patch. Default: 4.
        d_model (int): dimensão interna do Transformer. Default: 128.
        n_heads (int): número de cabeças de atenção. Default: 4.
        n_layers (int): número de camadas do encoder. Default: 3.
        dim_feedforward (int): dimensão da FFN interna. Default: 256.
        dropout (float): taxa de dropout. Default: 0.1.
        pos_embed_type (str): tipo de PE — 'learned' ou 'sinusoidal'.
    """

    def __init__(
        self,
        seq_len=32,
        num_features=3,
        num_classes=3,
        patch_size=4,
        d_model=128,
        n_heads=4,
        n_layers=3,
        dim_feedforward=256,
        dropout=0.1,
        pos_embed_type='learned',
    ):
        super().__init__()

        self.embed = PatchEmbedding(
            seq_len, num_features, patch_size, d_model,
            pos_embed_type=pos_embed_type,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        # enable_nested_tensor incompatível com norm_first=True; suprime UserWarning.
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
            enable_nested_tensor=False,
        )

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        """
        Args:
            x (Tensor): (batch, seq_len, num_features), float32.

        Returns:
            Tensor: (batch, num_classes) — logits crus (sem softmax).
        """
        tokens = self.embed(x)
        tokens = self.norm(self.encoder(tokens))
        return self.head(tokens.mean(dim=1))

    def count_parameters(self):
        """Retorna o número de parâmetros treináveis."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Smoke test ────────────────────────────────────────────────────────────────

def _smoke_test():
    device = (
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("cpu")
    )
    model = PatchTST(seq_len=32, num_features=3, num_classes=3).to(device)
    x     = torch.randn(4, 32, 3, device=device)
    out   = model(x)

    assert out.shape == (4, 3), f"shape inesperado: {out.shape}"
    assert not torch.isnan(out).any()

    print(f"device={device}  params={model.count_parameters():,}  "
          f"input={tuple(x.shape)}  output={tuple(out.shape)}")
    print("[OK]")


if __name__ == "__main__":
    _smoke_test()
