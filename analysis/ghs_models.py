"""
Neural architectures, kept in one place so training and inference cannot drift.
Definitions must stay byte-compatible with the saved state dicts.
"""
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Plain feed-forward net on one-hot encoded input."""
    def __init__(self, d_in, hidden=(256, 128, 64), p_drop=0.3):
        super().__init__()
        layers, prev = [], d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p_drop)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_num, x_cat=None):
        return self.net(x_num).squeeze(-1)


class TabTransformer(nn.Module):
    """Categorical embeddings -> self-attention across features -> MLP head."""
    def __init__(self, cardinalities, n_num, d_model=32, n_heads=4, n_blocks=2, p_drop=0.2):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c + 2, d_model) for c in cardinalities])
        self.col = nn.Parameter(torch.randn(len(cardinalities), d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                         dim_feedforward=d_model * 4, dropout=p_drop,
                                         batch_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(enc, num_layers=n_blocks)
        self.num_norm = nn.LayerNorm(n_num)
        d_cat = d_model * len(cardinalities)
        self.head = nn.Sequential(
            nn.Linear(d_cat + n_num, 128), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(p_drop), nn.Linear(64, 1))

    def forward(self, x_num, x_cat):
        toks = torch.stack([e(x_cat[:, i]) for i, e in enumerate(self.embs)], 1) + self.col
        z = self.tr(toks).flatten(1)
        return self.head(torch.cat([z, self.num_norm(x_num)], 1)).squeeze(-1)
