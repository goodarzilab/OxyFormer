"""Repaired masked-feature benchmark. Production v2 uses the common stage CLI."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from oxyformer.provenance import require


class TabularAttentionFoundationModel(nn.Module):
    def __init__(
        self,
        n_features: int,
        token_dim: int,
        embedding_dim: int,
        n_heads: int,
        n_layers: int,
        auxiliary_dim: int = 0,
    ) -> None:
        super().__init__()
        require(auxiliary_dim == 0, "disease auxiliary heads are disabled")
        self.n_features = n_features
        self.token_dim = token_dim
        self.feature_scale = nn.Parameter(torch.randn(n_features, token_dim) * 0.02)
        self.feature_bias = nn.Parameter(torch.zeros(n_features, token_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.input_norm = nn.LayerNorm(token_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=n_heads,
            dim_feedforward=token_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.embedding_head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, embedding_dim),
            nn.Tanh(),
        )
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, token_dim * 2),
            nn.GELU(),
            nn.Linear(token_dim * 2, n_features),
        )
        self.auxiliary_head = None

    def tokenize(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tokens = features.unsqueeze(-1) * self.feature_scale.unsqueeze(0) + self.feature_bias.unsqueeze(0)
        tokens = self.input_norm(tokens)
        mask_token = self.mask_token.expand(features.shape[0], self.n_features, self.token_dim)
        return torch.where(mask.unsqueeze(-1), mask_token, tokens)

    def encode(self, features: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.tokenize(features, mask)
        cls_token = self.cls_token.expand(features.shape[0], 1, self.token_dim)
        sequence = torch.cat([cls_token, tokens], dim=1)
        encoded = self.encoder(sequence)
        cls_encoded = encoded[:, 0]
        embedding = self.embedding_head(cls_encoded)
        projection = F.normalize(self.projection_head(embedding), dim=-1)
        return embedding, projection

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        embedding, projection = self.encode(features, mask)
        reconstruction = self.decoder(embedding)
        return {
            "embedding": embedding,
            "projection": projection,
            "reconstruction": reconstruction,
        }


def masked_reconstruction_loss(reconstruction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked_count = mask.sum()
    if masked_count.item() == 0:
        return F.mse_loss(reconstruction, target)
    return ((reconstruction - target) ** 2 * mask.float()).sum() / masked_count


def encode_embeddings(model: TabularAttentionFoundationModel, features: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        feature_tensor = torch.tensor(features, dtype=torch.float32)
        mask = torch.zeros_like(feature_tensor, dtype=torch.bool)
        embeddings, _ = model.encode(feature_tensor, mask)
    return embeddings.cpu().numpy().astype(float)


def fit_repaired(features, *, epochs=120, seed=20260306):
    """A covariate-only reconstruction benchmark; no supervised disease channel.

    Caller supplies approved training-fold features only. The benchmark keeps
    the historical architecture and defaults; it does not produce v2 nuisances.
    """
    require(epochs > 0, "positive epochs required")
    require(features.ndim == 2 and min(features.shape) > 0 and np.isfinite(features).all(),
            "finite nonempty feature matrix required")
    torch.manual_seed(seed)
    model = TabularAttentionFoundationModel(features.shape[1], 48, 24, 4, 3)
    tensor = torch.tensor(features, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    history = []
    for epoch in range(epochs):
        losses = []
        for indices in torch.randperm(len(tensor)).split(256):
            batch = tensor[indices]
            mask = torch.rand_like(batch) < 0.30
            outputs = model(batch, mask)
            loss = masked_reconstruction_loss(outputs["reconstruction"], batch, mask)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(loss.detach().item())
        history.append({"epoch": epoch + 1, "reconstruction_loss": float(np.mean(losses))})
    return model, history
