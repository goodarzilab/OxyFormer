from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


IDENTIFIER_COLUMNS = [
    "fips",
    "county_name",
    "county_label",
]

OUTCOME_COLUMNS = [
    "all_cancer_incidence_rate",
    "all_cancer_mortality_rate",
    "all_cancer_incidence_count",
    "all_cancer_mortality_count",
    "all_cancer_incidence_population",
    "all_cancer_mortality_population",
]

EXPOSURE_FEATURES = [
    "elevation_m",
    "oxygen_proxy_mmhg",
    "oxygen_fraction_of_sea_level",
    "hypoxia_burden",
]

MEDIATOR_FEATURES = [
    "local_diabetes_pct",
    "local_obesity_pct",
    "places_diabetes_pct",
    "places_obesity_pct",
    "places_high_blood_pressure_pct",
    "places_stroke_pct",
]

DEFAULT_EMBEDDING_DIM = 24
DEFAULT_TOKEN_DIM = 48
DEFAULT_LAYERS = 3
DEFAULT_HEADS = 4
DEFAULT_BATCH_SIZE = 256
DEFAULT_EPOCHS = 120
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_MASK_RATE = 0.30
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_EMBEDDING_PENALTY = 5e-4
DEFAULT_AUXILIARY_WEIGHT = 0.35
DEFAULT_CONSISTENCY_WEIGHT = 0.25
DEFAULT_SEED = 20260306


AUXILIARY_TARGET_COLUMNS = [f"z_{feature}" for feature in MEDIATOR_FEATURES]


def build_excluded_columns() -> set[str]:
    excluded = set(IDENTIFIER_COLUMNS + OUTCOME_COLUMNS)
    for feature in EXPOSURE_FEATURES + MEDIATOR_FEATURES:
        excluded.add(f"{feature}__missing")
        excluded.add(f"z_{feature}")
    return excluded


def load_foundation_table(root_dir: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    model_matrix_path = root_dir / "outputs" / "phase25" / "curated_model_matrix.csv"
    if not model_matrix_path.exists():
        raise FileNotFoundError(f"Curated model matrix not found at {model_matrix_path}")

    dataframe = pd.read_csv(model_matrix_path, low_memory=False, dtype={"fips": str})
    excluded = build_excluded_columns()
    feature_columns = [column for column in dataframe.columns if column not in excluded]
    auxiliary_columns = [column for column in AUXILIARY_TARGET_COLUMNS if column in dataframe.columns]
    if not feature_columns:
        raise ValueError("No foundation-model feature columns were selected.")
    if not auxiliary_columns:
        raise ValueError("No auxiliary multitask target columns were found in the model matrix.")
    return dataframe, feature_columns, auxiliary_columns


class TabularAttentionFoundationModel(nn.Module):
    def __init__(
        self,
        n_features: int,
        token_dim: int,
        embedding_dim: int,
        n_heads: int,
        n_layers: int,
        auxiliary_dim: int,
    ) -> None:
        super().__init__()
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
        self.auxiliary_head = nn.Sequential(
            nn.Linear(embedding_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, auxiliary_dim),
        )

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
        auxiliary = self.auxiliary_head(embedding)
        return {
            "embedding": embedding,
            "projection": projection,
            "reconstruction": reconstruction,
            "auxiliary": auxiliary,
        }


def masked_reconstruction_loss(reconstruction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked_count = mask.sum()
    if masked_count.item() == 0:
        return F.mse_loss(reconstruction, target)
    return ((reconstruction - target) ** 2 * mask.float()).sum() / masked_count


def masked_auxiliary_loss(prediction: torch.Tensor, target: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    observed_count = observed.sum()
    if observed_count.item() == 0:
        return torch.zeros((), device=prediction.device)
    return (((prediction - target) ** 2) * observed).sum() / observed_count


def build_loss_dict(
    outputs_a: dict[str, torch.Tensor],
    outputs_b: dict[str, torch.Tensor],
    features: torch.Tensor,
    mask_a: torch.Tensor,
    mask_b: torch.Tensor,
    auxiliary_targets: torch.Tensor,
    auxiliary_observed: torch.Tensor,
    embedding_penalty: float,
    auxiliary_weight: float,
    consistency_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    reconstruction_loss = masked_reconstruction_loss(outputs_a["reconstruction"], features, mask_a)
    reconstruction_loss = reconstruction_loss + masked_reconstruction_loss(outputs_b["reconstruction"], features, mask_b)

    auxiliary_loss = masked_auxiliary_loss(outputs_a["auxiliary"], auxiliary_targets, auxiliary_observed)
    auxiliary_loss = auxiliary_loss + masked_auxiliary_loss(outputs_b["auxiliary"], auxiliary_targets, auxiliary_observed)

    consistency_loss = ((outputs_a["projection"] - outputs_b["projection"]) ** 2).mean()
    embedding_loss = (outputs_a["embedding"] ** 2).mean() + (outputs_b["embedding"] ** 2).mean()

    total_loss = reconstruction_loss
    total_loss = total_loss + auxiliary_weight * auxiliary_loss
    total_loss = total_loss + consistency_weight * consistency_loss
    total_loss = total_loss + embedding_penalty * embedding_loss

    metrics = {
        "total_loss": float(total_loss.detach().cpu().item()),
        "reconstruction_loss": float(reconstruction_loss.detach().cpu().item()),
        "auxiliary_loss": float(auxiliary_loss.detach().cpu().item()),
        "consistency_loss": float(consistency_loss.detach().cpu().item()),
        "embedding_loss": float(embedding_loss.detach().cpu().item()),
    }
    return total_loss, metrics


def train_foundation_model(
    features: np.ndarray,
    auxiliary_targets: np.ndarray,
    auxiliary_observed: np.ndarray,
    embedding_dim: int,
    token_dim: int,
    n_layers: int,
    n_heads: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    mask_rate: float,
    weight_decay: float,
    embedding_penalty: float,
    auxiliary_weight: float,
    consistency_weight: float,
    seed: int,
) -> tuple[TabularAttentionFoundationModel, pd.DataFrame]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    feature_tensor = torch.tensor(features, dtype=torch.float32, device=device)
    auxiliary_tensor = torch.tensor(auxiliary_targets, dtype=torch.float32, device=device)
    observed_tensor = torch.tensor(auxiliary_observed, dtype=torch.float32, device=device)

    model = TabularAttentionFoundationModel(
        n_features=feature_tensor.shape[1],
        token_dim=token_dim,
        embedding_dim=embedding_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        auxiliary_dim=auxiliary_tensor.shape[1],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history_rows = []
    generator = np.random.default_rng(seed)

    row_count = feature_tensor.shape[0]
    for epoch_index in range(1, epochs + 1):
        permutation = generator.permutation(row_count)
        epoch_metrics = []
        model.train()

        for batch_start in range(0, row_count, batch_size):
            batch_indices = permutation[batch_start: batch_start + batch_size]
            batch_features = feature_tensor[batch_indices]
            batch_auxiliary = auxiliary_tensor[batch_indices]
            batch_observed = observed_tensor[batch_indices]

            mask_a = torch.tensor(generator.random(batch_features.shape) < mask_rate, dtype=torch.bool, device=device)
            mask_b = torch.tensor(generator.random(batch_features.shape) < mask_rate, dtype=torch.bool, device=device)

            outputs_a = model(batch_features, mask_a)
            outputs_b = model(batch_features, mask_b)
            total_loss, metrics = build_loss_dict(
                outputs_a=outputs_a,
                outputs_b=outputs_b,
                features=batch_features,
                mask_a=mask_a,
                mask_b=mask_b,
                auxiliary_targets=batch_auxiliary,
                auxiliary_observed=batch_observed,
                embedding_penalty=embedding_penalty,
                auxiliary_weight=auxiliary_weight,
                consistency_weight=consistency_weight,
            )

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_metrics.append(metrics)

        averaged = {
            key: float(np.mean([metrics[key] for metrics in epoch_metrics]))
            for key in epoch_metrics[0]
        }
        averaged["epoch"] = epoch_index
        history_rows.append(averaged)

    return model, pd.DataFrame(history_rows)


def encode_embeddings(model: TabularAttentionFoundationModel, features: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        feature_tensor = torch.tensor(features, dtype=torch.float32)
        mask = torch.zeros_like(feature_tensor, dtype=torch.bool)
        embeddings, _ = model.encode(feature_tensor, mask)
    return embeddings.cpu().numpy().astype(float)


def final_loss_summary(
    model: TabularAttentionFoundationModel,
    features: np.ndarray,
    auxiliary_targets: np.ndarray,
    auxiliary_observed: np.ndarray,
    mask_rate: float,
    embedding_penalty: float,
    auxiliary_weight: float,
    consistency_weight: float,
    seed: int,
) -> dict[str, float]:
    model.eval()
    generator = np.random.default_rng(seed + 99)
    with torch.no_grad():
        feature_tensor = torch.tensor(features, dtype=torch.float32)
        auxiliary_tensor = torch.tensor(auxiliary_targets, dtype=torch.float32)
        observed_tensor = torch.tensor(auxiliary_observed, dtype=torch.float32)
        mask_a = torch.tensor(generator.random(feature_tensor.shape) < mask_rate, dtype=torch.bool)
        mask_b = torch.tensor(generator.random(feature_tensor.shape) < mask_rate, dtype=torch.bool)
        outputs_a = model(feature_tensor, mask_a)
        outputs_b = model(feature_tensor, mask_b)
        _, metrics = build_loss_dict(
            outputs_a=outputs_a,
            outputs_b=outputs_b,
            features=feature_tensor,
            mask_a=mask_a,
            mask_b=mask_b,
            auxiliary_targets=auxiliary_tensor,
            auxiliary_observed=observed_tensor,
            embedding_penalty=embedding_penalty,
            auxiliary_weight=auxiliary_weight,
            consistency_weight=consistency_weight,
        )
    return metrics


def run_phase26(
    root_dir: Path,
    embedding_dim: int,
    token_dim: int,
    n_layers: int,
    n_heads: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    mask_rate: float,
    weight_decay: float,
    embedding_penalty: float,
    auxiliary_weight: float,
    consistency_weight: float,
    seed: int,
) -> dict:
    output_dir = root_dir / "outputs" / "phase26"
    output_dir.mkdir(parents=True, exist_ok=True)

    foundation_table, feature_columns, auxiliary_columns = load_foundation_table(root_dir)
    feature_matrix = foundation_table[feature_columns].astype(float).to_numpy()
    auxiliary_matrix = foundation_table[auxiliary_columns].astype(float).to_numpy()
    auxiliary_observed = (~np.isnan(auxiliary_matrix)).astype(float)
    auxiliary_filled = np.nan_to_num(auxiliary_matrix, nan=0.0)

    model, history = train_foundation_model(
        features=feature_matrix,
        auxiliary_targets=auxiliary_filled,
        auxiliary_observed=auxiliary_observed,
        embedding_dim=embedding_dim,
        token_dim=token_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        mask_rate=mask_rate,
        weight_decay=weight_decay,
        embedding_penalty=embedding_penalty,
        auxiliary_weight=auxiliary_weight,
        consistency_weight=consistency_weight,
        seed=seed,
    )

    embeddings = encode_embeddings(model, feature_matrix)
    embedding_columns = [f"foundation_embedding_{index + 1:02d}" for index in range(embedding_dim)]
    embedding_frame = pd.DataFrame(embeddings, columns=embedding_columns)
    embedding_frame.insert(0, "fips", foundation_table["fips"].astype(str))
    embedding_frame.insert(1, "county_name", foundation_table["county_name"])

    final_losses = final_loss_summary(
        model=model,
        features=feature_matrix,
        auxiliary_targets=auxiliary_filled,
        auxiliary_observed=auxiliary_observed,
        mask_rate=mask_rate,
        embedding_penalty=embedding_penalty,
        auxiliary_weight=auxiliary_weight,
        consistency_weight=consistency_weight,
        seed=seed,
    )

    torch.save(model.state_dict(), output_dir / "foundation_model_weights.pt")
    embedding_frame.to_csv(output_dir / "county_foundation_embeddings.csv", index=False)
    history.to_csv(output_dir / "foundation_training_history.csv", index=False)

    summary = {
        "model_family": "dual_view_masked_tabular_transformer_with_consistency_and_auxiliary_heads",
        "input_rows": int(feature_matrix.shape[0]),
        "input_dim": int(feature_matrix.shape[1]),
        "embedding_dim": int(embedding_dim),
        "token_dim": int(token_dim),
        "n_layers": int(n_layers),
        "n_heads": int(n_heads),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "mask_rate": float(mask_rate),
        "weight_decay": float(weight_decay),
        "embedding_penalty": float(embedding_penalty),
        "auxiliary_weight": float(auxiliary_weight),
        "consistency_weight": float(consistency_weight),
        "seed": int(seed),
        "feature_columns": feature_columns,
        "auxiliary_columns": auxiliary_columns,
        "final_losses": final_losses,
        "outputs": {
            "county_foundation_embeddings": "outputs/phase26/county_foundation_embeddings.csv",
            "foundation_training_history": "outputs/phase26/foundation_training_history.csv",
            "foundation_model_weights": "outputs/phase26/foundation_model_weights.pt",
        },
    }
    (output_dir / "phase26_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an attention-based county foundation model on the curated model matrix.")
    parser.add_argument("--root-dir", type=Path, default=Path("."), help="Project root directory.")
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM, help="Embedding dimension.")
    parser.add_argument("--token-dim", type=int, default=DEFAULT_TOKEN_DIM, help="Per-feature token width.")
    parser.add_argument("--layers", type=int, default=DEFAULT_LAYERS, help="Transformer encoder layer count.")
    parser.add_argument("--heads", type=int, default=DEFAULT_HEADS, help="Attention head count.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE, help="AdamW learning rate.")
    parser.add_argument("--mask-rate", type=float, default=DEFAULT_MASK_RATE, help="Fraction of tabular cells masked per view.")
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="AdamW weight decay.")
    parser.add_argument("--embedding-penalty", type=float, default=DEFAULT_EMBEDDING_PENALTY, help="Penalty on embedding magnitude.")
    parser.add_argument("--auxiliary-weight", type=float, default=DEFAULT_AUXILIARY_WEIGHT, help="Weight on the auxiliary multitask loss.")
    parser.add_argument("--consistency-weight", type=float, default=DEFAULT_CONSISTENCY_WEIGHT, help="Weight on the dual-view consistency loss.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            run_phase26(
                root_dir=args.root_dir,
                embedding_dim=args.embedding_dim,
                token_dim=args.token_dim,
                n_layers=args.layers,
                n_heads=args.heads,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                mask_rate=args.mask_rate,
                weight_decay=args.weight_decay,
                embedding_penalty=args.embedding_penalty,
                auxiliary_weight=args.auxiliary_weight,
                consistency_weight=args.consistency_weight,
                seed=args.seed,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
