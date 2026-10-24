"""NEST with two independent implementations selected only by ``mode``.

The public API is shared, but the research workflows are not merged internally:

* ``missing_gene`` uses a matched scRNA-seq reference, cross-modal mapping and
  gene-domain correction before graph-guided reconstruction.
* ``dropout`` uses only the input ST matrix. Autoencoder and graph-autoencoder
  reconstruction losses are evaluated on observed non-zero entries, and the
  default output replaces only input zeros.

NEST expects non-negative, consistently transformed expression matrices. It
does not normalize or log-transform inputs implicitly.
"""

from __future__ import annotations

import copy
import gc
import logging
import random
from dataclasses import dataclass, replace
from typing import Literal, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, dense_to_sparse

Mode = Literal["missing_gene", "dropout"]
LOGGER = logging.getLogger("nest_st")


@dataclass(slots=True)
class NESTConfig:
    """Public configuration for the two independently implemented modes.

    Parameters left as ``None`` are resolved by the selected task branch.
    """

    mode: Mode
    spatial_key: str = "spatial"
    n_neighbors: int | None = None
    mapping_neighbors: int | None = None
    gene_neighbors: int | None = None
    latent_dim: int | None = None
    n_pcs: int | None = None
    learning_rate: float | None = None
    ae_epochs: int | None = None
    gae_epochs: int | None = None
    mse_weight: float | None = None
    leiden_resolution: float = 1.0
    lof_contamination: float = 0.3
    fill_zeros_only: bool = True
    large_dataset_threshold: int = 80_000
    map_chunk_size: int = 128
    gene_chunk_size: int = 32
    decomposed_layers: int = 4
    cpu_threads: int = 4
    random_state: int = 42
    device: str = "auto"
    verbose: bool = True

    def __post_init__(self) -> None:
        if self.mode not in {"missing_gene", "dropout"}:
            raise ValueError("mode must be 'missing_gene' or 'dropout'")
        for name in ("n_neighbors", "mapping_neighbors", "gene_neighbors", "latent_dim", "n_pcs"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("ae_epochs", "gae_epochs", "map_chunk_size", "gene_chunk_size", "cpu_threads"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be positive")
        if self.learning_rate is not None and self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.mse_weight is not None and self.mse_weight <= 0:
            raise ValueError("mse_weight must be positive")
        if not 0 < self.lof_contamination <= 0.5:
            raise ValueError("lof_contamination must be in (0, 0.5]")


@dataclass(slots=True)
class _ResolvedConfig:
    mode: Mode
    spatial_key: str
    n_neighbors: int
    mapping_neighbors: int
    gene_neighbors: int
    latent_dim: int
    n_pcs: int
    learning_rate: float
    ae_epochs: int
    gae_epochs: int
    mse_weight: float
    leiden_resolution: float
    lof_contamination: float
    fill_zeros_only: bool
    low_memory: bool
    map_chunk_size: int
    gene_chunk_size: int
    decomposed_layers: int
    cpu_threads: int
    random_state: int
    device: torch.device
    verbose: bool
    zero_rate: float


class _AutoEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        hidden_dim = max(2, input_dim // 2)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm2(self.fc2(self.norm1(self.fc1(x))))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x)
        reconstructed = F.relu(self.fc4(F.relu(self.fc3(latent))))
        return latent, reconstructed


class _GraphAutoEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, n_clusters: int) -> None:
        super().__init__()
        hidden_dim = max(2, input_dim // 2)
        self.gc1 = GCNConv(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.gc2 = GCNConv(hidden_dim, latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.classifier = nn.Linear(latent_dim, n_clusters)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.norm1(self.gc1(x, edge_index))
        latent = self.norm2(self.gc2(hidden, edge_index))
        reconstructed = F.relu(self.fc2(F.relu(self.fc1(latent))))
        return latent, reconstructed, self.classifier(latent)


def _dense32(x: object) -> np.ndarray:
    return np.asarray(x.toarray() if sp.issparse(x) else x, dtype=np.float32)


def _zero_rate(x: object) -> float:
    total = int(x.shape[0] * x.shape[1])
    nonzero = int(x.nnz) if sp.issparse(x) else int(np.count_nonzero(x))
    return 1.0 - nonzero / total


def _spatial_coordinates(adata: ad.AnnData, key: str) -> np.ndarray:
    if key not in adata.obsm:
        raise KeyError(f"adata_st.obsm[{key!r}] is required")
    coordinates = np.asarray(adata.obsm[key], dtype=np.float32)
    if coordinates.ndim != 2 or coordinates.shape[1] not in (2, 3):
        raise ValueError(f"Spatial coordinates must have shape (n_obs, 2 or 3), got {coordinates.shape}")
    if coordinates.shape[0] != adata.n_obs or not np.isfinite(coordinates).all():
        raise ValueError("Spatial coordinates must be finite and aligned with adata_st.obs")
    if coordinates.shape[1] == 3:
        scale = coordinates.std(axis=0, keepdims=True)
        scale[scale < 1e-8] = 1.0
        coordinates = (coordinates - coordinates.mean(axis=0, keepdims=True)) / scale
    return coordinates.copy()


def _check_expression(adata: ad.AnnData, name: str) -> None:
    if adata.n_obs < 3 or adata.n_vars < 2:
        raise ValueError(f"{name} must contain at least 3 observations and 2 genes")
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError(f"{name} must have unique obs_names and var_names")
    x = adata.X
    values = x.data if sp.issparse(x) else np.asarray(x)
    if values.size and (not np.isfinite(values).all() or np.min(values) < 0):
        raise ValueError(f"{name}.X must contain finite, non-negative expression values")


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but CUDA is unavailable")
    return device


def _resolve_config(config: NESTConfig, x: object, n_reference_cells: int) -> _ResolvedConfig:
    rate = _zero_rate(x)
    n_obs, n_vars = x.shape
    if rate < 0.20:
        auto_latent, auto_pcs, auto_mse = max(2, n_vars // 3), 150, 1.5
        auto_lr, auto_ae, auto_gae = 5e-4, 2_000, 4_000
    elif rate > 0.90:
        auto_latent, auto_pcs, auto_mse = max(2, n_vars // 5), 15, 0.5
        auto_lr, auto_ae, auto_gae = 1e-3, 1_000, 2_000
    else:
        auto_latent, auto_pcs, auto_mse = max(2, n_vars // 4), 40, 1.0
        auto_lr, auto_ae, auto_gae = 1e-3, 1_000, 2_000
    return _ResolvedConfig(
        mode=config.mode,
        spatial_key=config.spatial_key,
        n_neighbors=config.n_neighbors or max(10, min(30, n_obs // 100)),
        mapping_neighbors=config.mapping_neighbors or max(5, int(np.floor(n_reference_cells / 1000 + 0.5))),
        gene_neighbors=config.gene_neighbors or max(3, n_vars // 20),
        latent_dim=config.latent_dim or auto_latent,
        n_pcs=config.n_pcs or auto_pcs,
        learning_rate=config.learning_rate or auto_lr,
        ae_epochs=config.ae_epochs or auto_ae,
        gae_epochs=config.gae_epochs or auto_gae,
        mse_weight=config.mse_weight or auto_mse,
        leiden_resolution=config.leiden_resolution,
        lof_contamination=config.lof_contamination,
        fill_zeros_only=config.fill_zeros_only,
        low_memory=n_obs > config.large_dataset_threshold,
        map_chunk_size=config.map_chunk_size,
        gene_chunk_size=config.gene_chunk_size,
        decomposed_layers=config.decomposed_layers,
        cpu_threads=config.cpu_threads,
        random_state=config.random_state,
        device=_resolve_device(config.device),
        verbose=config.verbose,
        zero_rate=rate,
    )


def _resolve_dropout_config(config: NESTConfig, x: object) -> _ResolvedConfig:
    """Defaults used by Reproduce_Methods.ipynb and NEST_dropout.py.

    Dropout recovery has no zero-rate parameter regimes.  Only KNN size and
    latent width follow the input dimensions, as in the original reproduction.
    """
    rate = _zero_rate(x)
    n_obs, n_vars = x.shape
    default_k = max(10, min(30, n_obs // 100))
    if n_obs > 1:
        default_k = min(default_k, n_obs - 1)
    else:
        default_k = 1
    return _ResolvedConfig(
        mode="dropout",
        spatial_key=config.spatial_key,
        n_neighbors=config.n_neighbors or default_k,
        mapping_neighbors=1,
        gene_neighbors=1,
        latent_dim=config.latent_dim or max(1, n_vars // 4),
        n_pcs=config.n_pcs or min(40, max(1, min(n_obs, n_vars) - 1)),
        learning_rate=config.learning_rate or 1e-3,
        ae_epochs=config.ae_epochs or 1_000,
        gae_epochs=config.gae_epochs or 2_000,
        mse_weight=1.0,
        leiden_resolution=1.0,
        lof_contamination=0.3,
        fill_zeros_only=config.fill_zeros_only,
        low_memory=False,
        map_chunk_size=config.map_chunk_size,
        gene_chunk_size=config.gene_chunk_size,
        decomposed_layers=config.decomposed_layers,
        cpu_threads=config.cpu_threads,
        random_state=config.random_state,
        device=_resolve_device(config.device),
        verbose=config.verbose,
        zero_rate=rate,
    )


def _masked_reconstruction_loss(
    prediction: torch.Tensor, target: torch.Tensor, mse_weight: float, masked: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not masked:
        mse = F.mse_loss(prediction, target)
        cosine = 1 - F.cosine_similarity(prediction, target, dim=1).mean()
        return mse_weight * mse + cosine, mse, cosine
    observed = target.ne(0)
    if not torch.any(observed):
        raise ValueError("dropout mode requires at least one observed non-zero expression value")
    mse = F.mse_loss(prediction[observed], target[observed])
    valid_rows = observed.any(dim=1)
    cosine = 1 - F.cosine_similarity(
        prediction[valid_rows] * observed[valid_rows],
        target[valid_rows] * observed[valid_rows],
        dim=1,
    ).mean()
    return mse_weight * mse + cosine, mse, cosine


def _train_autoencoder(x: np.ndarray, cfg: _ResolvedConfig) -> tuple[_AutoEncoder, np.ndarray]:
    tensor = torch.as_tensor(x, dtype=torch.float32, device=cfg.device)
    model = _AutoEncoder(x.shape[1], cfg.latent_dim).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.999))
    for epoch in range(cfg.ae_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _, prediction = model(tensor)
        loss, mse, cosine = _masked_reconstruction_loss(
            prediction, tensor, cfg.mse_weight, masked=cfg.mode == "dropout"
        )
        loss.backward()
        optimizer.step()
        if cfg.verbose and (epoch == 0 or (epoch + 1) % 500 == 0 or epoch + 1 == cfg.ae_epochs):
            LOGGER.info(
                "AE %d/%d: loss=%.4f, mse=%.4f, cosine=%.4f",
                epoch + 1,
                cfg.ae_epochs,
                loss.item(),
                mse.item(),
                cosine.item(),
            )
    model.eval()
    with torch.no_grad():
        latent = model.encode(tensor).cpu().numpy().astype(np.float32, copy=True)
    del tensor, optimizer, prediction, loss
    _release_memory()
    return model, latent


def _map_reference(
    model: _AutoEncoder,
    adata_sc: ad.AnnData,
    common_genes: Sequence[str],
    st_latent: np.ndarray,
    cfg: _ResolvedConfig,
) -> np.ndarray:
    x_sc_common = torch.as_tensor(
        _dense32(adata_sc[:, list(common_genes)].X), dtype=torch.float32, device=cfg.device
    )
    model.eval()
    with torch.no_grad():
        sc_latent = model.encode(x_sc_common).cpu().numpy().astype(np.float32, copy=True)
    del x_sc_common, model
    _release_memory()

    kmap = min(cfg.mapping_neighbors, adata_sc.n_obs)
    neighbors = NearestNeighbors(
        n_neighbors=kmap, metric="euclidean", n_jobs=cfg.cpu_threads
    ).fit(sc_latent)
    distances, indices = neighbors.kneighbors(st_latent)
    weights = 1.0 / (distances + 1e-8)
    weights /= weights.sum(axis=1, keepdims=True)
    mapped = np.empty((st_latent.shape[0], adata_sc.n_vars), dtype=np.float32)
    for gene_start in range(0, adata_sc.n_vars, cfg.gene_chunk_size):
        gene_stop = min(gene_start + cfg.gene_chunk_size, adata_sc.n_vars)
        sc_block = _dense32(adata_sc[:, gene_start:gene_stop].X)
        for cell_start in range(0, st_latent.shape[0], cfg.map_chunk_size):
            cell_stop = min(cell_start + cfg.map_chunk_size, st_latent.shape[0])
            mapped[cell_start:cell_stop, gene_start:gene_stop] = np.einsum(
                "nk,nkg->ng",
                weights[cell_start:cell_stop],
                sc_block[indices[cell_start:cell_stop]],
                optimize=True,
            )
    return mapped


def _correct_gene_domain(
    mapped: np.ndarray,
    observed_common: np.ndarray,
    common_indices: np.ndarray,
    cfg: _ResolvedConfig,
) -> None:
    target_indices = np.flatnonzero(~np.isin(np.arange(mapped.shape[1]), common_indices))
    common_profiles = mapped[:, common_indices].T
    for start in range(0, len(target_indices), cfg.gene_chunk_size):
        block = target_indices[start : start + cfg.gene_chunk_size]
        similarities = cosine_similarity(mapped[:, block].T, common_profiles)
        for row, target_index in enumerate(block):
            top = np.argsort(-similarities[row])[: min(cfg.gene_neighbors, len(common_indices))]
            weight = similarities[row, top]
            weight = np.full(len(top), 1.0 / len(top), dtype=np.float32) if np.isclose(weight.sum(), 0) else weight / weight.sum()
            mapped[:, target_index] = observed_common[:, top] @ weight
    mapped[:, common_indices] = observed_common


def _precluster(
    x: np.ndarray, cfg: _ResolvedConfig
) -> tuple[np.ndarray, np.ndarray]:
    work = ad.AnnData(X=x)
    n_pcs = max(1, min(cfg.n_pcs, work.n_obs - 1, work.n_vars - 1))
    n_neighbors = max(2, min(cfg.n_neighbors, work.n_obs - 1))
    # The original dropout pipeline clusters a capped, standardized matrix;
    # missing-gene completion clusters the native transformed expression.
    if cfg.mode == "dropout":
        sc.pp.scale(work, max_value=10)
    sc.pp.pca(work, n_comps=n_pcs)
    sc.pp.neighbors(work, n_neighbors=n_neighbors, n_pcs=n_pcs, random_state=cfg.random_state)
    sc.tl.leiden(work, resolution=cfg.leiden_resolution, random_state=cfg.random_state)
    sc.tl.umap(work, random_state=cfg.random_state)
    embedding = np.asarray(work.obsm["X_umap"], dtype=np.float32).copy()
    labels = work.obs["leiden"].astype("category").cat.codes.to_numpy()
    return embedding, labels


def _build_consensus_graph(
    observed_x: np.ndarray,
    graph_x: np.ndarray,
    spatial: np.ndarray,
    cfg: _ResolvedConfig,
) -> tuple[np.ndarray, np.ndarray]:
    observed_embedding, labels = _precluster(observed_x, cfg)
    if cfg.mode == "dropout":
        graph_embedding = observed_embedding
    else:
        graph_embedding, _ = _precluster(graph_x, cfg)

    n_obs = observed_x.shape[0]
    knn = max(2, min(cfg.n_neighbors, n_obs))
    expression_neighbors = NearestNeighbors(
        n_neighbors=knn, n_jobs=cfg.cpu_threads
    ).fit(observed_embedding).kneighbors(observed_embedding, return_distance=False)
    rows = np.repeat(np.arange(n_obs, dtype=np.int64), knn)
    cols = expression_neighbors.ravel().astype(np.int64)
    keep = (rows != cols) & (labels[rows] == labels[cols])
    row_parts, col_parts = [rows[keep]], [cols[keep]]

    outlier_parts: list[np.ndarray] = []
    for label in np.unique(labels):
        cells = np.flatnonzero(labels == label)
        if len(cells) < 4:
            continue
        lof_neighbors = min(len(cells) - 1, max(2, len(cells) // 3))
        flags = LocalOutlierFactor(
            n_neighbors=lof_neighbors,
            contamination=cfg.lof_contamination,
            n_jobs=cfg.cpu_threads,
        ).fit_predict(graph_embedding[cells])
        if np.any(flags == -1):
            outlier_parts.append(cells[flags == -1])

    if outlier_parts:
        outliers = np.concatenate(outlier_parts)
        spatial_neighbors = NearestNeighbors(
            n_neighbors=knn, n_jobs=cfg.cpu_threads
        ).fit(spatial).kneighbors(spatial[outliers], return_distance=False)
        rows = np.repeat(outliers, knn).astype(np.int64)
        cols = spatial_neighbors.ravel().astype(np.int64)
        keep = (rows != cols) & (labels[rows] == labels[cols])
        row_parts.append(rows[keep])
        col_parts.append(cols[keep])

    rows, cols = np.concatenate(row_parts), np.concatenate(col_parts)
    if not len(rows):
        raise RuntimeError("Consensus graph contains no edges; reduce leiden_resolution or n_neighbors")
    keys = np.unique(np.concatenate((rows * n_obs + cols, cols * n_obs + rows)))
    edge_index = np.vstack((keys // n_obs, keys % n_obs)).astype(np.int64)
    return labels, edge_index


def _checkpoint_forward(
    model: _GraphAutoEncoder, x: torch.Tensor, edge_index: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = checkpoint(
        lambda value: model.norm1(model.gc1(value, edge_index)), x, use_reentrant=False
    )
    latent = checkpoint(
        lambda value: model.norm2(model.gc2(value, edge_index)), hidden, use_reentrant=False
    )
    reconstructed = checkpoint(
        lambda value: F.relu(model.fc2(F.relu(model.fc1(value)))), latent, use_reentrant=False
    )
    return reconstructed, model.classifier(latent)


def _predict_columns(
    model: _GraphAutoEncoder,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    columns: torch.Tensor,
) -> torch.Tensor:
    hidden = model.norm1(model.gc1(x, edge_index))
    latent = model.norm2(model.gc2(hidden, edge_index))
    decoded = F.relu(model.fc1(latent))
    weight = model.fc2.weight.index_select(0, columns)
    bias = model.fc2.bias.index_select(0, columns) if model.fc2.bias is not None else None
    return F.relu(F.linear(decoded, weight, bias))


def _train_graph_autoencoder(
    graph_x: np.ndarray,
    ae_latent: np.ndarray,
    labels: np.ndarray,
    edge_array: np.ndarray,
    output_indices: np.ndarray,
    cfg: _ResolvedConfig,
) -> np.ndarray:
    edge_index = torch.as_tensor(edge_array, dtype=torch.long, device=cfg.device)
    edge_index, _ = add_self_loops(edge_index, num_nodes=graph_x.shape[0])
    x = torch.as_tensor(graph_x, dtype=torch.float32, device=cfg.device)
    n_clusters = max(1, len(np.unique(labels)))
    pseudo = KMeans(
        n_clusters=n_clusters, n_init=10, random_state=cfg.random_state
    ).fit_predict(ae_latent)
    pseudo = torch.as_tensor(pseudo, dtype=torch.long, device=cfg.device)
    model = _GraphAutoEncoder(graph_x.shape[1], cfg.latent_dim, n_clusters).to(cfg.device)
    if cfg.low_memory:
        model.gc1.decomposed_layers = cfg.decomposed_layers
        model.gc2.decomposed_layers = 1
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.999))
    for epoch in range(cfg.gae_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if cfg.low_memory:
            prediction, logits = _checkpoint_forward(model, x, edge_index)
        else:
            _, prediction, logits = model(x, edge_index)
        reconstruction, mse, cosine = _masked_reconstruction_loss(
            prediction, x, cfg.mse_weight, masked=cfg.mode == "dropout"
        )
        classification = F.cross_entropy(logits, pseudo)
        loss = reconstruction + classification
        loss.backward()
        optimizer.step()
        if cfg.verbose and (epoch == 0 or (epoch + 1) % 500 == 0 or epoch + 1 == cfg.gae_epochs):
            LOGGER.info(
                "GAE %d/%d: loss=%.4f, mse=%.4f, cosine=%.4f, class=%.4f",
                epoch + 1,
                cfg.gae_epochs,
                loss.item(),
                mse.item(),
                cosine.item(),
                classification.item(),
            )
    model.eval()
    columns = torch.as_tensor(output_indices, dtype=torch.long, device=cfg.device)
    with torch.no_grad():
        if cfg.low_memory:
            prediction = _predict_columns(model, x, edge_index, columns)
        else:
            prediction = model(x, edge_index)[1].index_select(1, columns)
        result = prediction.cpu().numpy().astype(np.float32, copy=True)
    return np.maximum(result, 0)


def _copy_axis_metadata(source: ad.AnnData, result: ad.AnnData) -> None:
    result.uns = copy.deepcopy(source.uns)
    for key, value in source.obsm.items():
        result.obsm[key] = value.copy() if hasattr(value, "copy") else copy.deepcopy(value)
    for key, value in source.obsp.items():
        result.obsp[key] = value.copy() if hasattr(value, "copy") else copy.deepcopy(value)


def _release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Dropout branch (kept independent; source: code/NEST_dropout.py)
# ---------------------------------------------------------------------------


class _DropoutAutoEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_size: int) -> None:
        super().__init__()
        hidden_size = max(2, input_dim // 2)
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, latent_size)
        self.layer_norm2 = nn.LayerNorm(latent_size)
        self.fc3 = nn.Linear(latent_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, input_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.layer_norm2(self.fc2(self.layer_norm1(self.fc1(x))))
        reconstructed = F.relu(self.fc4(F.relu(self.fc3(latent))))
        return latent, reconstructed


class _DropoutGraphAutoEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_size: int, n_clusters: int) -> None:
        super().__init__()
        hidden_size = max(2, input_dim // 2)
        self.gc1 = GCNConv(input_dim, hidden_size)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.gc2 = GCNConv(hidden_size, latent_size)
        self.layer_norm2 = nn.LayerNorm(latent_size)
        self.fc1 = nn.Linear(latent_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_dim)
        self.classifier = nn.Linear(latent_size, n_clusters)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.layer_norm1(self.gc1(x, edge_index))
        latent = self.layer_norm2(self.gc2(hidden, edge_index))
        reconstructed = F.relu(self.fc2(F.relu(self.fc1(latent))))
        return latent, reconstructed, self.classifier(latent)


def _dropout_dense(x: object) -> np.ndarray:
    """Return an owning array so dropout preprocessing cannot mutate its input."""
    if sp.issparse(x):
        x = x.toarray()
    return np.array(x, dtype=np.float32, copy=True)


def _dropout_train_autoencoder(
    adata_st: ad.AnnData, cfg: _ResolvedConfig
) -> np.ndarray:
    x = torch.tensor(_dropout_dense(adata_st.X), dtype=torch.float32, device=cfg.device)
    model = _DropoutAutoEncoder(x.shape[1], cfg.latent_dim).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.999))
    observed = x != 0
    for epoch in range(cfg.ae_epochs):
        model.train()
        optimizer.zero_grad()
        latent, reconstructed = model(x)
        loss_mse = F.mse_loss(reconstructed[observed], x[observed])
        loss_cos = 1 - F.cosine_similarity(
            reconstructed * observed, x * observed, dim=1
        ).mean()
        loss = loss_mse + loss_cos
        loss.backward()
        optimizer.step()
        if cfg.verbose and (epoch == 0 or (epoch + 1) % 500 == 0):
            LOGGER.info(
                "dropout AE %d/%d: loss=%.4f, mse=%.4f, cosine=%.4f",
                epoch + 1, cfg.ae_epochs, loss.item(), loss_mse.item(), loss_cos.item(),
            )
    model.eval()
    with torch.no_grad():
        latent = model(x)[0].cpu().numpy().astype(np.float32, copy=True)
    return latent


def _dropout_precluster(
    adata_st: ad.AnnData, cfg: _ResolvedConfig
) -> tuple[np.ndarray, np.ndarray]:
    # The copy is part of the original dropout workflow: scaling is used only
    # for preliminary clustering and must never overwrite reconstruction input.
    work = adata_st.copy()
    n_components = min(cfg.n_pcs, min(work.n_obs, work.n_vars) - 1)
    sc.pp.scale(work, max_value=10)
    sc.pp.pca(work, n_comps=n_components)
    sc.pp.neighbors(work, n_neighbors=cfg.n_neighbors, n_pcs=n_components)
    sc.tl.leiden(work, resolution=1.0)
    sc.tl.umap(work)
    labels = work.obs["leiden"].astype("category").cat.codes.to_numpy()
    return np.asarray(work.obsm["X_umap"], dtype=np.float32).copy(), labels


def _dropout_create_cell_network(
    adata_st: ad.AnnData, cfg: _ResolvedConfig
) -> tuple[np.ndarray, np.ndarray]:
    embedding, labels = _dropout_precluster(adata_st, cfg)
    adata_st.obs["leiden"] = labels
    outlier_parts: list[np.ndarray] = []
    for cluster in np.unique(labels):
        cells = np.flatnonzero(labels == cluster)
        if len(cells) < 3:
            continue
        lof_neighbors = max(1, min(len(cells) - 1, len(cells) // 3))
        prediction = LocalOutlierFactor(
            n_neighbors=lof_neighbors, contamination=0.3
        ).fit_predict(embedding[cells])
        outlier_parts.append(cells[prediction == -1])
    outliers = (
        np.concatenate(outlier_parts)
        if outlier_parts
        else np.empty(0, dtype=np.int64)
    )

    n_obs = adata_st.n_obs
    k = min(cfg.n_neighbors, n_obs)
    expression_neighbors = NearestNeighbors(n_neighbors=k).fit(embedding).kneighbors(
        embedding, return_distance=False
    )
    graph = np.zeros((n_obs, n_obs), dtype=np.float32)
    for i in range(n_obs):
        for neighbor in expression_neighbors[i]:
            if i < neighbor and labels[i] == labels[neighbor]:
                graph[i, neighbor] = 1
                graph[neighbor, i] = 1

    spatial = np.asarray(adata_st.obsm[cfg.spatial_key], dtype=np.float32)
    spatial_neighbors = NearestNeighbors(n_neighbors=k).fit(spatial).kneighbors(
        spatial, return_distance=False
    )
    for i in outliers:
        neighbors = spatial_neighbors[i]
        neighbors = neighbors[(neighbors != i) & (labels[neighbors] == labels[i])]
        for neighbor in neighbors:
            graph[i, neighbor] = 1
            graph[neighbor, i] = 1
    return graph, labels


def _dropout_train_graph_autoencoder(
    adata_st: ad.AnnData,
    graph: np.ndarray,
    latent_map: np.ndarray,
    cfg: _ResolvedConfig,
) -> np.ndarray:
    edge_index, _ = dense_to_sparse(torch.tensor(graph, dtype=torch.float32))
    edge_index, _ = add_self_loops(edge_index, num_nodes=graph.shape[0])
    x = torch.tensor(_dropout_dense(adata_st.X), dtype=torch.float32, device=cfg.device)
    data = Data(x=x, edge_index=edge_index.to(cfg.device))
    n_clusters = len(np.unique(adata_st.obs["leiden"]))
    pseudo_labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(
        latent_map
    )
    pseudo_labels = torch.tensor(pseudo_labels, dtype=torch.long, device=cfg.device)
    model = _DropoutGraphAutoEncoder(x.shape[1], cfg.latent_dim, n_clusters).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.999))
    observed = data.x != 0
    for epoch in range(cfg.gae_epochs):
        model.train()
        optimizer.zero_grad()
        _, reconstructed, logits = model(data.x, data.edge_index)
        loss_mse = F.mse_loss(reconstructed[observed], data.x[observed])
        loss_cos = 1 - F.cosine_similarity(
            reconstructed * observed, data.x * observed, dim=1
        ).mean()
        loss_cls = F.cross_entropy(logits, pseudo_labels)
        loss = loss_mse + loss_cos + loss_cls
        loss.backward()
        optimizer.step()
        if cfg.verbose and (epoch == 0 or (epoch + 1) % 500 == 0):
            LOGGER.info(
                "dropout GAE %d/%d: loss=%.4f, mse=%.4f, cosine=%.4f, class=%.4f",
                epoch + 1, cfg.gae_epochs, loss.item(), loss_mse.item(),
                loss_cos.item(), loss_cls.item(),
            )
    model.eval()
    with torch.no_grad():
        prediction = model(data.x, data.edge_index)[1].cpu().numpy()
    return np.asarray(prediction, dtype=np.float32)


def _run_dropout_branch(
    adata_st: ad.AnnData,
    fill_mask: np.ndarray | None,
    cfg: _ResolvedConfig,
) -> tuple[ad.AnnData, np.ndarray, np.ndarray]:
    original = _dropout_dense(adata_st.X)
    working = adata_st.copy()
    working.X = original.copy()
    latent = _dropout_train_autoencoder(working, cfg)
    working.obsm["X_latent"] = latent
    graph, _ = _dropout_create_cell_network(working, cfg)
    prediction = _dropout_train_graph_autoencoder(working, graph, latent, cfg)

    if fill_mask is not None:
        mask = np.asarray(fill_mask, dtype=bool)
        if mask.shape != original.shape:
            raise ValueError(f"fill_mask must have shape {original.shape}, got {mask.shape}")
    elif cfg.fill_zeros_only:
        mask = original == 0
    else:
        mask = np.ones_like(original, dtype=bool)
    output = original.copy()
    output[mask] = prediction[mask]
    result = adata_st.copy()
    result.X = output
    result.obsm["X_nest_latent"] = latent
    result.uns["nest_branch"] = "dropout:NEST_dropout.py:fixed_defaults"
    edge_index = np.vstack(np.nonzero(graph)).astype(np.int64)
    return result, latent, edge_index


class NEST:
    """NEST estimator dispatching to independent task implementations.

    Use :meth:`fit_transform` for reconstruction. The fitted instance records
    resolved parameters in ``params_`` and the consensus graph in
    ``edge_index_``.
    """

    def __init__(self, config: NESTConfig | None = None, *, mode: Mode | None = None, **kwargs: object) -> None:
        if config is not None and (mode is not None or kwargs):
            raise TypeError("Pass either config or mode/keyword options, not both")
        self.config = config if config is not None else NESTConfig(mode=mode or "missing_gene", **kwargs)
        self.params_: _ResolvedConfig | None = None
        self.edge_index_: np.ndarray | None = None
        self.target_genes_: pd.Index | None = None

    def fit_transform(
        self,
        adata_st: ad.AnnData,
        adata_sc: ad.AnnData | None = None,
        *,
        target_genes: Sequence[str] | None = None,
        fill_mask: np.ndarray | None = None,
    ) -> ad.AnnData:
        """Reconstruct an ST expression matrix and return a new AnnData object.

        ``target_genes`` is used only in ``missing_gene`` mode. If omitted, all
        genes present in ``adata_sc`` but absent from ``adata_st`` are completed.
        Supplying measured genes is supported for gene-wise cross-validation;
        those genes are excluded from training and replaced by predictions.

        In ``dropout`` mode, ``fill_mask`` optionally specifies entries to
        replace. Otherwise input zeros are replaced when ``fill_zeros_only`` is
        true, or the complete reconstructed matrix is returned.
        """

        _check_expression(adata_st, "adata_st")
        spatial = _spatial_coordinates(adata_st, self.config.spatial_key)
        _set_seed(self.config.random_state)
        if self.config.verbose:
            logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

        if self.config.mode == "missing_gene":
            if adata_sc is None:
                raise ValueError("adata_sc is required when mode='missing_gene'")
            if fill_mask is not None:
                raise ValueError("fill_mask is available only in mode='dropout'")
            return self._fit_missing_gene(adata_st, adata_sc, target_genes, spatial)
        if adata_sc is not None:
            raise ValueError("adata_sc must not be supplied when mode='dropout'")
        if target_genes is not None:
            raise ValueError("target_genes is available only in mode='missing_gene'")
        return self._fit_dropout(adata_st, fill_mask, spatial)

    def _fit_missing_gene(
        self,
        adata_st: ad.AnnData,
        adata_sc: ad.AnnData,
        target_genes: Sequence[str] | None,
        spatial: np.ndarray,
    ) -> ad.AnnData:
        _check_expression(adata_sc, "adata_sc")
        reference_genes = pd.Index(adata_sc.var_names)
        st_genes = pd.Index(adata_st.var_names)
        if target_genes is None:
            targets = reference_genes[~reference_genes.isin(st_genes)]
        else:
            targets = pd.Index(list(dict.fromkeys(target_genes)))
            missing = targets[~targets.isin(reference_genes)]
            if len(missing):
                raise ValueError(f"target_genes absent from adata_sc: {missing[:5].tolist()}")
        if not len(targets):
            raise ValueError("No target genes were selected for missing-gene reconstruction")
        common = st_genes[st_genes.isin(reference_genes) & ~st_genes.isin(targets)]
        if len(common) < 2:
            raise ValueError("At least two non-target genes shared by adata_st and adata_sc are required")
        st_train = _dense32(adata_st[:, common].X)
        cfg = _resolve_config(self.config, st_train, adata_sc.n_obs)
        self.params_ = cfg
        self.target_genes_ = targets
        _report_configuration(cfg)

        model, st_latent = _train_autoencoder(st_train, cfg)
        mapped = _map_reference(model, adata_sc, common, st_latent, cfg)
        del model
        _release_memory()
        common_indices = reference_genes.get_indexer(common)
        _correct_gene_domain(mapped, st_train, common_indices, cfg)
        labels, edges = _build_consensus_graph(st_train, mapped, spatial, cfg)
        self.edge_index_ = edges
        output_indices = reference_genes.get_indexer(targets)
        predictions = _train_graph_autoencoder(
            mapped, st_latent, labels, edges, output_indices, cfg
        )
        result = _missing_gene_result(adata_st, adata_sc, targets, predictions)
        result.obsm["X_nest_latent"] = st_latent
        result.uns["nest"] = _result_metadata(cfg, targets.tolist())
        _release_memory()
        return result

    def _fit_dropout(
        self,
        adata_st: ad.AnnData,
        fill_mask: np.ndarray | None,
        spatial: np.ndarray,
    ) -> ad.AnnData:
        # Do not route dropout through the missing-gene implementation.  This
        # branch is the original NEST_dropout.py workflow with its own models,
        # scaled-copy preclustering and non-zero-only reconstruction losses.
        x = _dropout_dense(adata_st.X)
        cfg = _resolve_dropout_config(self.config, x)
        self.params_ = cfg
        self.target_genes_ = pd.Index(adata_st.var_names)
        _report_configuration(cfg)
        result, latent, edges = _run_dropout_branch(adata_st, fill_mask, cfg)
        self.edge_index_ = edges
        result.uns["nest"] = _result_metadata(cfg, adata_st.var_names.tolist())
        result.uns["nest"]["implementation"] = "independent_dropout_fixed_defaults"
        _release_memory()
        return result


def _missing_gene_result(
    adata_st: ad.AnnData,
    adata_sc: ad.AnnData,
    targets: pd.Index,
    predictions: np.ndarray,
) -> ad.AnnData:
    st_genes = pd.Index(adata_st.var_names)
    appended = targets[~targets.isin(st_genes)]
    output_genes = st_genes.append(appended)
    output = np.empty((adata_st.n_obs, len(output_genes)), dtype=np.float32)
    output[:, : adata_st.n_vars] = _dense32(adata_st.X)
    for j, gene in enumerate(targets):
        if gene in st_genes:
            output[:, st_genes.get_loc(gene)] = predictions[:, j]
        else:
            output[:, adata_st.n_vars + appended.get_loc(gene)] = predictions[:, j]
    var = pd.concat(
        [adata_st.var.copy(), adata_sc.var.loc[appended].copy()], axis=0
    ).reindex(output_genes)
    result = ad.AnnData(X=output, obs=adata_st.obs.copy(), var=var)
    _copy_axis_metadata(adata_st, result)
    return result


def _result_metadata(cfg: _ResolvedConfig, target_genes: list[str]) -> dict[str, object]:
    return {
        "mode": cfg.mode,
        "target_genes": target_genes,
        "zero_rate": cfg.zero_rate,
        "n_neighbors": cfg.n_neighbors,
        "mapping_neighbors": cfg.mapping_neighbors,
        "gene_neighbors": cfg.gene_neighbors,
        "latent_dim": cfg.latent_dim,
        "n_pcs": cfg.n_pcs,
        "learning_rate": cfg.learning_rate,
        "ae_epochs": cfg.ae_epochs,
        "gae_epochs": cfg.gae_epochs,
        "mse_weight": cfg.mse_weight,
        "random_state": cfg.random_state,
    }


def _report_configuration(cfg: _ResolvedConfig) -> None:
    if cfg.verbose:
        LOGGER.info(
            "mode=%s, zero_rate=%.3f, k=%d, latent=%d, n_pcs=%d, device=%s",
            cfg.mode,
            cfg.zero_rate,
            cfg.n_neighbors,
            cfg.latent_dim,
            cfg.n_pcs,
            cfg.device,
        )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reconstruct(
    adata_st: ad.AnnData,
    *,
    mode: Mode,
    adata_sc: ad.AnnData | None = None,
    target_genes: Sequence[str] | None = None,
    fill_mask: np.ndarray | None = None,
    config: NESTConfig | None = None,
    **kwargs: object,
) -> ad.AnnData:
    """Functional convenience wrapper around :class:`NEST`.

    Examples
    --------
    ``reconstruct(st, mode="missing_gene", adata_sc=reference)``

    ``reconstruct(st, mode="dropout")``
    """

    if config is not None:
        if kwargs:
            raise TypeError("Do not combine config with keyword configuration overrides")
        if config.mode != mode:
            config = replace(config, mode=mode)
        estimator = NEST(config)
    else:
        estimator = NEST(mode=mode, **kwargs)
    return estimator.fit_transform(
        adata_st, adata_sc, target_genes=target_genes, fill_mask=fill_mask
    )

