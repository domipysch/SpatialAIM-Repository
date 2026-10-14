"""
Reconstruction of one K: how well each spot's measured expression is predicted
by the mean expression of its mapped state (cosine similarity).
"""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import anndata as ad
import pandas as pd
from spatialaim.adata_schema import (
    OBS_MAPPING_HARD,
    OBSM_LOGNORM_SHARED_GENES,
    OBSM_MAPPING_SOFT,
    UNS_SHARED_GENES,
    UNS_START_CLUSTER_EXPR_SUMS_SHARED_GENES,
    UNS_START_CLUSTER_EXPR_SUMS_SHARED_GENES_NORM,
    UNS_START_CLUSTER_SIZES,
)
from spatialaim.utils import to_dense
from ..metrics.cossim import cosine_similarity

logger = logging.getLogger(__name__)


N_PERM_RECONSTRUCTION = 20  # label shuffles for the null


def analyse_reconstruction(
    adata_sc: ad.AnnData,
    adata_st: ad.AnnData,
    start_cluster_to_state: np.ndarray,
    output_data_dir: Path,
    n_perm: int = N_PERM_RECONSTRUCTION,
    seed: int = 0,
) -> None:
    """Cosine similarity of predicted vs. measured spot expression, per gene
    and per spot, for soft / hard mapping x raw / lognorm expression.

    The prediction is the mapping times the states' mean expression. All four
    variants go to ``cossim/*.json`` (for the GUI); ``cossim_summary.csv`` holds
    the hard-raw medians, which the K selection uses, with their chance level
    from ``n_perm`` label shuffles.
    """
    P = adata_st.obsm[OBSM_MAPPING_SOFT]
    spot_states_hard = adata_st.obs[OBS_MAPPING_HARD].to_numpy()
    k = P.shape[1]

    shared_genes = list(adata_sc.uns[UNS_SHARED_GENES])
    sizes = adata_sc.uns[UNS_START_CLUSTER_SIZES]

    # Measured ST expression on the shared genes, raw and lognorm.
    adata_st_shared = adata_st[:, shared_genes]
    adata_st_norm = ad.AnnData(
        X=to_dense(adata_st.obsm[OBSM_LOGNORM_SHARED_GENES]),
        obs=pd.DataFrame(index=adata_st.obs_names),
        var=pd.DataFrame(index=shared_genes),
    )

    centroids_raw = _assemble_state_centroids(
        start_cluster_to_state,
        k,
        adata_sc.uns[UNS_START_CLUSTER_EXPR_SUMS_SHARED_GENES],
        sizes,
    )
    centroids_norm = _assemble_state_centroids(
        start_cluster_to_state,
        k,
        adata_sc.uns[UNS_START_CLUSTER_EXPR_SUMS_SHARED_GENES_NORM],
        sizes,
    )

    # Hard: each spot gets its state's centroid; soft: mapping @ centroids.
    spot_names = adata_st.obs_names.tolist()
    combos = {
        "soft-raw": (P, centroids_raw, adata_st_shared, False),
        "hard-raw": (spot_states_hard, centroids_raw, adata_st_shared, True),
        "soft-norm": (P, centroids_norm, adata_st_norm, False),
        "hard-norm": (spot_states_hard, centroids_norm, adata_st_norm, True),
    }
    cossim_summary: dict[str, dict] = {}
    cossim_dir = output_data_dir / "cossim"
    rng = np.random.default_rng(seed)
    for label, (m, centroids, st_ref, is_hard) in combos.items():
        pred = centroids[m] if is_hard else _predict_expression(m, centroids)
        pred_adata = ad.AnnData(
            X=pred.T.astype(np.float32),
            obs=pd.DataFrame(index=shared_genes),
            var=pd.DataFrame(index=spot_names),
        )
        result = _compute_and_save_cossim(
            st_ref, pred_adata, cossim_dir, suffix=f"-{label}"
        )
        if label != "hard-raw":
            continue  # only hard-raw is used by the K selection
        null = _cossim_null_medians(
            to_dense(st_ref), centroids, m, n_perm=n_perm, rng=rng
        )
        cossim_summary[label] = {
            "median_gene": result.median_gene,
            "median_spot": result.median_spot,
            "median_gene_null": null["median_gene"],
            "median_spot_null": null["median_spot"],
        }

    pd.DataFrame(cossim_summary).T.to_csv(output_data_dir / "cossim_summary.csv")


def _predict_expression(mapping, centroids) -> np.ndarray:
    """Predicted expression: ``mapping`` (spots x states) @ ``centroids``."""
    return np.asarray(mapping) @ np.asarray(centroids)


def _assemble_state_centroids(
    start_cluster_to_state: np.ndarray,
    k: int,
    expr_sums: np.ndarray,
    sizes: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Mean expression per state (k x genes), pooled from the start
    clusters' summed expression ``expr_sums`` and cell counts ``sizes``."""
    state_sums = np.zeros((k, expr_sums.shape[1]), dtype=expr_sums.dtype)
    np.add.at(state_sums, start_cluster_to_state, expr_sums)
    state_sizes = np.zeros(k, dtype=sizes.dtype)
    np.add.at(state_sizes, start_cluster_to_state, sizes)
    return state_sums / (state_sizes[:, None] + eps)


@dataclass
class _CossimResult:
    """Cosine similarity per gene and per spot, keyed by name."""

    per_gene: dict[str, float]
    per_spot: dict[str, float]

    @property
    def median_gene(self) -> float | None:
        return _median(self.per_gene)

    @property
    def median_spot(self) -> float | None:
        return _median(self.per_spot)


def _compute_and_save_cossim(
    st: ad.AnnData,
    prediction: ad.AnnData,
    output_folder: Path,
    suffix: str = "",
) -> _CossimResult:
    """Cosine similarity between the measured ST data ``st`` (spots x genes)
    and ``prediction`` (genes x spots), on their shared genes. Writes
    ``cossim-per-gene{suffix}.json`` and ``cossim-per-spot{suffix}.json`` to
    ``output_folder``, each ``{"median": ..., "values": {name: cossim}}``."""
    measured, predicted = _align(st, prediction.transpose())
    M, P = to_dense(measured), to_dense(predicted)
    result = _CossimResult(
        per_gene=dict(
            zip(measured.var_names, cosine_similarity(M, P, axis=0).tolist())
        ),
        per_spot=dict(
            zip(measured.obs_names, cosine_similarity(M, P, axis=1).tolist())
        ),
    )
    output_folder.mkdir(parents=True, exist_ok=True)
    _write_json(output_folder / f"cossim-per-gene{suffix}.json", result.per_gene)
    _write_json(output_folder / f"cossim-per-spot{suffix}.json", result.per_spot)
    return result


def _cossim_null_medians(
    measured: np.ndarray,
    centroids: np.ndarray,
    labels: np.ndarray,
    *,
    n_perm: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Chance level of the reconstruction cosine similarity.

    ``n_perm`` times: shuffle the spots' state ``labels``, predict each spot as
    its state's centroid, and take the median cosine over spots and over genes
    against ``measured`` (spots x genes). Returns the means of these medians as
    ``{"median_spot", "median_gene"}`` (NaN for ``n_perm <= 0``).
    """
    measured = np.asarray(measured)
    labels = np.asarray(labels)
    per_spot, per_gene = [], []
    for _ in range(n_perm):
        predicted = np.asarray(centroids)[rng.permutation(labels)]
        per_spot.append(
            float(np.median(cosine_similarity(measured, predicted, axis=1)))
        )
        per_gene.append(
            float(np.median(cosine_similarity(measured, predicted, axis=0)))
        )
    if not per_spot:
        return {"median_spot": float("nan"), "median_gene": float("nan")}
    return {
        "median_spot": float(np.mean(per_spot)),
        "median_gene": float(np.mean(per_gene)),
    }


def _align(
    measured: ad.AnnData, predicted: ad.AnnData
) -> tuple[ad.AnnData, ad.AnnData]:
    """Both (spots x genes) restricted to their shared genes, in the order of
    ``measured``. Raises if no gene is shared or the spots differ."""
    predicted_genes = set(predicted.var_names)
    shared_genes = [g for g in measured.var_names if g in predicted_genes]
    if not shared_genes:
        raise ValueError("No shared genes between ground-truth ST and prediction.")
    if set(measured.obs_names) != set(predicted.obs_names):
        raise ValueError(
            "Spot identities differ between ground-truth ST and prediction "
            f"({measured.n_obs} vs {predicted.n_obs} spots)."
        )
    return measured[:, shared_genes], predicted[measured.obs_names, shared_genes]


def _median(values: dict[str, float]) -> float | None:
    return float(np.median(list(values.values()))) if values else None


def _write_json(path: Path, values: dict[str, float]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump({"median": _median(values), "values": values}, f, indent=4)
