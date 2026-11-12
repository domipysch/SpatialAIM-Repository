"""
Reconstruction of one K: how well each spot's measured expression is predicted
by the mean expression of its mapped state (cosine similarity).
"""

from __future__ import annotations
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
from spatialaim.utils import box_stats, to_dense
from .criterion import Criterion, finite, load_json

logger = logging.getLogger(__name__)

# Per soft/hard x raw/norm combo, the box-plot statistics (spatialaim.utils.box_stats)
# of the per-gene and per-spot cosine similarities; hard-raw also carries the
# chance level of its medians under "null".
COSSIM_METRICS_JSON = "cossim_metrics.json"
# What older outputs wrote instead: the box-plot statistics alone, and the
# hard-raw medians with their null in a CSV.
COSSIM_BOXPLOTS_JSON_LEGACY = "cossim_boxplots.json"
COSSIM_SUMMARY_CSV_LEGACY = "cossim_summary.csv"


N_PERM_RECONSTRUCTION = 20  # label shuffles for the null


class Reconstruction(Criterion):
    """Cosine similarity of predicted vs. measured spot expression."""

    key = "recon"
    label = "Reconstruction fidelity"
    short = "Reconstruction"
    unit = "cosine similarity"
    filename = COSSIM_METRICS_JSON
    columns = ("cossim_hard_raw_spot", "cossim_hard_raw_gene")
    curve_labels = ("raw · spot", "raw · gene")

    n_perm: int = N_PERM_RECONSTRUCTION
    seed: int = 0

    def analyse(self, adata_sc, adata_st, start_cluster_to_state) -> dict:
        """Cosine similarity of predicted vs. measured spot expression, per gene
        and per spot, for soft / hard mapping x raw / lognorm expression.

        The prediction is the mapping times the states' mean expression. Per
        variant the box plot statistics (for the GUI), and for hard-raw, whose
        medians the K selection uses, their chance level from ``n_perm`` label
        shuffles under ``"null"``.
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
        metrics: dict[str, dict] = {}
        rng = np.random.default_rng(self.seed)
        for label, (m, centroids, st_ref, is_hard) in combos.items():
            pred = centroids[m] if is_hard else _predict_expression(m, centroids)
            pred_adata = ad.AnnData(
                X=pred.T.astype(np.float32),
                obs=pd.DataFrame(index=shared_genes),
                var=pd.DataFrame(index=spot_names),
            )
            result = _compute_cossim(st_ref, pred_adata)
            metrics[label] = {
                "per_gene": box_stats(list(result.per_gene.values())),
                "per_spot": box_stats(list(result.per_spot.values())),
            }
            if label != "hard-raw":
                continue  # only hard-raw is used by the K selection
            metrics[label]["null"] = _cossim_null_medians(
                to_dense(st_ref), centroids, m, n_perm=self.n_perm, rng=rng
            )
        return metrics

    def collect(self, data_dir: Path) -> dict[str, float]:
        """The hard-raw median cosine per spot and per gene, and their nulls
        (from the ``cossim_summary.csv`` of older outputs if needed)."""
        cell = _read_cossim(Path(data_dir))
        return {
            "cossim_hard_raw_spot": cell("median_spot"),
            "cossim_hard_raw_gene": cell("median_gene"),
            "cossim_hard_raw_spot_null": cell("median_spot_null"),
            "cossim_hard_raw_gene_null": cell("median_gene_null"),
        }


def cosine_similarity(A: np.ndarray, B: np.ndarray, axis: int) -> np.ndarray:
    """Cosine similarity of ``A`` and ``B`` per column (``axis=0``) or per row
    (``axis=1``); 0 where a vector is all zero."""
    dot = np.sum(A * B, axis=axis)
    denom = np.linalg.norm(A, axis=axis) * np.linalg.norm(B, axis=axis)
    with np.errstate(divide="ignore", invalid="ignore"):
        cs = np.where(denom == 0.0, 0.0, dot / denom)
    return np.clip(cs, -1.0, 1.0)


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


def _compute_cossim(st: ad.AnnData, prediction: ad.AnnData) -> _CossimResult:
    """Cosine similarity between the measured ST data ``st`` (spots x genes)
    and ``prediction`` (genes x spots), on their shared genes."""
    measured, predicted = _align(st, prediction.transpose())
    M, P = to_dense(measured), to_dense(predicted)
    return _CossimResult(
        per_gene=dict(
            zip(measured.var_names, cosine_similarity(M, P, axis=0).tolist())
        ),
        per_spot=dict(
            zip(measured.obs_names, cosine_similarity(M, P, axis=1).tolist())
        ),
    )


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


def _read_cossim(data_dir: Path):
    """A lookup ``median_{spot,gene}[_null] -> float`` into the ``hard-raw``
    entry of ``cossim_metrics.json`` (or of the ``cossim_summary.csv`` older
    outputs wrote); NaN for a missing file or entry."""
    hard_raw = load_json(data_dir / COSSIM_METRICS_JSON).get("hard-raw")
    if hard_raw is None:
        legacy = data_dir / COSSIM_SUMMARY_CSV_LEGACY
        table = pd.read_csv(legacy, index_col=0) if legacy.exists() else None
        row = (
            {}
            if table is None or "hard-raw" not in table.index
            else table.loc["hard-raw"]
        )
        return lambda column: finite(row.get(column))

    def cell(column: str) -> float:
        side = column.removeprefix("median_").removesuffix("_null")
        if column.endswith("_null"):
            return finite((hard_raw.get("null") or {}).get(f"median_{side}"))
        return finite((hard_raw.get(f"per_{side}") or {}).get("median"))

    return cell
