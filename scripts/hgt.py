"""Horizontal gene transfer (HGT) helper utilities."""

from __future__ import annotations

from collections import defaultdict
from functools import partial
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from statsmodels.nonparametric.smoothers_lowess import lowess

PathLike = Union[str, Path]

# Ordered list of habitat clusters referenced throughout the manuscript.
HABITAT_CLUSTERS: Sequence[str] = [
    "HTS",    "OLL",    "OIL",    "OHL",    "POL",    "EST",    "FRW",    "TEH",
    "TER",    "TEG",    "ADC",    "HOR",    "HUV",    "HSS",    "BLD",    "HET",
    "WT1",    "ANI",    "MOG",    "RMG",    "CKG",    "APG",    "HG1",    "MKG",
    "PG1",    "PG2",    "WT2",    "HG2",    "HG3",    "HG4",    "HG5",    "HG6",
    "HG7",    "NBG",    "IFG",    "CDG",    "BG1",    "BG2",    "BG3",    "HRG",
]

_HGT_RESULTS_CACHE: dict[Path, list[list[str]]] = {}


def _load_hgt_results(hgt_result_filename: PathLike, use_cache: bool = True) -> list[list[str]]:
    """Read HGT genome pair table (optionally cached) and return split rows."""

    path = Path(hgt_result_filename).resolve()
    if use_cache and path in _HGT_RESULTS_CACHE:
        return _HGT_RESULTS_CACHE[path]

    with path.open() as fp:
        next(fp, None)
        rows = [line.rstrip("\n").split("\t") for line in fp if line.strip()]

    if use_cache:
        _HGT_RESULTS_CACHE[path] = rows

    return rows


def _sorted_pair(hc1: str, hc2: str) -> tuple[str, str]:
    return tuple(sorted((hc1, hc2)))


def filter_cluster(cluster_to_magcount_filename: PathLike, mag_count_threshold: int):
    """Return MAG counts and valid cluster set based on coverage threshold."""

    path = Path(cluster_to_magcount_filename)
    valid_clusters = set()
    cluster_to_mag_count = {}

    with path.open() as fp:
        for line in fp:
            elements = line.rstrip().split("\t")
            if len(elements) < 2:
                continue
            count = int(elements[1])
            cluster_to_mag_count[elements[0]] = count
            if count >= mag_count_threshold:
                valid_clusters.add(elements[0])

    return cluster_to_mag_count, valid_clusters


def get_cluster_to_magcount(cluster_to_magcount_filename: PathLike) -> dict[str, int]:
    """Return cluster -> MAG count mapping."""

    cluster_to_mag_count, _ = filter_cluster(cluster_to_magcount_filename, mag_count_threshold=0)
    return cluster_to_mag_count


def load_cluster_metadata(
    cluster_to_magcount_filename: PathLike,
    clusterpair_distance_filename: PathLike,
    mag_count_threshold: int,
):
    """Load MAG counts and cluster-pair metadata with validity flags."""

    cluster_to_mag_count, valid_clusters = filter_cluster(
        cluster_to_magcount_filename, mag_count_threshold
    )
    clusterpair_info_df = pd.read_csv(clusterpair_distance_filename, sep="\t")
    clusterpair_info_df["validity_cluster1"] = clusterpair_info_df["cluster1"].isin(valid_clusters)
    clusterpair_info_df["validity_cluster2"] = clusterpair_info_df["cluster2"].isin(valid_clusters)
    clusterpair_info_df["validity_pair"] = (
        clusterpair_info_df["validity_cluster1"] & clusterpair_info_df["validity_cluster2"]
    )
    clusterpair_info_df["self"] = clusterpair_info_df["cluster1"] == clusterpair_info_df["cluster2"]
    return cluster_to_mag_count, valid_clusters, clusterpair_info_df


def get_hgt_scores(
    target_ctidx: Iterable[int],
    hgt_result_filename: PathLike,
    cluster_to_mag_count: Mapping[str, int],
    scaling_constant: float = 1_000_000,
    exclude_inferred: bool = False,
    habitat_clusters: Sequence[str] | None = HABITAT_CLUSTERS,
    mag_count_threshold: int | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Aggregate HGT events between cluster pairs and normalize by MAG counts."""

    target_ctidx = set(int(idx) for idx in target_ctidx) if target_ctidx is not None else set()
    cluster_pair_to_ctidx = defaultdict(set)

    if mag_count_threshold is not None:
        valid_clusters = {
            cluster for cluster, count in cluster_to_mag_count.items() if count >= mag_count_threshold
        }
    elif habitat_clusters is not None:
        valid_clusters = set(habitat_clusters)
    else:
        valid_clusters = set(cluster_to_mag_count)

    for (
        ctidx,
        _ct_length,
        _sp1_short,
        _sp2_short,
        _gscore_1,
        _gscore_2,
        _genome1,
        _genome2,
        hc1,
        hc2,
        _distance,
        inferred,
    ) in _load_hgt_results(hgt_result_filename, use_cache=use_cache):
        if exclude_inferred and inferred == "True":
            continue

        if hc1 not in valid_clusters or hc2 not in valid_clusters:
            continue

        ctidx_int = int(ctidx)
        if target_ctidx and ctidx_int not in target_ctidx:
            continue

        cluster_pair_to_ctidx[_sorted_pair(hc1, hc2)].add(ctidx_int)

    result_as_list = []
    for hc1, hc2 in combinations_with_replacement(sorted(valid_clusters), 2):
        hc1, hc2 = _sorted_pair(hc1, hc2)
        hgt_event_count = len(cluster_pair_to_ctidx.get((hc1, hc2), set()))

        hc1_mag_count = cluster_to_mag_count.get(hc1)
        hc2_mag_count = cluster_to_mag_count.get(hc2)
        if hc1_mag_count is None or hc2_mag_count is None:
            raise ValueError(f"Missing MAG counts for cluster pair {(hc1, hc2)}")
        if hc1_mag_count <= 0 or hc2_mag_count <= 0:
            raise ValueError(
                f"MAG counts must be positive for clusters {(hc1, hc2)}; got {(hc1_mag_count, hc2_mag_count)}"
            )

        normalizing_factor = scaling_constant / (hc1_mag_count * hc2_mag_count)
        normalized_score = hgt_event_count * normalizing_factor
        result_as_list.append([hc1, hc2, hgt_event_count, normalized_score])

    return pd.DataFrame(
        result_as_list, columns=["cluster1", "cluster2", "hgt_count", "normalized_score"]
    )


def prepare_hgt_matrix(result_df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Standardize HGT score columns for downstream merges."""

    if "normalized_score" not in result_df.columns:
        raise ValueError("expected 'normalized_score' column in result_df")

    renamed = result_df.rename(columns={"normalized_score": f"{label}_HGT_score"})
    return renamed[["cluster1", "cluster2", f"{label}_HGT_score"]]


def split_hgt_events_by_generalism(
    cotransfer_annotation_df: pd.DataFrame,
    generalism_cutoff: float,
    hgt_result_filename: PathLike,
    cluster_to_mag_count: Mapping[str, int],
    exclude_inferred: bool = False,
) -> dict[str, pd.DataFrame]:
    """Compute total/specialist/mixed/generalist HGT matrices in one pass."""

    total_idx = set(cotransfer_annotation_df["co-transfer_index"])
    specialist_idx = set(
        cotransfer_annotation_df[
            cotransfer_annotation_df["max_generalism_score"] < generalism_cutoff
        ]["co-transfer_index"]
    )
    generalist_idx = set(
        cotransfer_annotation_df[
            cotransfer_annotation_df["min_generalism_score"] >= generalism_cutoff
        ]["co-transfer_index"]
    )
    mixed_idx = set(
        cotransfer_annotation_df[
            (cotransfer_annotation_df["max_generalism_score"] >= generalism_cutoff)
            & (cotransfer_annotation_df["min_generalism_score"] < generalism_cutoff)
        ]["co-transfer_index"]
    )

    def _matrix(indices: set[int], label: str) -> pd.DataFrame:
        result = get_hgt_scores(
            indices,
            hgt_result_filename,
            cluster_to_mag_count,
            exclude_inferred=exclude_inferred,
        )
        return prepare_hgt_matrix(result, label)

    return {
        "total": _matrix(total_idx, "total"),
        "specialist": _matrix(specialist_idx, "ss"),
        "mixed": _matrix(mixed_idx, "sg"),
        "generalist": _matrix(generalist_idx, "gg"),
    }


def build_hgt_distance_dataframe(
    hgt_matrices: Mapping[str, pd.DataFrame], clusterpair_info_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge cluster-pair metadata with each HGT matrix."""

    merged = clusterpair_info_df.copy()
    for df in hgt_matrices.values():
        merged = merged.merge(df, on=["cluster1", "cluster2"], how="left")

    score_cols = [c for c in merged.columns if c.endswith("_HGT_score")]
    merged[score_cols] = merged[score_cols].fillna(0)
    return merged


def plot_hgt_weight_vs_ED(hgt_distance_df: pd.DataFrame, save=None):
    """Scatterplot of normalized HGT counts vs. ecological distance."""

    required = {
        "cluster1",
        "cluster2",
        "ecological_distance",
        "validity_pair",
        "total_HGT_score",
        "ss_HGT_score",
    }
    missing = required - set(hgt_distance_df.columns)
    if missing:
        raise ValueError(f"Missing columns for plotting: {missing}")

    valid_nonzero = hgt_distance_df[
        (hgt_distance_df["validity_pair"]) & (hgt_distance_df["total_HGT_score"] > 0)
    ].copy()
    valid_nonzero["proportion_generalist_involved_hgt"] = (
        valid_nonzero["total_HGT_score"] - valid_nonzero["ss_HGT_score"]
    ) / valid_nonzero["total_HGT_score"]

    valid_zero = hgt_distance_df[
        (hgt_distance_df["validity_pair"]) & (hgt_distance_df["total_HGT_score"] == 0)
    ]

    eval_x = np.linspace(
        hgt_distance_df["ecological_distance"].min(),
        hgt_distance_df["ecological_distance"].max(),
        50,
    )
    smoothed = lowess(
        exog=hgt_distance_df["ecological_distance"],
        endog=hgt_distance_df["total_HGT_score"],
        xvals=eval_x,
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(
        data=valid_zero, x="ecological_distance", y="total_HGT_score", color="black", ax=ax, alpha=0.3
    )
    sns.scatterplot(
        data=valid_nonzero,
        x="ecological_distance",
        y="total_HGT_score",
        hue="proportion_generalist_involved_hgt",
        ax=ax,
        palette="coolwarm",
    )
    ax.plot(eval_x, smoothed, c="k", alpha=0.5)

    ax.set_ylabel("Normalized HGT count")
    ax.set_xlabel("Ecological distance")

    if save is not None:
        plt.savefig(save)

    plt.show()

    return hgt_distance_df


def plot_generalist_proportion_vs_ED(hgt_distance_df: pd.DataFrame, save=None):
    """Visualize the proportion of generalist-involved HGT vs. ecological distance."""

    required = {
        "ecological_distance",
        "validity_pair",
        "total_HGT_score",
        "ss_HGT_score",
    }
    missing = required - set(hgt_distance_df.columns)
    if missing:
        raise ValueError(f"Missing columns for plotting: {missing}")

    valid_nonzero = hgt_distance_df[
        (hgt_distance_df["validity_pair"]) & (hgt_distance_df["total_HGT_score"] > 0)
    ].copy()
    valid_nonzero["proportion_generalist_involved_hgt"] = (
        valid_nonzero["total_HGT_score"] - valid_nonzero["ss_HGT_score"]
    ) / valid_nonzero["total_HGT_score"]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.regplot(
        data=valid_nonzero,
        x="ecological_distance",
        y="proportion_generalist_involved_hgt",
        ax=ax,
    )

    ax.set_ylabel("Proportion of generalist-involved HGT")
    ax.set_xlabel("Ecological distance")

    if save is not None:
        plt.savefig(save)

    plt.show()

    return valid_nonzero


def plot_ED_vs_HGT(hgt_result_df: pd.DataFrame, save=None):
    """Scatter ecological distance vs. normalized HGT score (Figure 5C)."""

    no_hgt_part = hgt_result_df[hgt_result_df["hgt_count"] == 0]
    hgt_part = hgt_result_df[hgt_result_df["hgt_count"] != 0]

    eval_x = np.linspace(
        hgt_result_df["ecological_distance"].min(),
        hgt_result_df["ecological_distance"].max(),
        50,
    )
    smoothed = lowess(
        exog=hgt_result_df["ecological_distance"],
        endog=hgt_result_df["normalized_score"],
        xvals=eval_x,
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(
        data=no_hgt_part, x="ecological_distance", y="normalized_score", color="black", ax=ax, alpha=0.3
    )
    sns.scatterplot(
        data=hgt_part,
        x="ecological_distance",
        y="normalized_score",
        hue="proportion_of_generalist_involved_HGT",
        ax=ax,
        palette="coolwarm",
    )
    ax.plot(eval_x, smoothed, c="k", alpha=0.5)

    ax.set_ylabel("Normalized HGT count")
    ax.set_xlabel("Ecological distance")

    if save is not None:
        plt.savefig(save)

    plt.show()


def plot_ED_vs_GProportion(hgt_result_df: pd.DataFrame, save=None):
    """Scatter ecological distance vs. proportion of generalist-involved HGT (Figure 5D)."""

    hgt_part = hgt_result_df[hgt_result_df["hgt_count"] != 0]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.regplot(
        data=hgt_part,
        x="ecological_distance",
        y="proportion_of_generalist_involved_HGT",
        ax=ax,
    )
    ax.set_ylabel("Proportion of generalist-involved HGT")
    ax.set_xlabel("Ecological distance")

    r, p = pearsonr(
        hgt_part["ecological_distance"],
        hgt_part["proportion_of_generalist_involved_HGT"],
    )
    print("Pearson", r, p)

    if save is not None:
        plt.savefig(save)

    plt.show()


def split_hgt_by_ED_and_Gscore(
    cotransfer_info_df: pd.DataFrame,
    hgt_result_filename: PathLike,
    cluster_to_mag_count: Mapping[str, int],
    clusterpair_info_df: pd.DataFrame,
    generalism_cutoff: float = 0.1,
    ecological_distance_cutoff: float = 0.5,
    scaling_constant: float = 1_000_000,
    mag_count_threshold: int = 1000,
    exclude_inferred: bool = False,
):
    """Split HGT weights by ecological distance and generalism strata."""

    total_cotransfer_indexes = set(cotransfer_info_df["co-transfer_index"])

    among_S_cotransfer_indexes = set(
        cotransfer_info_df[cotransfer_info_df["max_generalism_score"] < generalism_cutoff][
            "co-transfer_index"
        ]
    )

    between_SG_cotransfer_indexes = set(
        cotransfer_info_df[
            (cotransfer_info_df["max_generalism_score"] >= generalism_cutoff)
            & (cotransfer_info_df["min_generalism_score"] < generalism_cutoff)
        ]["co-transfer_index"]
    )

    among_G_cotransfer_indexes = set(
        cotransfer_info_df[cotransfer_info_df["min_generalism_score"] >= generalism_cutoff][
            "co-transfer_index"
        ]
    )

    get_scores = partial(
        get_hgt_scores,
        hgt_result_filename=hgt_result_filename,
        cluster_to_mag_count=cluster_to_mag_count,
        mag_count_threshold=mag_count_threshold,
        scaling_constant=scaling_constant,
        exclude_inferred=exclude_inferred,
    )

    total_HGT_df = get_scores(target_ctidx=total_cotransfer_indexes)
    among_S_HGT_result = get_scores(target_ctidx=among_S_cotransfer_indexes)
    between_SG_HGT_result = get_scores(target_ctidx=between_SG_cotransfer_indexes)
    among_G_HGT_result = get_scores(target_ctidx=among_G_cotransfer_indexes)

    merged_df = pd.merge(total_HGT_df, among_S_HGT_result, on=["cluster1", "cluster2"], suffixes=("", "_among_S"))
    merged_df = pd.merge(merged_df, between_SG_HGT_result, on=["cluster1", "cluster2"], suffixes=("", "_between_SG"))
    merged_df = pd.merge(merged_df, among_G_HGT_result, on=["cluster1", "cluster2"], suffixes=("", "_among_G"))
    merged_df = pd.merge(merged_df, clusterpair_info_df, on=["cluster1", "cluster2"], how="left")

    merged_df["disparate_habitat_HGT"] = merged_df["ecological_distance"] > ecological_distance_cutoff

    result_df = merged_df.groupby("disparate_habitat_HGT")[
        ["normalized_score", "normalized_score_among_S", "normalized_score_between_SG", "normalized_score_among_G"]
    ].sum()
    proportion_df = result_df[["normalized_score_among_S", "normalized_score_between_SG", "normalized_score_among_G"]].div(
        result_df["normalized_score"], axis=0
    )
    return result_df, proportion_df


def get_unique_HGT(
    HGT_result_filename: PathLike,
    species_to_full_taxonomy: Union[Mapping[str, str], PathLike, None] = None,
    exclude_inferred: bool = True,
    valid_clusters: set[str] | None = None,
    cluster_to_mag_count: Mapping[str, int] | None = None,
    mag_count_threshold: int | None = None,
    use_cache: bool = True,
):
    """Collect unique HGT events and map them to taxa at each rank."""

    prefix_to_rank = {"d": "domain", "p": "phylum", "c": "class", "o": "order", "f": "family", "g": "genus", "s": "species"}

    unique_HGTs = set()
    rank_taxon_to_unique_HGTs = {
        "domain": defaultdict(set),
        "phylum": defaultdict(set),
        "class": defaultdict(set),
        "order": defaultdict(set),
        "family": defaultdict(set),
        "genus": defaultdict(set),
        "species": defaultdict(set),
    }

    if isinstance(species_to_full_taxonomy, (str, Path)) or species_to_full_taxonomy is None:
        taxmap_path = (
            Path(species_to_full_taxonomy)
            if species_to_full_taxonomy is not None
            else Path(HGT_result_filename).resolve().parent / "GTDBr207_defined_species.txt"
        )
        with taxmap_path.open() as fp:
            species_to_full_taxonomy = {
                line.strip().split(";")[-1]: line.strip() for line in fp if line.strip()
            }

    if valid_clusters is None and cluster_to_mag_count is not None and mag_count_threshold is not None:
        valid_clusters = {
            cluster for cluster, count in cluster_to_mag_count.items() if count >= mag_count_threshold
        }

    for (
        ctidx,
        _ct_length,
        sp1_short,
        sp2_short,
        _gscore_1,
        _gscore_2,
        _genome1,
        _genome2,
        hc1,
        hc2,
        _distance,
        inferred,
    ) in _load_hgt_results(HGT_result_filename, use_cache=use_cache):

        if exclude_inferred and inferred == "True":
            continue

        if valid_clusters is not None and (hc1 not in valid_clusters or hc2 not in valid_clusters):
            continue

        if sp1_short not in species_to_full_taxonomy or sp2_short not in species_to_full_taxonomy:
            raise KeyError("Taxonomy missing for one of the species in HGT file.")

        ctidx = int(ctidx)
        cluster_pair = "\t".join(_sorted_pair(hc1, hc2))
        sp1_full_taxonomy = species_to_full_taxonomy[sp1_short]
        sp2_full_taxonomy = species_to_full_taxonomy[sp2_short]

        unique_HGT = (sp1_short, sp2_short, ctidx, cluster_pair)
        unique_HGTs.add(unique_HGT)

        for taxon in sp1_full_taxonomy.split(";"):
            rank = prefix_to_rank[taxon[0]]
            rank_taxon_to_unique_HGTs[rank][taxon].add(unique_HGT)

        for taxon in sp2_full_taxonomy.split(";"):
            rank = prefix_to_rank[taxon[0]]
            rank_taxon_to_unique_HGTs[rank][taxon].add(unique_HGT)

    return unique_HGTs, rank_taxon_to_unique_HGTs


def get_unique_taxa_hgt(
    hgt_result_filename: PathLike,
    cluster_to_mag_count: Mapping[str, int],
    taxonomy_path: PathLike | None = None,
    mag_count_threshold: int = 1000,
    exclude_inferred: bool = False,
):
    """Notebook-friendly wrapper around get_unique_HGT with cluster filtering."""

    return get_unique_HGT(
        hgt_result_filename,
        species_to_full_taxonomy=taxonomy_path,
        exclude_inferred=exclude_inferred,
        cluster_to_mag_count=cluster_to_mag_count,
        mag_count_threshold=mag_count_threshold,
    )


def HGT_weight_taxa_removed(
    unique_HGTs,
    taxon_to_unique_HGTs,
    taxa_to_remove,
    cluster_to_mag_count,
    scaling_constant=1_000_000,
    habitat_clusters: Sequence[str] | None = HABITAT_CLUSTERS,
    mag_count_threshold: int | None = None,
):
    """Recompute HGT weights after removing selected taxa."""

    valid_unique_HGTs = unique_HGTs
    for taxon in taxa_to_remove:
        valid_unique_HGTs = valid_unique_HGTs - taxon_to_unique_HGTs[taxon]

    cluster_pair_to_ctidxs = defaultdict(set)
    for (_sp1, _sp2, ctidx, cluster_pair) in valid_unique_HGTs:
        cluster_pair_to_ctidxs[cluster_pair].add(ctidx)

    cluster_pair_hgt_score_aslist = []
    if mag_count_threshold is not None:
        valid_clusters = {
            cluster for cluster, count in cluster_to_mag_count.items() if count >= mag_count_threshold
        }
    elif habitat_clusters is not None:
        valid_clusters = set(habitat_clusters)
    else:
        valid_clusters = set(cluster_to_mag_count)

    for hc1, hc2 in combinations_with_replacement(sorted(valid_clusters), 2):
        hc1, hc2 = _sorted_pair(hc1, hc2)
        cluster_pair = "\t".join([hc1, hc2])
        HGT_count = len(cluster_pair_to_ctidxs.get(cluster_pair, set()))

        hc1_mag_count = cluster_to_mag_count[hc1]
        hc2_mag_count = cluster_to_mag_count[hc2]
        normalizing_factor = scaling_constant / (hc1_mag_count * hc2_mag_count)
        normalized_score = HGT_count * normalizing_factor
        cluster_pair_hgt_score_aslist.append([hc1, hc2, HGT_count, normalized_score])

    return pd.DataFrame(
        cluster_pair_hgt_score_aslist,
        columns=["cluster1", "cluster2", "HGT_count", "normalized_HGT_count"],
    )


def summarize_distance_contributions(ed_hgt_weight_df: pd.DataFrame, distance_cutoff: float):
    """Summarize specialist/generalist contributions for near vs. distant habitats."""

    results = []
    groups = {
        "close": ed_hgt_weight_df[ed_hgt_weight_df["ecological_distance"] < distance_cutoff],
        "disparate": ed_hgt_weight_df[ed_hgt_weight_df["ecological_distance"] >= distance_cutoff],
    }
    for label, df in groups.items():
        total = df["total_HGT_score"].sum()
        summary = {
            "distance_group": label,
            "total_HGT_score": total,
            "ss_fraction": (df["ss_HGT_score"].sum() / total) if total else np.nan,
            "sg_fraction": (df["sg_HGT_score"].sum() / total) if total else np.nan,
            "gg_fraction": (df["gg_HGT_score"].sum() / total) if total else np.nan,
        }
        results.append(summary)

    return pd.DataFrame(results)


def line_and_bar_dualaxes_plot(df, x_axis, y_axis1, y_axis2, save=None):
    """Dual-axis visualization used for Figure 5E"""

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(df[x_axis], df[y_axis1], color="lightblue", label=y_axis1, alpha=0.7)
    ax1.set_ylabel(y_axis1, color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.set_xticklabels(df[x_axis], rotation=45, ha="right")
    ax1.set_ylim([0, 0.5])

    ax2 = ax1.twinx()
    ax2.plot(df[x_axis], df[y_axis2], color="red", marker="o", label=y_axis2, zorder=10)
    ax2.set_ylabel(y_axis2, color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.set_ylim([0, 1])

    plt.tight_layout()

    if save is not None:
        plt.savefig(save)

    plt.show()
