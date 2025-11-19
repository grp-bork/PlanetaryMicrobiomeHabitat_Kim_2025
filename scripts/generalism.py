"""Generalism scoring utilities shared across notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, spearmanr
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances

# Default axes used throughout the manuscript figures.
DEFAULT_AXES = [17, 28, 38]

# Color palette shared between notebooks.
CLUSTER_COLORMAP = {
    "HTS": "#733C93",
    "OLL": "#004C87",
    "OIL": "#004677",
    "OHL": "#004067",
    "POL": "#50B4C4",
    "EST": "#0073BB",
    "FRW": "#A3B8E0",
    "TEH": "#604242",
    "TER": "#AE9482",
    "TEG": "#9C7862",
    "ADC": "#901D22",
    "HOR": "#BF1A20",
    "HUV": "#ff60d4",
    "HSS": "#c9c28b",
    "BLD": "#5D5E5E",
    "HET": "#A0A0A0",
    "WT1": "#322D31",
    "ANI": "#C35E22",
    "MOG": "#D58A81",
    "RMG": "#D58A89",
    "CKG": "#D86F7C",
    "APG": "#EFA080",
    "HG1": "#EF5905",
    "MKG": "#EC9706",
    "PG1": "#E78D69",
    "PG2": "#EA8F6B",
    "WT2": "#302028",
    "HG2": "#E84C05",
    "HG3": "#F64C05",
    "HG4": "#FF4C05",
    "HG5": "#FA4C05",
    "HG6": "#EF5305",
    "HG7": "#E14C05",
    "NBG": "#F87030",
    "IFG": "#F07030",
    "CDG": "#F5CEBA",
    "BG1": "#DAAA9A",
    "BG2": "#DCA79A",
    "BG3": "#D8A598",
    "HRG": "#DCAA9D",
}


@dataclass(frozen=True)
class GeneralismResult:
    score: float
    weighted_centroid: pd.Series
    cluster_weights: pd.Series


def load_taxon_profile(
    taxon_table: Union[str, Path, pd.DataFrame],
    target_taxon: str,
    prevalence_cutoff: float = 0.01,
) -> pd.DataFrame:
    """Return rows for ``target_taxon`` that pass the prevalence cutoff."""

    if isinstance(taxon_table, (str, Path)):
        taxon_prevalence_abundance_df = pd.read_csv(taxon_table, sep="\t")
    else:
        taxon_prevalence_abundance_df = taxon_table

    filtered = taxon_prevalence_abundance_df[
        taxon_prevalence_abundance_df["prevalence"] > prevalence_cutoff
    ]
    target_df = filtered[filtered["taxon"] == target_taxon].copy()
    if target_df.empty:
        raise ValueError(
            f"No rows found for taxon '{target_taxon}'. "
            "Check the GTDB identifier or adjust the prevalence cutoff."
        )

    return target_df


def do_MDS(
    distance_matrix: pd.DataFrame,
    n_components: int = 40,
    random_state: int = 1,
    **mds_kwargs,
) -> pd.DataFrame:
    """Run MDS on the ecological distance matrix and keep the original index."""

    mds = MDS(
        n_components=n_components,
        dissimilarity="precomputed",
        max_iter=500_000,
        n_init=500,
        eps=1e-08,
        random_state=random_state,
        normalized_stress="auto",
        **mds_kwargs,
    )
    coordinates = mds.fit_transform(distance_matrix.values)
    df = pd.DataFrame(coordinates, index=distance_matrix.index)
    return df


def compare_distances(
    d_arr_1: Sequence[float],
    d_arr_2: Sequence[float],
    visualize: bool = True,
):
    """Compare two flattened distance arrays using Pearson and Spearman."""

    pearson_stat, pearson_p = pearsonr(d_arr_1, d_arr_2)
    spearman_stat, spearman_p = spearmanr(d_arr_1, d_arr_2)

    if visualize:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(d_arr_1, d_arr_2)
        plt.show()
        plt.close(fig)

    return {"pearson": (pearson_stat, pearson_p), "spearman":(spearman_stat, spearman_p)}


def find_best_axes(
    habitat_coordinates: pd.DataFrame,
    original_distances: Sequence[float],
    dim: int = 3,
) -> Tuple[List[int], float, np.ndarray]:
    """Return the axes that best recapitulate the original distance matrix."""

    max_r = 0.0
    max_axes: List[int] = []
    max_distance_arr: np.ndarray | None = None

    for axes in combinations(habitat_coordinates.columns, dim):
        distances = pdist(habitat_coordinates[list(axes)])
        r, _ = pearsonr(original_distances, distances)
        if r > max_r:
            max_r = r
            max_axes = list(axes)
            max_distance_arr = distances

    if max_distance_arr is None:
        raise ValueError("No axes evaluated; check inputs.")

    return max_axes, max_r, max_distance_arr


def compute_generalism(
    target_taxon_df: pd.DataFrame,
    habitat_cluster_coordinates: pd.DataFrame,
) -> GeneralismResult:
    """Calculate the weighted generalism score, centroid, and weights."""

    if len(target_taxon_df) == 0:
        raise ValueError("target_taxon_df is empty; ensure the taxon exists in the table.")

    missing = set(target_taxon_df["cluster"]) - set(habitat_cluster_coordinates.index)
    if missing:
        raise ValueError(
            "The following habitat clusters are missing from the coordinate table: "
            + ", ".join(sorted(missing))
        )

    target_taxon_df = target_taxon_df.copy()
    target_taxon_df["weight"] = (
        target_taxon_df["mean_abundance"] / target_taxon_df["mean_abundance"].max()
    )

    defined_coordinates = habitat_cluster_coordinates.loc[target_taxon_df["cluster"]]
    w_centroid = defined_coordinates.T * np.array(target_taxon_df["weight"])
    w_centroid = w_centroid.sum(axis=1) / target_taxon_df["weight"].sum()

    distances = pairwise_distances(
        defined_coordinates, np.array([w_centroid]), metric="euclidean"
    ).flatten()
    generalism_score = float((distances**2 * target_taxon_df["weight"]).sum())

    return GeneralismResult(
        score=generalism_score,
        weighted_centroid=pd.Series(w_centroid, index=defined_coordinates.columns),
        cluster_weights=target_taxon_df.set_index("cluster")["weight"],
    )


def visualize_generalism(
    result: GeneralismResult,
    target_taxon_df: pd.DataFrame,
    habitat_cluster_coordinates: pd.DataFrame,
    axes: Sequence[int] = DEFAULT_AXES,
):
    """Draw the weighted centroid and contributing clusters in 2D or 3D."""

    if len(axes) not in {2, 3}:
        raise ValueError("Unsupported dimension: only 2D and 3D axes are supported.")

    coordinates_target_axes = habitat_cluster_coordinates[list(axes)].copy()
    coordinates_target_axes["cluster_color"] = coordinates_target_axes.index.map(
        CLUSTER_COLORMAP
    )
    coordinates_target_axes["edge_color"] = coordinates_target_axes.index.map(
        lambda x: "red" if x in target_taxon_df["cluster"].values else "none"
    )
    coordinates_target_axes["alpha"] = coordinates_target_axes.index.map(
        lambda x: 1 if x in target_taxon_df["cluster"].values else 0.7
    )
    coordinates_target_axes["size"] = coordinates_target_axes.index.map(
        lambda x: 100 if x in target_taxon_df["cluster"].values else 70
    )
    coordinates_target_axes = coordinates_target_axes.merge(
        result.cluster_weights.rename("weight"),
        left_index=True,
        right_index=True,
        how="outer",
    ).fillna(0)

    rgba_colors = [
        mcolors.to_rgba(color, alpha=a)
        for color, a in zip(
            coordinates_target_axes["cluster_color"],
            coordinates_target_axes["alpha"],
        )
    ]

    coordinates_target_axes["weight"] = (
        (coordinates_target_axes["weight"] / (coordinates_target_axes["weight"].max() or 1))
        * 7
        + 0.1
    )

    fig = plt.figure(figsize=(6, 6))

    if len(axes) == 2:
        ax = fig.add_subplot(111)
        for _, row in coordinates_target_axes[
            coordinates_target_axes["edge_color"] == "red"
        ].iterrows():
            ax.plot(
                [result.weighted_centroid[axes[0]], row[axes[0]]],
                [result.weighted_centroid[axes[1]], row[axes[1]]],
                color="black",
                linewidth=row["weight"],
                zorder=1,
            )

        ax.scatter(
            coordinates_target_axes[axes[0]],
            coordinates_target_axes[axes[1]],
            c=rgba_colors,
            marker="o",
            edgecolors=coordinates_target_axes["edge_color"],
            s=coordinates_target_axes["size"],
            zorder=2,
        )
        ax.scatter(
            result.weighted_centroid[axes[0]],
            result.weighted_centroid[axes[1]],
            c="red",
            marker="*",
            s=200,
            zorder=2,
            edgecolors="white",
        )
        ax.set_xlabel(f"MDS {axes[0]}")
        ax.set_ylabel(f"MDS {axes[1]}")

    else:
        ax = fig.add_subplot(111, projection="3d")
        for _, row in coordinates_target_axes[
            coordinates_target_axes["edge_color"] == "red"
        ].iterrows():
            ax.plot(
                [result.weighted_centroid[axes[0]], row[axes[0]]],
                [result.weighted_centroid[axes[1]], row[axes[1]]],
                [result.weighted_centroid[axes[2]], row[axes[2]]],
                color="black",
                linewidth=row["weight"],
            )

        ax.scatter(
            coordinates_target_axes[axes[0]],
            coordinates_target_axes[axes[1]],
            coordinates_target_axes[axes[2]],
            c=rgba_colors,
            marker="o",
            edgecolors=coordinates_target_axes["edge_color"],
            s=coordinates_target_axes["size"],
            depthshade=False,
            zorder=1,
        )
        ax.scatter(
            result.weighted_centroid[axes[0]],
            result.weighted_centroid[axes[1]],
            result.weighted_centroid[axes[2]],
            c="red",
            marker="*",
            s=200,
            edgecolors="white",
        )
        ax.set_xlabel(f"MDS {axes[0]}")
        ax.set_ylabel(f"MDS {axes[1]}")
        ax.set_zlabel(f"MDS {axes[2]}")

    plt.show()
    plt.close(fig)

    print(
        f"Generalism score: {result.score:.4f}\n"
        f"Clusters count: {len(target_taxon_df)}"
    )
