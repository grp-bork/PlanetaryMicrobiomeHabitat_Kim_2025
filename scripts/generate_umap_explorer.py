"""Generate an interactive 3D UMAP explorer HTML for metagenomic samples.

Controls:
- Color by cluster (distinct palette)
- Color by cluster (habitat-context palette)
- Color by simplified microntology

Notes:
- Cluster '#N/A' and microntology '0' (missing/ambiguous) are rendered as
  semi-transparent gray so they do not dominate the view.
- Samples with multiple microntology labels (pipe-delimited) are treated as
  microntology == '0' per requirements.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

SAMPLE_PATH = DATA / "sample_coordinate_and_annotation.tsv"
CLUSTER_INFO_PATH = DATA / "cluster_information.tsv"
MICRO_COLORMAP_PATH = DATA / "microntology.simplified.colormap.tsv"
OUTPUT_PATH = ROOT / "explore_habitat_cluster.html"

GREY = "rgba(128,128,128,0.3)"


def load_data():
    samples = pd.read_csv(SAMPLE_PATH, sep="\t")
    cluster_info = pd.read_csv(CLUSTER_INFO_PATH, sep="\t")
    micro_cmap = pd.read_csv(MICRO_COLORMAP_PATH, sep="\t", header=0, names=["microntology", "color"])
    return samples, cluster_info, micro_cmap


def prepare_microntology(samples: pd.DataFrame) -> pd.DataFrame:
    simplicol = "Simplified microntology"
    cleaned = samples.copy()
    cleaned[simplicol] = cleaned[simplicol].fillna("0")
    multi_mask = cleaned[simplicol].str.contains("|", regex=False)
    cleaned.loc[multi_mask, simplicol] = "NA or multiple annotation"
    return cleaned


def cluster_traces(samples: pd.DataFrame, cluster_colors: dict[str, str]) -> list[go.Scatter3d]:
    traces = []
    for cluster_code, group in samples.groupby("Habitat cluster", sort=False):
        color = cluster_colors.get(cluster_code, GREY)
        opacity = 0.9 if color != GREY else 0.3
        traces.append(
            go.Scatter3d(
                x=group["UMAP_0"],
                y=group["UMAP_1"],
                z=group["UMAP_2"],
                mode="markers",
                name=str(cluster_code),
                legendgroup=f"cluster-{cluster_code}",
                marker=dict(size=3, color=color, opacity=opacity),
                hovertemplate=(
                    "Sample: %{customdata[0]}<br>"
                    "UMAP_0: %{x:.3f}<br>UMAP_1: %{y:.3f}<br>UMAP_2: %{z:.3f}<br>"
                    "Cluster: %{customdata[1]}<br>Simplified microntology: %{customdata[2]}<extra></extra>"
                ),
                customdata=group[["Sample", "Habitat cluster", "Simplified microntology"]],
                showlegend=True,
                visible=False,
            )
        )
    return traces


def microntology_traces(samples: pd.DataFrame, micro_colors: dict[str, str]) -> list[go.Scatter3d]:
    traces = []
    simplicol = "Simplified microntology"
    for mic, group in samples.groupby(simplicol, sort=False):
        color = micro_colors.get(mic, GREY)
        opacity = 0.9 if color != GREY else 0.3
        traces.append(
            go.Scatter3d(
                x=group["UMAP_0"],
                y=group["UMAP_1"],
                z=group["UMAP_2"],
                mode="markers",
                name=str(mic),
                legendgroup=f"microntology-{mic}",
                marker=dict(size=3, color=color, opacity=opacity),
                hovertemplate=(
                    "Sample: %{customdata[0]}<br>"
                    "UMAP_0: %{x:.3f}<br>UMAP_1: %{y:.3f}<br>UMAP_2: %{z:.3f}<br>"
                    "Cluster: %{customdata[1]}<br>Simplified microntology: %{customdata[2]}<extra></extra>"
                ),
                customdata=group[["Sample", "Habitat cluster", simplicol]],
                showlegend=True,
                visible=False,
            )
        )
    return traces


def build_figure():
    samples, cluster_info, micro_cmap = load_data()
    samples = prepare_microntology(samples)

    cluster_distinct = dict(
        zip(cluster_info["Cluster code"], cluster_info["Cluster color code - distinct color (in Figure 1B)"])
    )
    cluster_habitat = dict(
        zip(cluster_info["Cluster code"], cluster_info["Cluster color code - representative habitat considered (in Figure 2A)"])
    )
    micro_colors = dict(zip(micro_cmap["microntology"], micro_cmap["color"]))
    micro_colors.setdefault("0", GREY)

    traces_cluster_distinct = cluster_traces(samples, cluster_distinct)
    traces_cluster_habitat = cluster_traces(samples, cluster_habitat)
    traces_microntology = microntology_traces(samples, micro_colors)

    all_traces = traces_cluster_distinct + traces_cluster_habitat + traces_microntology

    n_cd = len(traces_cluster_distinct)
    n_ch = len(traces_cluster_habitat)
    n_mi = len(traces_microntology)

    def visibility(distinct=False, habitat=False, microntology=False):
        return [distinct] * n_cd + [habitat] * n_ch + [microntology] * n_mi

    fig = go.Figure(data=all_traces)
    fig.update_traces(visible=False)
    fig.update_layout(
        scene=dict(
            xaxis_title="UMAP 0",
            yaxis_title="UMAP 1",
            zaxis_title="UMAP 2",
        ),
        height=900,
        width=1300,
        legend_title_text="Cluster (distinct color)",
        title="Metagenomic Sample UMAP Explorer",
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=True,
        updatemenus=[
            dict(
                buttons=[
                    dict(
                        label="Color by cluster – Distinct color (Fig S2H right)",
                        method="update",
                        args=[{"visible": visibility(distinct=True)}, {"legend_title_text": "Cluster (distinct color)"}],
                    ),
                    dict(
                        label="Color by cluster – Habitat context (Fig 2A)",
                        method="update",
                        args=[{"visible": visibility(habitat=True)}, {"legend_title_text": "Cluster (habitat context)"}],
                    ),
                    dict(
                        label="Color by microntology (Figure S2H left)",
                        method="update",
                        args=[{"visible": visibility(microntology=True)}, {"legend_title_text": "Simplified microntology"}],
                    ),
                ],
                direction="down",
                showactive=True,
                x=0.02,
                y=1.12,
            )
        ],
    )

    # Start with cluster distinct view
    initial_visibility = visibility(distinct=True)
    for trace, vis in zip(fig.data, initial_visibility):
        trace.visible = vis

    fig.write_html(OUTPUT_PATH, include_plotlyjs=True, full_html=True)

    return OUTPUT_PATH

if __name__ == "__main__":
    path = build_figure()
    print(f"Wrote {path}")
