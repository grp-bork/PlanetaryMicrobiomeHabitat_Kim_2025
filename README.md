# PlanetaryMicrobiomeHabitat_Kim_2025

## Description
This repository contains the python scripts used in *Kim, Podlesny, Schiller et al. 2025 (Under review)*.

**1. data.tar.bz2**: Please unzip this data file first. `tar -xjvf data.tar.bz2`.
The archive expands into `data/` and provides the precomputed habitat cluster embeddings, microntology colormaps, and other tabular inputs consumed by the notebooks and HTML explorer.

**2. evaluate_generalism.ipynb**: Calculates a generalism score for each prokaryotic species using the shared utilities in `scripts/generalism.py`, and contains the code needed to reproduce Figure 4A from the habitat cluster ecological distance matrix.

**3. horizontal_gene_transfer_analysis.ipynb**: Uses `scripts/hgt.py` to load HGT event tables, normalize by MAG counts, and relate exchange rates to ecological distance, reproducing Figures 5C, 5D, 5E, and 6D.

**4. explore_habitat_cluster.html**: Interactively explore the habitat clustering 3D UMAP embedding (generated via `scripts/generate_umap_explorer.py`), with toggles to color by cluster palette or simplified microntology annotations.

## Environment
Create the conda environment defined in `environment.yml` (Python 3.11 with pinned scientific stack):

```bash
conda env create -f environment.yml
conda activate planetary-microbiome
```

If the environment already exists, update it to match the file and prune stale packages:

```bash
conda env update -f environment.yml --prune
```

Citation: https://doi.org/10.1101/2025.07.18.664989
