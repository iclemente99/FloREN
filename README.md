# FloREN: An Interpretable Sample Representation Method to unveil immune networks through Graph Transformers

<p align="center">
  <img src="docs/floren.png" width="600">
</p>


## 📥 Setup & Installation

### 1. Clone FloREN Locally

```bash
# Download locally the repository
git clone https://github.com/iclemente99/FloREN

# Move inside the repository
cd FloREN

```

### 2. HGT Environment

OPTION 1 - Create conda HGT environment

```bash

conda env create -f env/hgt_env.yml
conda activate hgt_env

```

OPTION 2 - Create uv HGT environment

```bash
# Install UV just once
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install the python environment
uv venv hgt_env
source hgt_env/bin/activate # For Windows run: hgt_env/Scripts/activate
uv pip install -r env/hgt_env.txt

```

### 3. Unzip Prior Knowledge Reference for Usage

```bash
# Unzip the file
mkdir -p temp
unzip ./data/Prior_Knowledge_PRECISEADS_compI_compII.zip -d temp # For Windows run: Expand-Archive -Path ".\data\Prior_Knowledge_PRECISEADS_compI_compII.zip" -DestinationPath ".\temp"
unzip temp/*.zip -d temp/inner # For Windows run: Get-ChildItem "temp\*.zip" | ForEach-Object { Expand-Archive -Path $_.FullName -DestinationPath "temp\inner" -Force }
mv temp/inner/*.csv ./data/Prior_Knowledge_PRECISEADS.csv # For Windows run: Get-ChildItem "temp\inner\*.csv" | Select-Object -First 1 | Copy-Item -Destination ".\data\Prior_Knowledge_PRECISEADS.csv"

```

## Understanding the Input

To run the FloREN pipeline, the input data must be organized in the `./data/` directory with specific formats for gene expression and cell connection data. Below are the requirements for each sample:

### 1. Adata object (`./data/`)
- **VARS**: The adata object should only contain the genes that you want to work with (e.x: 2000 HVG). adata.var_names will be the gene names used.
- **MATRIX**: The matrix used for the model is going to be the adata.layers['logcounts'] by default - You can change the adata.layers to use.
- **OBS**: The adata.obs_names will be the cell names used. The obs used are going to be "patient_id" for sample aggregation and "group" for supervised classification task.

### 2. Cell-Cell Connection Matrices (`./data/cell_connections/`)
- **Location**: `./data/cell_connections/`
- **Format**: One matrix file per sample, stored as a csv file.
- **Dimensions**: Each matrix should have dimensions `[M, M]`, where `M` = number of cells (matching the number of cells in the corresponding gene expression matrix).
- **Content**: The matrix represents cell-cell connections (e.g., adjacency matrix).
- **Naming**: Files should match the sample identifiers used in `adata.obs.patient_id`

## 🚀 Usage

### Step 1: Build Heterogenous Graph

This step will make sure the input is in the correct format, will run the autoencoders compression and the gene-regulatory network inference.

```bash

# Make sure you're in the activated environment

# Run floren_input.py function
python src/floren_input.py \
  --adata_path './data/binvignat_example.h5ad' \
  --cell_comm_path './data/cell_connections/' \
  --output_path './floren_output/' \
  --epochs 150 \
  --grn_cutoff 0.9 \

```
**If cell_comm_path not given, the model will keep running without cell communication information.**

**Note**: All scripts must be run from the `src/` directory, or the working directory must be set to `src/` before running (e.g. `cd src && python floren_input.py ...`).

### Step 2: Train Heterogenous Graph Transformer Self-Supervised Learning (HGTSSL) with Supervised finetunning

This step will train the model and save all the results: patient embeddings, cell embeddings gene embeddings, cell attention, gene attention and attention network.

```bash

# Run floren_training.py
python src/floren_training.py \
  --adata_path './data/binvignat_example.h5ad' \
  --result_dir './floren_output/' \
  --epochs 100 \
  --patient_id "patient_id" \
  --metadata_group "disease" \
  --min_count 0  \

```

### Step 3: Visualize results

Once the model is trained, this step will help with the initial visualization of the results.

```bash

# Run floren_visualization.py
python src/floren_visualization.py \
  --adata_path './data/binvignat_example.h5ad' \
  --result_dir './floren_output/' \
  --patient_id "patient_id" \
  --metadata_group "disease" \

```

### Step 4: Downstream Analysis

After training, FloREN provides a set of ready-to-use analysis functions in `src/downstream.py`.
Import them directly in a Python script or Jupyter notebook — no extra training step required.

**Available functions:**

| Function | What it does |
|---|---|
| `plot_celltype_saliency_ranking` | Ranks cell types by mean attention score across all patients |
| `plot_celltype_gene_signatures` | Builds per-cell-type gene attention profiles and plots the top N genes |
| `plot_celltype_differential_abundance` | Compares cell-type saliency between two patient groups (Mann-Whitney + FDR) |
| `run_differential_gene_expression` | Compares gene embedding shift between groups; outputs a volcano plot |
| `plot_celltype_niche_heatmaps` | Computes within- and cross-group cell-type niche similarity heatmaps |
| `plot_grn_leiden_network` | Builds gene co-attention networks and detects Leiden modules per group |
| `plot_celltype_attention_heatmaps` | Plots cell-type × cell-type edge-attention matrices side by side for two groups |

**Example — rank cell types by saliency:**

```python
import sys
sys.path.insert(0, "src")
from downstream import plot_celltype_saliency_ranking

saliency = plot_celltype_saliency_ranking(
    floren_results_path = "./floren_output",   # folder passed as --result_dir during training
    h5ad_path           = "./data/binvignat_example.h5ad",
    celltype_col        = "Celltype.Lev1.manuscript",  # adata.obs column with cell-type labels
    sample_col          = "patient_id",                # adata.obs column with patient IDs
    agg                 = "mean",   # summarise saliency per cell type: "mean", "median", or a quantile float
    top_n               = 15,       # plot only the 15 highest-saliency cell types; None = show all
)
# saliency is a pd.Series sorted by mean score — inspect it directly:
print(saliency.head(10))
```

This saves a lollipop PDF to `./floren_output/plots/Celltype_Saliency_Ranking.pdf` and returns
a `pd.Series` you can filter or export for further analysis.

**Example — compare cell-type saliency between two patient groups:**

```python
from downstream import plot_celltype_differential_abundance

results_df, all_cells = plot_celltype_differential_abundance(
    floren_results_path = "./floren_output",
    h5ad_path           = "./data/binvignat_example.h5ad",
    group_assignment    = {"patient_A": "Disease", "patient_B": "Control", "patient_C": "Disease"},
    # alternatively, use a substring rule: ("MS", ["MS", "HC"])
    celltype_col        = "Celltype.Lev1.manuscript",
    sample_col          = "patient_id",
    agg                 = "mean",   # "mean" is safer than "sum" when patient cell counts differ
    scale               = "both",   # save linear and log-scale panels in one PDF
)
# results_df contains Mann-Whitney U statistics and BH-corrected p-values per cell type
significant = results_df[results_df["p_adj"] < 0.05].sort_values("p_adj")
print(significant[["celltype", "p_adj", "Disease_median", "Control_median"]])
```

**Example — cell-type × cell-type attention heatmaps:**

```python
import scanpy as sc
from downstream import plot_celltype_attention_heatmaps

adata = sc.read_h5ad("./data/binvignat_example.h5ad")
meta = adata.obs[["Celltype.Lev1.manuscript", "patient_id"]].copy()
meta.columns = ["celltype", "patient"]
meta["barcode"] = meta["patient"] + "__" + adata.obs_names

matrices = plot_celltype_attention_heatmaps(
    att_dir          = "./floren_output/floren_attention_embeddings",
    cell_names_path  = "./floren_output/All_AUC_Cell_names.csv",
    meta             = meta.reset_index(drop=True),
    gene_names       = list(adata.var_names),
    group_assignment = {"patient_A": "Disease", "patient_B": "Control"},
    # alternatively: ("MS", ["MS", "HC"])
    output_pdf       = "./floren_output/plots/celltype_attention_heatmaps.pdf",
)
# matrices is a dict {group_label: pd.DataFrame (celltype x celltype)}
```

The function saves a two-panel PDF with a shared color scale — one heatmap per group — and
returns the underlying DataFrames for further analysis.

## ✍️ Citation & Acknowledgements

This work was developed at LBAI-UBO. Please cite accordingly if used in academic research.

## 🖥️ Maintainers

Iñigo Clemente Larramendi — inigo.clementelarramendi@univ-brest.fr
