"""
FloREN Pipeline — Python Example
=================================
Shows how to run the full pipeline programmatically from a Python script
instead of calling the scripts directly from the terminal.

Usage:
    python examples/run_floren.py

Run from the repository root so that relative paths resolve correctly.
Adjust DATA, OUTPUT, and the parameters below to match your dataset.
"""

import subprocess
import sys
import os

# ── Configuration ─────────────────────────────────────────────────────────────

PYTHON = sys.executable                               # same interpreter that runs this script
DATA   = "./data/binvignat_example.h5ad"              # path to your AnnData h5ad file
OUTPUT = "./floren_output"                            # all outputs land here

PATIENT_ID     = "patient_id"                        # adata.obs column with sample IDs
METADATA_GROUP = "disease"                           # adata.obs column with group labels (e.g. disease vs control)
MIN_COUNT      = 0                                   # min expression threshold for gene-cell edges

AE_EPOCHS  = 150      # autoencoder epochs for graph construction (Step 1)
GNN_EPOCHS = 100      # HGT training epochs (Step 2)
GRN_CUTOFF = 0.9      # gene-regulatory network edge cutoff (0 = all edges, 1 = none)


# ── Step 1 — Build heterogeneous graphs ───────────────────────────────────────
# Runs gene/cell autoencoders and infers the gene-regulatory network.
# Output: embeddings, gene-gene connections, cell names — all in OUTPUT/.

print("\n" + "="*60)
print("  STEP 1 — Graph construction (floren_input.py)")
print("="*60)

subprocess.run(
    [PYTHON, "src/floren_input.py",
     "--adata_path",    DATA,
     "--output_path",   OUTPUT,
     "--epochs",        str(AE_EPOCHS),
     "--grn_cutoff",    str(GRN_CUTOFF),
     # optional: add --cell_comm_path if you have cell-cell communication CSVs
     # "--cell_comm_path", "./data/cell_connections/",
     ],
    check=True,
)


# ── Step 2 — Train HGT ────────────────────────────────────────────────────────
# Trains the Heterogeneous Graph Transformer with a self-supervised contrastive
# objective and a supervised classification head.
# Output: patient/cell/gene embeddings and attention scores in OUTPUT/.

print("\n" + "="*60)
print("  STEP 2 — HGT training (floren_training.py)")
print("="*60)

subprocess.run(
    [PYTHON, "src/floren_training.py",
     "--adata_path",     DATA,
     "--output_path",    OUTPUT,
     "--result_dir",     OUTPUT,
     "--epochs",         str(GNN_EPOCHS),
     "--patient_id",     PATIENT_ID,
     "--metadata_group", METADATA_GROUP,
     "--min_count",      str(MIN_COUNT),
     ],
    check=True,
)

# ── Step 2 (resume) ───────────────────────────────────────────────────────────
# If the training above was interrupted (e.g. SLURM OOM or time limit),
# uncomment the block below and re-run this script.  FloREN will pick up
# exactly where it stopped — no changes to the other steps are needed.
#
# subprocess.run(
#     [PYTHON, "src/floren_training.py",
#      "--adata_path",     DATA,
#      "--output_path",    OUTPUT,
#      "--result_dir",     OUTPUT,
#      "--epochs",         str(GNN_EPOCHS),
#      "--patient_id",     PATIENT_ID,
#      "--metadata_group", METADATA_GROUP,
#      "--min_count",      str(MIN_COUNT),
#      "--resume",                             # <-- resume from last checkpoint
#      ],
#     check=True,
# )


# ── Step 3 — Visualize ────────────────────────────────────────────────────────
# Generates UMAP projections of patient/cell/gene embeddings and attention plots.
# Output: PDF figures saved to OUTPUT/plots/.

print("\n" + "="*60)
print("  STEP 3 — Visualization (floren_visualization.py)")
print("="*60)

subprocess.run(
    [PYTHON, "src/floren_visualization.py",
     "--adata_path",     DATA,
     "--output_path",    OUTPUT,
     "--result_dir",     OUTPUT,
     "--patient_id",     PATIENT_ID,
     "--metadata_group", METADATA_GROUP,
     "--epochs",         str(GNN_EPOCHS),
     ],
    check=True,
)


# ── Step 4 — Downstream analysis ──────────────────────────────────────────────
# All downstream functions can be imported directly — no retraining required.
# They read the embeddings and attention scores produced by Steps 1-3.

print("\n" + "="*60)
print("  STEP 4 — Downstream analysis (downstream.py)")
print("="*60)

sys.path.insert(0, "src")
import scanpy as sc
import pandas as pd
import numpy as np
from downstream import (
    cells_attention_ranking,
    gene_signatures,
    differential_abundance_analysis,
    run_differential_gene_expression,
    cell_niches_analysis,
    plot_grn_leiden_network,
    cell_communication_profiling,
    immune_network_plot,
)

adata = sc.read_h5ad(DATA)
gene_names = list(adata.var_names)
patients   = list(adata.obs[PATIENT_ID].unique())

# Build group assignment from adata metadata
_tmp = (adata.obs[[PATIENT_ID, METADATA_GROUP]]
        .drop_duplicates(PATIENT_ID)
        .set_index(PATIENT_ID)[METADATA_GROUP])
group_assignment = {p: str(_tmp[p]) for p in patients}
groups = sorted(set(group_assignment.values()))

os.makedirs(os.path.join(OUTPUT, "plots"), exist_ok=True)

# 4-A: rank cell types by mean attention saliency
saliency = cells_attention_ranking(
    floren_results_path = OUTPUT,
    h5ad_path           = DATA,
    celltype_col        = "cell_type",
    sample_col          = PATIENT_ID,
    agg                 = "mean",
    top_n               = 15,
)
print("\nTop cell types by attention:\n", saliency.head(5))

# 4-B: per-cell-type gene attention profiles
ct_gene_matrix = gene_signatures(
    floren_results_path = OUTPUT,
    h5ad_path           = DATA,
    gene_names          = gene_names,
    celltype_col        = "cell_type",
    sample_col          = PATIENT_ID,
    top_n_genes         = 20,
)

# 4-C: differential cell-type abundance between groups
differential_abundance_analysis(
    floren_results_path = OUTPUT,
    h5ad_path           = DATA,
    group_assignment    = group_assignment,
    celltype_col        = "cell_type",
    sample_col          = PATIENT_ID,
    agg                 = "mean",
    scale               = "both",
)

# 4-D: differential gene expression in embedding space
dge_result = run_differential_gene_expression(
    gene_embeddings_dir = os.path.join(OUTPUT, "floren_gene_embeddings"),
    gene_names          = gene_names,
    group_assignment    = group_assignment,
    make_plots          = True,
    output_dir          = os.path.join(OUTPUT, "plots"),
)

# 4-E: cell-type niche similarity (cosine + Euclidean + Pearson)
cell_niches_analysis(
    cell_embeddings_dir = os.path.join(OUTPUT, "floren_cell_embeddings"),
    h5ad_path           = DATA,
    group_assignment    = group_assignment,
    patient_col         = PATIENT_ID,
    celltype_col        = "cell_type",
    output_pdf          = os.path.join(OUTPUT, "plots", "cell_niches_analysis.pdf"),
)

# 4-F: GRN Leiden community network
gene_results_df = dge_result[0] if isinstance(dge_result, tuple) else dge_result
target_genes = (
    list(gene_results_df.sort_values("distance", ascending=False).head(40).index)
    if gene_results_df is not None and "distance" in gene_results_df.columns
    else gene_names[:40]
)
gene_scores = {g: pd.Series(np.ones(len(target_genes)), index=target_genes) for g in groups}

plot_grn_leiden_network(
    att_dir          = os.path.join(OUTPUT, "floren_attention_embeddings"),
    gene_names       = gene_names,
    target_genes     = target_genes,
    gene_scores      = gene_scores,
    group_assignment = group_assignment,
    leiden_resolution = 0.5,
    min_module_size  = 2,
    output_pdf       = os.path.join(OUTPUT, "plots", "GRN_leiden.pdf"),
)

# 4-G: cell-cell communication profiling
meta = pd.DataFrame({
    "celltype": adata.obs["cell_type"].astype(str).values,
    "patient":  adata.obs[PATIENT_ID].astype(str).values,
}, index=adata.obs.index)
meta["barcode"] = meta["patient"] + "__" + meta.index.astype(str)

cell_communication_profiling(
    att_dir          = os.path.join(OUTPUT, "floren_attention_embeddings"),
    cell_names_path  = os.path.join(OUTPUT, "All_AUC_Cell_names.csv"),
    meta             = meta,
    gene_names       = gene_names,
    group_assignment = group_assignment,
    output_pdf       = os.path.join(OUTPUT, "plots", "cell_communication_profiling.pdf"),
)

# 4-H: immune network plot (cell types + top genes as bipartite graph)
import networkx as nx

top_n      = 15
gene_means = ct_gene_matrix.mean(axis=0)
top_genes  = list(gene_means.sort_values(ascending=False).head(top_n).index)
sal_max    = saliency.max() or 1.0
gene_max   = float(gene_means[top_genes].max()) or 1.0

G = nx.Graph()
for ct in ct_gene_matrix.index:
    G.add_node(ct, size=float(saliency.get(ct, 0.0)) / sal_max)
for g in top_genes:
    G.add_node(g, size=float(gene_means[g]) / gene_max)

flat   = ct_gene_matrix[top_genes].values.flatten()
pos    = flat[flat > 0]
thresh = float(np.percentile(pos, 75)) if len(pos) > 0 else 0.0
for ct in ct_gene_matrix.index:
    for g in top_genes:
        w = float(ct_gene_matrix.at[ct, g])
        if w >= thresh:
            G.add_edge(ct, g, weight=w)

immune_network_plot(
    G          = G,
    cell_nodes = list(ct_gene_matrix.index),
    gene_nodes = top_genes,
    layout     = "kamada",
    save_path  = os.path.join(OUTPUT, "plots", "immune_network_plot.pdf"),
)

print("\n" + "="*60)
print("  FloREN pipeline complete.")
print(f"  All outputs saved to: {os.path.abspath(OUTPUT)}")
print("="*60)
