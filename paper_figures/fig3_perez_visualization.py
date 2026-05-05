################################################################
#
#                             FLOREN
#
################################################################

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

import scanpy as sc
import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import phate

l_128 = pd.read_csv("/Users/Inigo/Desktop/FloREN3.0/perez_output/l_128.csv")
X = l_128.iloc[:,1:].values

embeddings = {}

# UMAP
embeddings["UMAP"] = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    metric="euclidean",
    random_state=42
).fit_transform(X)

# t-SNE
embeddings["tSNE"] = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    random_state=42
).fit_transform(X)

# PHATE
embeddings["PHATE"] = phate.PHATE(
    n_components=2,
    knn=15,
    random_state=42
).fit_transform(X)

# PCA
embeddings["PCA"] = PCA(n_components=2).fit_transform(X)

adata_path = "C:/Users/Inigo/Desktop/FloREN/Perez/GSE174188_CLUES1_adjusted.h5ad"
adata = sc.read(adata_path)

all_ids = list(adata.obs["ind_cov"].unique())
meta = (
    adata.obs.groupby("ind_cov")[["Age", "Sex", "batch_cov", "Status", "SLE_status"]]
    .first()
    .loc[all_ids]
)

metadata = pd.read_csv("C:/Users/Inigo/Desktop/FloREN/Perez/samples_metadata_unique.csv")
metadata.columns = ["Unnamed: 0","0","patient_id","group","Age","Sex","Ethnicity","Pool","Batch","Age_Group"]
metadata["group"] = metadata["group"].astype("category").cat.codes

meta = meta.reindex(metadata["patient_id"])
meta.index = metadata["patient_id"]

#embeddings_patient_id = (
#    l_128["Unnamed: 0"]
#    #.str.split("__")
#    #.str[0]
#)

#meta = (meta
#    .loc[embeddings_patient_id]
#    .reset_index()
#)

#meta.index = metadata["patient_id"]
#groups = []
#for pid in meta.index:
#    if str(pid).startswith(("HC-", "IGTB")):
#        groups.append("HC")
#    elif str(pid).startswith("FLARE"):
#        groups.append("FLARE")
#    else:
#        groups.append("SLE")

groups = meta["Status"].astype(str) + "_" + meta["SLE_status"].astype(str)
meta["Group"] = pd.Categorical(groups)
meta["Age"] = pd.to_numeric(meta["Age"], errors="coerce")
meta["Sex"] = meta["Sex"].astype(str)
meta["batch_cov"] = meta["batch_cov"].astype(str)

#group_colors = {"HC": "#1f77b4", "SLE": "#ff7f0e", "FLARE": "#2ca02c"}
group_colors = {
    "Healthy_Healthy": "#4C78A8",   # calm blue
    "Managed_SLE":     "#F58518",   # muted orange
    #"Treated_SLE":     "#E45756",   # soft red
    "Treated_SLE": "#6BAF92",  # soft sage green
    "Flare_SLE":       "#B22222"    # deep flare red
}
sex_colors = {"Male": "#4C72B0", "Female": "#DD8452"}

def plot_embedding(embedding, meta, title, pdf):
    df = pd.DataFrame(embedding, index=meta.index, columns=["Dim1", "Dim2"])
    #
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    #
    # Group
    ax = axes[0]
    for g in meta["Group"].unique():
        m = meta["Group"] == g
        ax.scatter(df.loc[m, "Dim1"], df.loc[m, "Dim2"],
                   s=25, alpha=0.8, label=g,
                   color=group_colors.get(g, "gray"))
    #
    ax.set_title("Group")
    ax.legend()
    #
    # Age
    ax = axes[1]
    sca = ax.scatter(df["Dim1"], df["Dim2"],
                     c=meta["Age"], cmap="viridis",
                     s=25, alpha=0.8)
    ax.set_title("Age")
    plt.colorbar(sca, ax=ax)
    #
    # Sex
    ax = axes[2]
    for s in meta["Sex"].unique():
        m = meta["Sex"] == s
        ax.scatter(df.loc[m, "Dim1"], df.loc[m, "Dim2"],
                   s=25, alpha=0.8, label=s,
                   color=sex_colors.get(s, "gray"))
    #
    ax.set_title("Sex")
    ax.legend()
    #
    # Batch
    ax = axes[3]
    batches = meta["batch_cov"].unique()
    palette = sns.color_palette("tab10", len(batches))
    batch_colors = dict(zip(batches, palette))
    for b in batches:
        m = meta["batch_cov"] == b
        ax.scatter(df.loc[m, "Dim1"], df.loc[m, "Dim2"],
                   s=25, alpha=0.8, color=batch_colors[b])
    #
    ax.set_title("Batch")
    #
    for ax in axes:
        ax.set_xlabel(f"{title}1")
        ax.set_ylabel(f"{title}2")
        ax.spines[['top','right']].set_visible(False)
    #
    plt.suptitle(f"{title} – FloREN embeddings", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    #
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

#pdf_path = "C:/Users/Inigo/Desktop/FloREN3.0/perez_output/perez_floren_UMAP_tSNE_PHATE_PCA.pdf"
pdf_path = "C:/Users/Inigo/Desktop/FloREN3.0/perez_output/perez_floren_full_status_specific.pdf"
with PdfPages(pdf_path) as pdf:
    for method, emb in embeddings.items():
        plot_embedding(emb, meta, method, pdf)










################################################################
#
#                            SAMPLECLR
#
################################################################

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

import scanpy as sc
import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import phate

l_128 = pd.read_csv("C:/Users/Inigo/Desktop/Samp_Emb_benchmark/SampleCLR/perez_sampleclr_tiny_correct.csv")
X = l_128.iloc[:,1:].values

embeddings = {}

# UMAP
embeddings["UMAP"] = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    metric="euclidean",
    random_state=42
).fit_transform(X)

# t-SNE
embeddings["tSNE"] = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    random_state=42
).fit_transform(X)

# PHATE
embeddings["PHATE"] = phate.PHATE(
    n_components=2,
    knn=15,
    random_state=42
).fit_transform(X)

# PCA
embeddings["PCA"] = PCA(n_components=2).fit_transform(X)

adata_path = "C:/Users/Inigo/Desktop/FloREN/Perez/GSE174188_CLUES1_adjusted.h5ad"
adata = sc.read(adata_path)

all_ids = list(adata.obs["ind_cov"].unique())
meta = (
    adata.obs.groupby("ind_cov")[["Age", "Sex", "batch_cov", "Status", "SLE_status"]]
    .first()
    .loc[all_ids]
)

metadata = pd.read_csv("C:/Users/Inigo/Desktop/FloREN/Perez/samples_metadata_unique.csv")
metadata.columns = ["Unnamed: 0","0","patient_id","group","Age","Sex","Ethnicity","Pool","Batch","Age_Group"]
metadata["group"] = metadata["group"].astype("category").cat.codes

meta = meta.reindex(metadata["patient_id"])
meta.index = metadata["patient_id"]

embeddings_patient_id = (
    l_128["Unnamed: 0"]
    #.str.split("__")
    #.str[0]
)

meta = (meta
    .loc[embeddings_patient_id]
    .reset_index()
)

meta.index = meta["patient_id"]
#groups = []
#for pid in meta.index:
#    if str(pid).startswith(("HC-", "IGTB")):
#        groups.append("HC")
#    elif str(pid).startswith("FLARE"):
#        groups.append("FLARE")
#    else:
#        groups.append("SLE")

groups = meta["Status"].astype(str) + "_" + meta["SLE_status"].astype(str)
meta["Group"] = pd.Categorical(groups)
meta["Age"] = pd.to_numeric(meta["Age"], errors="coerce")
meta["Sex"] = meta["Sex"].astype(str)
meta["batch_cov"] = meta["batch_cov"].astype(str)

#group_colors = {"HC": "#1f77b4", "SLE": "#ff7f0e", "FLARE": "#2ca02c"}
group_colors = {
    "Healthy_Healthy": "#4C78A8",   # calm blue
    "Managed_SLE":     "#F58518",   # muted orange
    #"Treated_SLE":     "#E45756",   # soft red
    "Treated_SLE": "#6BAF92",  # soft sage green
    "Flare_SLE":       "#B22222"    # deep flare red
}
sex_colors = {"Male": "#4C72B0", "Female": "#DD8452"}

def plot_embedding(embedding, meta, title, pdf):
    df = pd.DataFrame(embedding, index=meta.index, columns=["Dim1", "Dim2"])
    #
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    #
    # Group
    ax = axes[0]
    for g in meta["Group"].unique():
        m = meta["Group"] == g
        ax.scatter(df.loc[m, "Dim1"], df.loc[m, "Dim2"],
                   s=25, alpha=0.8, label=g,
                   color=group_colors.get(g, "gray"))
    #
    ax.set_title("Group")
    ax.legend(fontsize=6)
    #
    # Age
    ax = axes[1]
    sca = ax.scatter(df["Dim1"], df["Dim2"],
                     c=meta["Age"], cmap="viridis",
                     s=25, alpha=0.8)
    ax.set_title("Age")
    plt.colorbar(sca, ax=ax)
    #
    # Sex
    ax = axes[2]
    for s in meta["Sex"].unique():
        m = meta["Sex"] == s
        ax.scatter(df.loc[m, "Dim1"], df.loc[m, "Dim2"],
                   s=25, alpha=0.8, label=s,
                   color=sex_colors.get(s, "gray"))
    #
    ax.set_title("Sex")
    ax.legend(fontsize=6)
    #
    # Batch
    ax = axes[3]
    batches = meta["batch_cov"].unique()
    palette = sns.color_palette("tab10", len(batches))
    batch_colors = dict(zip(batches, palette))
    for b in batches:
        m = meta["batch_cov"] == b
        ax.scatter(df.loc[m, "Dim1"], df.loc[m, "Dim2"],
                   s=25, alpha=0.8, color=batch_colors[b])
    #
    ax.set_title("Batch")
    #
    for ax in axes:
        ax.set_xlabel(f"{title}1")
        ax.set_ylabel(f"{title}2")
        ax.spines[['top','right']].set_visible(False)
    #
    plt.suptitle(f"{title} – FloREN embeddings", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    #
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

#pdf_path = "C:/Users/Inigo/Desktop/Samp_Emb_benchmark/perez_sampleclr_UMAP_tSNE_PHATE_PCA.pdf"
pdf_path = "C:/Users/Inigo/Desktop/Samp_Emb_benchmark/perez_sampleclr_full_status_specific.pdf"
with PdfPages(pdf_path) as pdf:
    for method, emb in embeddings.items():
        plot_embedding(emb, meta, method, pdf)










################################################################
#
#                            MRVI
#
################################################################

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

import scanpy as sc
import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import phate

l_128 = pd.read_csv("C:/Users/Inigo/Desktop/Samp_Emb_benchmark/mrvi/perez_mrvi_zmean_representations.csv")
X = l_128.iloc[:,1:].values

embeddings = {}

# UMAP
embeddings["UMAP"] = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    metric="euclidean",
    random_state=42
).fit_transform(X)

# t-SNE
embeddings["tSNE"] = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    random_state=42
).fit_transform(X)

# PHATE
embeddings["PHATE"] = phate.PHATE(
    n_components=2,
    knn=15,
    random_state=42
).fit_transform(X)

# PCA
embeddings["PCA"] = PCA(n_components=2).fit_transform(X)

adata_path = "C:/Users/Inigo/Desktop/FloREN/Perez/GSE174188_CLUES1_adjusted.h5ad"
adata = sc.read(adata_path)

all_ids = list(adata.obs["ind_cov"].unique())
meta = (
    adata.obs.groupby("ind_cov")[["Age", "Sex", "batch_cov", "Status", "SLE_status"]]
    .first()
    .loc[all_ids]
)

metadata = pd.read_csv("C:/Users/Inigo/Desktop/FloREN/Perez/samples_metadata_unique.csv")
metadata.columns = ["Unnamed: 0","0","patient_id","group","Age","Sex","Ethnicity","Pool","Batch","Age_Group"]
metadata["group"] = metadata["group"].astype("category").cat.codes

meta = meta.reindex(metadata["patient_id"])
meta.index = metadata["patient_id"]

embeddings_patient_id = (
    l_128["sample_id"]
    #.str.split("__")
    #.str[0]
)

meta = (meta
    .loc[embeddings_patient_id]
    .reset_index()
)

meta.index = meta["patient_id"]
groups = []
for pid in meta.index:
    if str(pid).startswith(("HC-", "IGTB")):
        groups.append("HC")
    elif str(pid).startswith("FLARE"):
        groups.append("FLARE")
    else:
        groups.append("SLE")

groups = meta["Status"].astype(str) + "_" + meta["SLE_status"].astype(str)
meta["Group"] = pd.Categorical(groups)
meta["Age"] = pd.to_numeric(meta["Age"], errors="coerce")
meta["Sex"] = meta["Sex"].astype(str)
meta["batch_cov"] = meta["batch_cov"].astype(str)

#group_colors = {"HC": "#1f77b4", "SLE": "#ff7f0e", "FLARE": "#2ca02c"}
group_colors = {
    "Healthy_Healthy": "#4C78A8",   # calm blue
    "Managed_SLE":     "#F58518",   # muted orange
    #"Treated_SLE":     "#E45756",   # soft red
    "Treated_SLE": "#6BAF92",  # soft sage green
    "Flare_SLE":       "#B22222"    # deep flare red
}
sex_colors = {"Male": "#4C72B0", "Female": "#DD8452"}

def plot_embedding(embedding, meta, title, pdf):
    df = pd.DataFrame(embedding, index=meta.index, columns=["Dim1", "Dim2"])
    #
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    #
    # Group
    ax = axes[0]
    for g in meta["Group"].unique():
        m = meta["Group"] == g
        ax.scatter(df.loc[m, "Dim1"], df.loc[m, "Dim2"],
                   s=25, alpha=0.8, label=g,
                   color=group_colors.get(g, "gray"))
    #
    ax.set_title("Group")
    ax.legend(fontsize=6)
    #
    # Age
    ax = axes[1]
    sca = ax.scatter(df["Dim1"], df["Dim2"],
                     c=meta["Age"], cmap="viridis",
                     s=25, alpha=0.8)
    ax.set_title("Age")
    plt.colorbar(sca, ax=ax)
    #
    # Sex
    ax = axes[2]
    for s in meta["Sex"].unique():
        m = meta["Sex"] == s
        ax.scatter(df.loc[m, "Dim1"], df.loc[m, "Dim2"],
                   s=25, alpha=0.8, label=s,
                   color=sex_colors.get(s, "gray"))
    #
    ax.set_title("Sex")
    ax.legend(fontsize=6)
    #
    # Batch
    ax = axes[3]
    batches = meta["batch_cov"].unique()
    palette = sns.color_palette("tab10", len(batches))
    batch_colors = dict(zip(batches, palette))
    for b in batches:
        m = meta["batch_cov"] == b
        ax.scatter(df.loc[m, "Dim1"], df.loc[m, "Dim2"],
                   s=25, alpha=0.8, color=batch_colors[b])
    #
    ax.set_title("Batch")
    #
    for ax in axes:
        ax.set_xlabel(f"{title}1")
        ax.set_ylabel(f"{title}2")
        ax.spines[['top','right']].set_visible(False)
    #
    plt.suptitle(f"{title} – FloREN embeddings", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    #
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

#pdf_path = "C:/Users/Inigo/Desktop/Samp_Emb_benchmark/perez_mrvi_UMAP_tSNE_PHATE_PCA.pdf"
pdf_path = "C:/Users/Inigo/Desktop/Samp_Emb_benchmark/perez_mrvi_full_status_specific.pdf"
with PdfPages(pdf_path) as pdf:
    for method, emb in embeddings.items():
        plot_embedding(emb, meta, method, pdf)










################################################################
#
#                          PSEUDOBULK
#
################################################################

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

import scanpy as sc
import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import phate

l_128 = pd.read_csv("C:/Users/Inigo/Desktop/Samp_Emb_benchmark/pseudobulk/perez_pseudobulk_embs.csv")
X = l_128.iloc[:,1:].values

embeddings = {}

# UMAP
embeddings["UMAP"] = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    metric="euclidean",
    random_state=42
).fit_transform(X)

# t-SNE
embeddings["tSNE"] = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    random_state=42
).fit_transform(X)

# PHATE
embeddings["PHATE"] = phate.PHATE(
    n_components=2,
    knn=15,
    random_state=42
).fit_transform(X)

# PCA
embeddings["PCA"] = PCA(n_components=2).fit_transform(X)

adata_path = "C:/Users/Inigo/Desktop/FloREN/Perez/GSE174188_CLUES1_adjusted.h5ad"
adata = sc.read(adata_path)

all_ids = list(adata.obs["ind_cov"].unique())
meta = (
    adata.obs.groupby("ind_cov")[["Age", "Sex", "batch_cov", "Status", "SLE_status"]]
    .first()
    .loc[all_ids]
)

metadata = pd.read_csv("C:/Users/Inigo/Desktop/FloREN/Perez/samples_metadata_unique.csv")
metadata.columns = ["Unnamed: 0","0","patient_id","group","Age","Sex","Ethnicity","Pool","Batch","Age_Group"]
metadata["group"] = metadata["group"].astype("category").cat.codes

meta = meta.reindex(metadata["patient_id"])
meta.index = metadata["patient_id"]

embeddings_patient_id = (
    l_128["sample_id"]
    #.str.split("__")
    #.str[0]
)

meta = (meta
    .loc[embeddings_patient_id]
    .reset_index()
)

meta.index = meta["patient_id"]
groups = []
for pid in meta.index:
    if str(pid).startswith(("HC-", "IGTB")):
        groups.append("HC")
    elif str(pid).startswith("FLARE"):
        groups.append("FLARE")
    else:
        groups.append("SLE")

groups = meta["Status"].astype(str) + "_" + meta["SLE_status"].astype(str)
meta["Group"] = pd.Categorical(groups)
meta["Age"] = pd.to_numeric(meta["Age"], errors="coerce")
meta["Sex"] = meta["Sex"].astype(str)
meta["batch_cov"] = meta["batch_cov"].astype(str)

#group_colors = {"HC": "#1f77b4", "SLE": "#ff7f0e", "FLARE": "#2ca02c"}
group_colors = {
    "Healthy_Healthy": "#4C78A8",   # calm blue
    "Managed_SLE":     "#F58518",   # muted orange
    #"Treated_SLE":     "#E45756",   # soft red
    "Treated_SLE": "#6BAF92",  # soft sage green
    "Flare_SLE":       "#B22222"    # deep flare red
}
sex_colors = {"Male": "#4C72B0", "Female": "#DD8452"}

def plot_embedding(embedding, meta, title, pdf):
    df = pd.DataFrame(embedding, index=meta.index, columns=["Dim1", "Dim2"])
    #
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    #
    # Group
    ax = axes[0]
    for g in meta["Group"].unique():
        m = meta["Group"] == g
        ax.scatter(df.loc[m, "Dim1"], df.loc[m, "Dim2"],
                   s=25, alpha=0.8, label=g,
                   color=group_colors.get(g, "gray"))
    #
    ax.set_title("Group")
    ax.legend(fontsize=6)
    #
    # Age
    ax = axes[1]
    sca = ax.scatter(df["Dim1"], df["Dim2"],
                     c=meta["Age"], cmap="viridis",
                     s=25, alpha=0.8)
    ax.set_title("Age")
    plt.colorbar(sca, ax=ax)
    #
    # Sex
    ax = axes[2]
    for s in meta["Sex"].unique():
        m = meta["Sex"] == s
        ax.scatter(df.loc[m, "Dim1"], df.loc[m, "Dim2"],
                   s=25, alpha=0.8, label=s,
                   color=sex_colors.get(s, "gray"))
    #
    ax.set_title("Sex")
    ax.legend(fontsize=6)
    #
    # Batch
    ax = axes[3]
    batches = meta["batch_cov"].unique()
    palette = sns.color_palette("tab10", len(batches))
    batch_colors = dict(zip(batches, palette))
    for b in batches:
        m = meta["batch_cov"] == b
        ax.scatter(df.loc[m, "Dim1"], df.loc[m, "Dim2"],
                   s=25, alpha=0.8, color=batch_colors[b])
    #
    ax.set_title("Batch")
    #
    for ax in axes:
        ax.set_xlabel(f"{title}1")
        ax.set_ylabel(f"{title}2")
        ax.spines[['top','right']].set_visible(False)
    #
    plt.suptitle(f"{title} – FloREN embeddings", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    #
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

#pdf_path = "C:/Users/Inigo/Desktop/Samp_Emb_benchmark/perez_pseudobulk_UMAP_tSNE_PHATE_PCA.pdf"
pdf_path = "C:/Users/Inigo/Desktop/Samp_Emb_benchmark/perez_pseudobulk_full_status_specific.pdf"
with PdfPages(pdf_path) as pdf:
    for method, emb in embeddings.items():
        plot_embedding(emb, meta, method, pdf)










################################################################
#
#                          GLOSCOPE
#
################################################################

from sklearn.manifold import TSNE, MDS, Isomap, SpectralEmbedding

l_128 = pd.read_csv("C:/Users/Inigo/Desktop/Samp_Emb_benchmark/gloscope/perez_gloscope_gmm_divergence.csv")
X = l_128.iloc[:,1:].values


embeddings = {}

# MDS
embeddings["MDS"] = MDS(
    n_components=2,
    dissimilarity='precomputed',
    random_state=42,
    normalized_stress='auto'
).fit_transform(X)
# Isomap
embeddings["Isomap"] = Isomap(
    n_components=2,
    metric='precomputed',
    n_neighbors=10
).fit_transform(X)
# Spectral Embedding
embeddings["Spectral"] = SpectralEmbedding(
    n_components=2,
    affinity='precomputed',
    random_state=42
).fit_transform(X)

all_ids = list(adata.obs["ind_cov"].unique())
meta = (
    adata.obs.groupby("ind_cov")[["Age", "Sex", "batch_cov", "Status", "SLE_status"]]
    .first()
    .loc[all_ids]
)
metadata = pd.read_csv("C:/Users/Inigo/Desktop/FloREN/Perez/samples_metadata_unique.csv")
metadata.columns = ["Unnamed: 0","0","patient_id","group","Age","Sex","Ethnicity","Pool","Batch","Age_Group"]
metadata["group"] = metadata["group"].astype("category").cat.codes
meta = meta.reindex(metadata["patient_id"])
meta.index = metadata["patient_id"]
embeddings_patient_id = (
    l_128["Unnamed: 0"]
    #.str.split("__")
    #.str[0]
)
meta = (meta
    .loc[embeddings_patient_id]
    .reset_index()
)
meta.index = meta["patient_id"]
groups = []
for pid in meta.index:
    if str(pid).startswith(("HC-", "IGTB")):
        groups.append("HC")
    elif str(pid).startswith("FLARE"):
        groups.append("FLARE")
    else:
        groups.append("SLE")

groups = meta["Status"].astype(str) + "_" + meta["SLE_status"].astype(str)
meta["Group"] = pd.Categorical(groups)
meta["Age"] = pd.to_numeric(meta["Age"], errors="coerce")
meta["Sex"] = meta["Sex"].astype(str)
meta["batch_cov"] = meta["batch_cov"].astype(str)

group_colors = {"HC": "#1f77b4", "SLE": "#ff7f0e", "FLARE": "#2ca02c"}
group_colors = {
    "Healthy_Healthy": "#4C78A8",   # calm blue
    "Managed_SLE":     "#F58518",   # muted orange
    #"Treated_SLE":     "#E45756",   # soft red
    "Treated_SLE": "#6BAF92",  # soft sage green
    "Flare_SLE":       "#B22222"    # deep flare red
}
sex_colors = {"Male": "#4C72B0", "Female": "#DD8452"}

def plot_embedding(embedding, meta, title, pdf):
    df = pd.DataFrame(embedding, index=meta.index, columns=["Dim1", "Dim2"])
    #
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    #
    # Group
    ax = axes[0]
    for g in meta["Group"].unique():
        m = meta["Group"] == g
        ax.scatter(df.loc[m, "Dim1"], df.loc[m, "Dim2"],
                   s=25, alpha=0.8, label=g,
                   color=group_colors.get(g, "gray"))
    #
    ax.set_title("Group")
    ax.legend(fontsize=6)
    #
    # Age
    ax = axes[1]
    sca = ax.scatter(df["Dim1"], df["Dim2"],
                     c=meta["Age"], cmap="viridis",
                     s=25, alpha=0.8)
    ax.set_title("Age")
    plt.colorbar(sca, ax=ax)
    #
    # Sex
    ax = axes[2]
    for s in meta["Sex"].unique():
        m = meta["Sex"] == s
        ax.scatter(df.loc[m, "Dim1"], df.loc[m, "Dim2"],
                   s=25, alpha=0.8, label=s,
                   color=sex_colors.get(s, "gray"))
    #
    ax.set_title("Sex")
    ax.legend(fontsize=6)
    #
    # Batch
    ax = axes[3]
    batches = meta["batch_cov"].unique()
    palette = sns.color_palette("tab10", len(batches))
    batch_colors = dict(zip(batches, palette))
    for b in batches:
        m = meta["batch_cov"] == b
        ax.scatter(df.loc[m, "Dim1"], df.loc[m, "Dim2"],
                   s=25, alpha=0.8, color=batch_colors[b])
    #
    ax.set_title("Batch")
    #
    for ax in axes:
        ax.set_xlabel(f"{title}1")
        ax.set_ylabel(f"{title}2")
        ax.spines[['top','right']].set_visible(False)
    #
    plt.suptitle(f"{title} – FloREN embeddings", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    #
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

pdf_path = "C:/Users/Inigo/Desktop/Samp_Emb_benchmark/perez_gloscope_MDS_ISOMAP_SPECTRAL.pdf"
#pdf_path = "C:/Users/Inigo/Desktop/Samp_Emb_benchmark/perez_pseudobulk_full_status_specific.pdf"
with PdfPages(pdf_path) as pdf:
    for method, emb in embeddings.items():
        plot_embedding(emb, meta, method, pdf)






################################################################
#
#                       PATIENTS HEATMAPS
#
################################################################

l_128 = pd.read_csv("/Users/Inigo/Desktop/FloREN3.0/perez_output/l_128.csv")
X = l_128.iloc[:,1:].values

adata_path = "C:/Users/Inigo/Desktop/FloREN/Perez/GSE174188_CLUES1_adjusted.h5ad"
adata = sc.read(adata_path)

all_ids = list(adata.obs["ind_cov"].unique())
meta = (
    adata.obs.groupby("ind_cov")[["Age", "Sex", "batch_cov", "Status", "SLE_status"]]
    .first()
    .loc[all_ids]
)

metadata = pd.read_csv("C:/Users/Inigo/Desktop/FloREN/Perez/samples_metadata_unique.csv")
metadata.columns = ["Unnamed: 0","0","patient_id","group","Age","Sex","Ethnicity","Pool","Batch","Age_Group"]
metadata["group"] = metadata["group"].astype("category").cat.codes

meta = meta.reindex(metadata["patient_id"])
meta.index = metadata["patient_id"]

groups = []
for pid in meta.index:
    if str(pid).startswith(("HC-", "IGTB")):
        groups.append("HC")
    elif str(pid).startswith("FLARE"):
        groups.append("FLARE")
    else:
        groups.append("SLE")

#groups = meta["Status"].astype(str) + "_" + meta["SLE_status"].astype(str)
meta["Group"] = pd.Categorical(groups)
meta["Age"] = pd.to_numeric(meta["Age"], errors="coerce")
meta["Sex"] = meta["Sex"].astype(str)
meta["batch_cov"] = meta["batch_cov"].astype(str)

def row_zscore(X):
    """
    Row-wise z-score normalization
    """
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std[std == 0] = 1
    return (X - mean) / std

import sklearn
X_cosine = sklearn.metrics.pairwise.cosine_similarity(X)
X_euclidean = sklearn.metrics.pairwise.euclidean_distances(row_zscore(X))
#X_euclidean = sklearn.metrics.pairwise.euclidean_distances(X)
X_pearson = np.corrcoef(X)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

# ---------- 1. Prepare metadata annotations ----------
# Define annotation colors (edit if needed)
group_colors = {"HC": "#1f77b4", "SLE": "#ff7f0e", "FLARE": "#2ca02c"}
#group_colors = {
#    "HC": "#4C78A8",
#    "SLE": "#E45756"
#}
sex_colors = {
    "Female": "#E17C9A",
    "Male": "#4C9BE8"
}
batch_colors = dict(zip(
    meta.batch_cov.unique(),
    sns.color_palette("Set2", len(meta.batch_cov.unique()))
))
# Create row color dataframe
row_colors = pd.DataFrame({
    "Group": meta["Group"].map(group_colors),
    "Sex": meta["Sex"].map(sex_colors),
    "Batch": meta["batch_cov"].map(batch_colors)
})

# ---------- 2. Clustering helper ----------
def plot_clustermap(matrix, title, pdf):
    # Convert similarity to distance if needed
    matrix_df = pd.DataFrame(
        matrix,
        #row_zscore(matrix),
        index=meta.index,
        columns=meta.index
    )
    if "cosine" in title.lower() or "pearson" in title.lower():
        dist = 1 - matrix
    else:
        dist = matrix
    #
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0)
    linkage_rows = linkage(squareform(dist), method='average')
    linkage_cols = linkage(squareform(dist), method='average')
    g = sns.clustermap(
        matrix_df,
        row_linkage=linkage_rows,
        col_linkage=linkage_cols,
        row_colors=row_colors,
        cmap="vlag",
        figsize=(10, 10),
        xticklabels=False,
        yticklabels=False,
        dendrogram_ratio=(0.08, 0.08),
        cbar_pos=(0.02, 0.8, 0.03, 0.15)
    )
    g.fig.suptitle(title, fontsize=14, fontweight="bold")
    g.fig.tight_layout()
    pdf.savefig(g.fig)
    plt.close(g.fig)

# ---------- 3. Generate PDF ----------
#with PdfPages("C:/Users/Inigo/Desktop/FloREN3.0/perez_output/distance_heatmaps.pdf") as pdf:
with PdfPages("C:/Users/Inigo/Desktop/FloREN3.0/perez_output/distance_heatmaps_zscore_clean.pdf") as pdf:
    plot_clustermap(X_cosine, "Cosine Similarity", pdf)
    plot_clustermap(X_euclidean, "Euclidean Distance", pdf)
    plot_clustermap(X_pearson, "Pearson Correlation", pdf)

print("Saved as distance_heatmaps.pdf")
pd.DataFrame(X_pearson, index=meta.index, columns=meta.index).to_csv("C:/Users/Inigo/Desktop/FloREN3.0/perez_output/X_pearson.csv")
pd.DataFrame(X_euclidean, index=meta.index, columns=meta.index).to_csv("C:/Users/Inigo/Desktop/FloREN3.0/perez_output/X_euclidean.csv")
pd.DataFrame(X_cosine, index=meta.index, columns=meta.index).to_csv("C:/Users/Inigo/Desktop/FloREN3.0/perez_output/X_cosine.csv")
meta.to_csv("C:/Users/Inigo/Desktop/FloREN3.0/perez_output/meta.csv")


def plot_ordered_heatmap(matrix, title, pdf, meta):
    """
    Plot a heatmap with rows and columns ordered by meta.Group
    """
    # Convert to DataFrame
    matrix_df = pd.DataFrame(matrix, index=meta.index, columns=meta.index)
    # Order by Group
    ordered_idx = meta.sort_values("Group").index
    matrix_df = matrix_df.loc[ordered_idx, ordered_idx]
    # Create row colors
    # You can reuse the row_colors DataFrame you defined earlier
    row_colors_ordered = row_colors.loc[ordered_idx]
    # Plot heatmap
    plt.figure(figsize=(10, 10))
    sns.heatmap(
        matrix_df,
        cmap="vlag",  # diverging color map
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"shrink": 0.5},
    )
    # Add colored bars for metadata (optional)
    # Use a small trick: matplotlib's imshow + extra axes can do this, or just skip if row_colors are sufficient
    plt.title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    # Save to PDF
    pdf.savefig()
    plt.close()


with PdfPages("C:/Users/Inigo/Desktop/FloREN3.0/perez_output/distance_heatmaps_nocluster.pdf") as pdf:
    plot_ordered_heatmap(X_cosine, "Cosine Similarity", pdf, meta)
    plot_ordered_heatmap(X_euclidean, "Euclidean Distance", pdf, meta)
    plot_ordered_heatmap(X_pearson, "Pearson Correlation", pdf, meta)










################################################################
#
#                       CELLTYPE HEATMAPS
#
################################################################

import pandas as pd
import scanpy as sc
import glob
import os

# Path to all patient embedding files
folder_path = "C:/Users/Inigo/Desktop/FloREN3.0/perez_output/floren_cell_embeddings/"
file_pattern = os.path.join(folder_path, "*_cell_embs.csv")

# Load adata and metadata once
adata_path = "C:/Users/Inigo/Desktop/FloREN/Perez/GSE174188_CLUES1_adjusted.h5ad"
adata = sc.read(adata_path)
meta = adata.obs.copy()
meta["barcode"] = meta.ind_cov.astype(str) + "__" + meta.index

# Dictionary to store aggregated embeddings
patient_agg_dict = {}

# Iterate over all files
for fpath in glob.glob(file_pattern):
    # Extract patient_id from filename
    fname = os.path.basename(fpath)  # e.g., "1004_1004_cell_embs.csv"
    patient_id = fname.split("_cell_embs.csv")[0]
    # Read patient cells embeddings (no header)
    patient_cells = pd.read_csv(fpath, header=None)
    # Filter meta for this patient
    meta_filtered = meta[meta.ind_cov.str.contains(patient_id)].reset_index(drop=True)
    # Make sure rows match number of cells in patient_cells
    if patient_cells.shape[0] != meta_filtered.shape[0]:
        print(f"Warning: Row mismatch for patient {patient_id}. Skipping...")
        continue
    #
    # Aggregate by celltype (cg_cov)
    patient_cells_agg = patient_cells.groupby(meta_filtered["cg_cov"]).mean()
    patient_cells_agg = patient_cells_agg.fillna(0)
    # Store in dict
    patient_agg_dict[patient_id] = patient_cells_agg

print(f"Processed {len(patient_agg_dict)} patients.")

all_ids = list(adata.obs["ind_cov"].unique())
meta = (
    adata.obs.groupby("ind_cov")[["Age", "Sex", "batch_cov", "Status", "SLE_status"]]
    .first()
    .loc[all_ids]
)
metadata = pd.read_csv("C:/Users/Inigo/Desktop/FloREN/Perez/samples_metadata_unique.csv")
metadata.columns = ["Unnamed: 0","0","patient_id","group","Age","Sex","Ethnicity","Pool","Batch","Age_Group"]
metadata["group"] = metadata["group"].astype("category").cat.codes
meta = meta.reindex(metadata["patient_id"])
meta.index = metadata["patient_id"]
groups = []
for pid in meta.index:
    if str(pid).startswith(("HC-", "IGTB")):
        groups.append("HC")
    else:
        groups.append("SLE")

#groups = meta["Status"].astype(str) + "_" + meta["SLE_status"].astype(str)
meta["Group"] = pd.Categorical(groups)
meta["Age"] = pd.to_numeric(meta["Age"], errors="coerce")
meta["Sex"] = meta["Sex"].astype(str)
meta["batch_cov"] = meta["batch_cov"].astype(str)

# Split patient IDs by group
hc_ids = [pid for pid in patient_agg_dict.keys() if meta.loc[pid, "Group"] == "HC"]
sle_ids = [pid for pid in patient_agg_dict.keys() if meta.loc[pid, "Group"] == "SLE"]

print(f"HC patients: {len(hc_ids)}")
print(f"SLE patients: {len(sle_ids)}")

def compute_group_mean(patient_ids):
    """
    Compute mean (celltype × 128) embedding across patients
    """
    dfs = []
    for pid in patient_ids:
        df = patient_agg_dict[pid].copy()
        df["patient_id"] = pid
        dfs.append(df)
    #
    # Concatenate all patients
    combined = pd.concat(dfs)
    # Compute mean per celltype
    group_mean = combined.iloc[:,:-1].groupby(combined.index).mean()
    return group_mean

# Compute group means
hc_mean = compute_group_mean(hc_ids)
sle_mean = compute_group_mean(sle_ids)
print("HC mean shape:", hc_mean.shape)
print("SLE mean shape:", sle_mean.shape)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.stats import pearsonr
from scipy.cluster.hierarchy import linkage, leaves_list

# -----------------------------
# Helper functions
# -----------------------------

def row_zscore(X):
    """
    Row-wise z-score normalization
    """
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std[std == 0] = 1
    return (X - mean) / std

def compute_pearson_matrix(X):
    """
    Compute Pearson correlation matrix between rows of X
    """
    corr = np.corrcoef(X)
    return corr

def cluster_matrix(M, method="ward"):
    """
    Reorder rows and columns of matrix using hierarchical clustering.
    """
    linkage_rows = linkage(M, method=method)
    linkage_cols = linkage(M.T, method=method)
    row_order = leaves_list(linkage_rows)
    col_order = leaves_list(linkage_cols)
    return M[row_order][:, col_order], row_order, col_order

def plot_three_heatmaps(metric_name, hc_mean, sle_mean, pdf):
    # Ensure same celltype order
    celltypes = sorted(set(hc_mean.index) & set(sle_mean.index))
    hc = hc_mean.loc[celltypes].values
    sle = sle_mean.loc[celltypes].values
    # ----- Compute matrices -----
    if metric_name == "cosine":
        hc_mat = cosine_similarity(hc)
        sle_mat = cosine_similarity(sle)
        cross_mat = cosine_similarity(hc, sle)
        cmap = "vlag"
        vmin, vmax = -1, 1
    elif metric_name == "euclidean":
        hc = row_zscore(hc)
        sle = row_zscore(sle)
        hc_mat = euclidean_distances(hc)
        sle_mat = euclidean_distances(sle)
        cross_mat = euclidean_distances(hc, sle)
        #cmap = "inferno_r"
        #cmap = "viridis"
        cmap = "plasma"
        vmin, vmax = None, None
    elif metric_name == "pearson":
        hc_mat = compute_pearson_matrix(hc)
        sle_mat = compute_pearson_matrix(sle)
        cross_mat = np.corrcoef(np.vstack([hc, sle]))[:len(celltypes), len(celltypes):]
        cmap = "vlag"
        vmin, vmax = -1, 1
    #
    #hc_mat = row_zscore(hc_mat)
    #sle_mat = row_zscore(sle_mat)
    #cross_mat = row_zscore(cross_mat)
    # Labels
    hc_labels = [f"{ct}_HC" for ct in celltypes]
    sle_labels = [f"{ct}_SLE" for ct in celltypes]
    # Hierarchical clustering
    #hc_mat, hc_row_order, hc_col_order = cluster_matrix(hc_mat)
    #sle_mat, sle_row_order, sle_col_order = cluster_matrix(sle_mat)
    #cross_mat, cross_row_order, cross_col_order = cluster_matrix(cross_mat)
    # Reorder labels accordingly
    hc_labels = np.array(hc_labels)
    sle_labels = np.array(sle_labels)
    #hc_labels_rows = hc_labels[hc_row_order]
    #hc_labels_cols = hc_labels[hc_col_order]
    #sle_labels_rows = sle_labels[sle_row_order]
    #sle_labels_cols = sle_labels[sle_col_order]
    #cross_rows = hc_labels[cross_row_order]
    #cross_cols = sle_labels[cross_col_order]
    # ----- Plot -----
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    # Cross HC vs SLE
    sns.heatmap(
        cross_mat,
        ax=axes[0],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        #xticklabels=cross_cols,
        #yticklabels=cross_rows,
        xticklabels=sle_labels,
        yticklabels=hc_labels,
        cbar=False
    )
    axes[0].set_title("HC vs SLE")
    # HC only
    sns.heatmap(
        hc_mat,
        ax=axes[1],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        #xticklabels=hc_labels_cols,
        #yticklabels=hc_labels_rows,
        xticklabels=hc_labels,
        yticklabels=hc_labels,
        cbar=False
    )
    axes[1].set_title("HC")
    # SLE only
    sns.heatmap(
        sle_mat,
        ax=axes[2],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        #xticklabels=sle_labels_cols,
        #yticklabels=sle_labels_rows,
        xticklabels=sle_labels,
        yticklabels=sle_labels,
        cbar=False
        #cbar_kws={
        #    "shrink": 0.6,  # smaller height
        #    "aspect": 20,  # thinner bar
        #    "pad": 0.08  # move slightly right
        #}
    )
    axes[2].set_title("SLE")
    # Put y-axis labels on right side
    for ax in axes:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.tick_params(axis='y', rotation=0)
        ax.tick_params(axis='x', rotation=90)
    #
    fig.suptitle(metric_name.upper(), fontsize=16, fontweight="bold")
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

def plot_three_heatmaps_cluster(metric_name, hc_mean, sle_mean, pdf):
    # Ensure same celltype order
    celltypes = sorted(set(hc_mean.index) & set(sle_mean.index))
    hc = hc_mean.loc[celltypes].values
    sle = sle_mean.loc[celltypes].values
    # ----- Compute matrices -----
    if metric_name == "cosine":
        hc_mat = cosine_similarity(hc)
        sle_mat = cosine_similarity(sle)
        cross_mat = cosine_similarity(hc, sle)
        cmap = "vlag"
        vmin, vmax = -1, 1
    elif metric_name == "euclidean":
        hc = row_zscore(hc)
        sle = row_zscore(sle)
        hc_mat = euclidean_distances(hc)
        sle_mat = euclidean_distances(sle)
        cross_mat = euclidean_distances(hc, sle)
        #cmap = "inferno_r"
        #cmap = "viridis"
        cmap = "plasma"
        #vmin, vmax = 0, np.max([np.max(hc_mat), np.max(sle_mat), np.max(cross_mat)])
        vmin, vmax = None, None
    elif metric_name == "pearson":
        hc_mat = compute_pearson_matrix(hc)
        sle_mat = compute_pearson_matrix(sle)
        cross_mat = np.corrcoef(np.vstack([hc, sle]))[:len(celltypes), len(celltypes):]
        cmap = "vlag"
        vmin, vmax = -1, 1
    #
    #hc_mat = row_zscore(hc_mat)
    #sle_mat = row_zscore(sle_mat)
    #cross_mat = row_zscore(cross_mat)
    # Labels
    hc_labels = [f"{ct}_HC" for ct in celltypes]
    sle_labels = [f"{ct}_SLE" for ct in celltypes]
    # Hierarchical clustering
    hc_mat, hc_row_order, hc_col_order = cluster_matrix(hc_mat)
    sle_mat, sle_row_order, sle_col_order = cluster_matrix(sle_mat)
    cross_mat, cross_row_order, cross_col_order = cluster_matrix(cross_mat)
    # Reorder labels accordingly
    hc_labels = np.array(hc_labels)
    sle_labels = np.array(sle_labels)
    hc_labels_rows = hc_labels[hc_row_order]
    hc_labels_cols = hc_labels[hc_col_order]
    sle_labels_rows = sle_labels[sle_row_order]
    sle_labels_cols = sle_labels[sle_col_order]
    cross_rows = hc_labels[cross_row_order]
    cross_cols = sle_labels[cross_col_order]
    # ----- Plot -----
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    # Cross HC vs SLE
    sns.heatmap(
        cross_mat,
        ax=axes[0],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        xticklabels=cross_cols,
        yticklabels=cross_rows,
        #xticklabels=sle_labels,
        #yticklabels=hc_labels,
        cbar=True
    )
    axes[0].set_title("HC vs SLE")
    # HC only
    sns.heatmap(
        hc_mat,
        ax=axes[1],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        xticklabels=hc_labels_cols,
        yticklabels=hc_labels_rows,
        #xticklabels=hc_labels,
        #yticklabels=hc_labels,
        cbar=True
    )
    axes[1].set_title("HC")
    # SLE only
    sns.heatmap(
        sle_mat,
        ax=axes[2],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        xticklabels=sle_labels_cols,
        yticklabels=sle_labels_rows,
        #xticklabels=sle_labels,
        #yticklabels=sle_labels,
        cbar=True
        #cbar_kws={
        #    "shrink": 0.6,  # smaller height
        #    "aspect": 20,  # thinner bar
        #    "pad": 0.08  # move slightly right
        #}
    )
    axes[2].set_title("SLE")
    # Put y-axis labels on right side
    for ax in axes:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.tick_params(axis='y', rotation=0)
        ax.tick_params(axis='x', rotation=90)
    #
    fig.suptitle(metric_name.upper(), fontsize=16, fontweight="bold")
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

# -----------------------------
# Generate PDF
# -----------------------------
output_path = "C:/Users/Inigo/Desktop/FloREN3.0/perez_output/group_level_heatmaps_cbar_plasma.pdf"
with PdfPages(output_path) as pdf:
    #plot_three_heatmaps("cosine", hc_mean, sle_mean, pdf)
    #plot_three_heatmaps("euclidean", hc_mean, sle_mean, pdf)
    #plot_three_heatmaps("pearson", hc_mean, sle_mean, pdf)
    plot_three_heatmaps_cluster("cosine", hc_mean, sle_mean, pdf)
    plot_three_heatmaps_cluster("euclidean", hc_mean, sle_mean, pdf)
    plot_three_heatmaps_cluster("pearson", hc_mean, sle_mean, pdf)

print("PDF saved:", output_path)










################################################################
#
#                      CLUSTERING METRICS
#
################################################################

import numpy as np
import pandas as pd
from sklearn.metrics import (
    calinski_harabasz_score,
    silhouette_score,
    davies_bouldin_score
)

# -----------------------------
# Choose X
# -----------------------------
# FloREN
l_128 = pd.read_csv("/Users/Inigo/Desktop/FloREN3.0/perez_output/l_128.csv")
X = l_128.iloc[:,1:].values

# SampleCLR
l_128 = pd.read_csv("C:/Users/Inigo/Desktop/Samp_Emb_benchmark/SampleCLR/perez_sampleclr_tiny_correct.csv")
X = l_128.iloc[:,1:].values

# MrVI
l_128 = pd.read_csv("C:/Users/Inigo/Desktop/Samp_Emb_benchmark/mrvi/perez_mrvi_zmean_representations.csv")
X = l_128.iloc[:,1:].values

# Pseudobulk
l_128 = pd.read_csv("C:/Users/Inigo/Desktop/Samp_Emb_benchmark/pseudobulk/perez_pseudobulk_embs.csv")
X = l_128.iloc[:,1:].values

# GloScope
l_128 = pd.read_csv("C:/Users/Inigo/Desktop/Samp_Emb_benchmark/gloscope/perez_gloscope_embs.csv")
X = l_128.iloc[:,3:].values

# -----------------------------
# Ensure alignment
# -----------------------------
all_ids = list(adata.obs["ind_cov"].unique())
meta = (
    adata.obs.groupby("ind_cov")[["Age", "Sex", "batch_cov", "Status", "SLE_status"]]
    .first()
    .loc[all_ids]
)
metadata = pd.read_csv("C:/Users/Inigo/Desktop/FloREN/Perez/samples_metadata_unique.csv")
metadata.columns = ["Unnamed: 0","0","patient_id","group","Age","Sex","Ethnicity","Pool","Batch","Age_Group"]
metadata["group"] = metadata["group"].astype("category").cat.codes
meta = meta.reindex(metadata["patient_id"])
meta.index = metadata["patient_id"]
embeddings_patient_id = (
    l_128["sample_id"],
    #l_128["Unnamed: 0"],
    #.str.split("__")
    #.str[0]
)
meta = (meta
    .loc[embeddings_patient_id]
    .reset_index()
)
meta.index = meta["patient_id"]
groups = []
for pid in meta.index:
    if str(pid).startswith(("HC-", "IGTB")):
        groups.append("HC")
    elif str(pid).startswith("FLARE"):
        groups.append("FLARE")
    else:
        groups.append("SLE")

meta["Group"] = pd.Categorical(groups)
meta["Age"] = pd.to_numeric(meta["Age"], errors="coerce")
meta["Sex"] = meta["Sex"].astype(str)
meta["batch_cov"] = meta["batch_cov"].astype(str)
#meta.index = meta["patient_id"]
labels = meta.loc[meta.index, "Group"].values
# -----------------------------
# Compute metrics
# -----------------------------
ch_score = calinski_harabasz_score(X, labels)
asw_score = silhouette_score(
    X,
    labels,
    metric="euclidean"
)
dbs_score = davies_bouldin_score(X, labels)

# Print results
print(f"Calinski-Harabasz (CH): {ch_score:.3f}")
print(f"Average Silhouette Width (ASW): {asw_score:.3f}")
print(f"Davies-Bouldin Score (DBS): {dbs_score:.3f}")




import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# Data
data = {
    "Method": ["Pseudobulk", "GloScope", "MrVI", "SampleCLR", "FloREN"],
    "CH": [28.773, 30.318, 29.726, 51.884, 41.348],
    "ASW": [0.110, 0.054, 0.109, 0.124, 0.037],
    "DBS": [3.713, 3.704, 4.665, 2.122, 4.221],
}

df = pd.DataFrame(data)
df["ASW"] = 1-df["ASW"]
# Publication styling
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "font.family": "sans-serif",
    "pdf.fonttype": 42,  # editable in Illustrator
})
# Subtle, high-end color palette
colors = {
    "Pseudobulk": "#7f7f7f",
    "GloSope": "#4c72b0",
    "MrVI": "#55a868",
    "SampleCLR": "#c44e52",
    "FloREN": "#000000",  # highlight method
}
colors = {
    "FloREN": "#2ca02c",  # highlight method
    "SampleCLR": "#1f77b4",
    "MrVI": "#f1c40f",
    "GloScope": "#ff7f0e",
    "Pseudobulk": "#d62728"
}
bar_colors = [colors[m] for m in df["Method"]]
# Create PDF
output_path = "C:/Users/Inigo/Desktop/FloREN3.0/perez_output/Nature_Methods_cluster_metrics.pdf"
with PdfPages(output_path) as pdf:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))
    metrics = ["CH", "ASW", "DBS"]
    titles = ["Calinski–Harabasz",
              "Average Silhouette Width",
              "Davies–Bouldin"]
    for ax, metric, title in zip(axes, metrics, titles):
        bars = ax.bar(
            df["Method"],
            df[metric],
            color=bar_colors,
            edgecolor="black",
            linewidth=0.5
        )
        ax.set_title(title, pad=8)
        # Rotate method labels for clarity
        ax.set_xticklabels(df["Method"], rotation=45, ha="right")
        # Remove top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # Thin remaining spines
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.tick_params(width=0.8)
        # No grid
        ax.grid(False)
    #
    plt.subplots_adjust(wspace=0.35)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

print("PDF saved:", output_path)










################################################################
#
#                      CELL EMBEDDINGS SPACE
#
################################################################

import pandas as pd
import scanpy as sc
import glob
import os

# Path to all patient embedding files
folder_path = "C:/Users/Inigo/Desktop/FloREN3.0/perez_output/floren_cell_embeddings/"
file_pattern = os.path.join(folder_path, "*_cell_embs.csv")

# Load adata and metadata once
adata_path = "C:/Users/Inigo/Desktop/FloREN/Perez/GSE174188_CLUES1_adjusted.h5ad"
adata = sc.read(adata_path)
meta = adata.obs.copy()
meta["barcode"] = meta.ind_cov.astype(str) + "__" + meta.index

# Dictionary to store aggregated embeddings
patient_agg_dict = {}

# Iterate over all files
for fpath in glob.glob(file_pattern):
    # Extract patient_id from filename
    fname = os.path.basename(fpath)  # e.g., "1004_1004_cell_embs.csv"
    patient_id = fname.split("_cell_embs.csv")[0]
    # Read patient cells embeddings (no header)
    patient_cells = pd.read_csv(fpath, header=None)
    # Filter meta for this patient
    meta_filtered = meta[meta.ind_cov.str.contains(patient_id)].reset_index(drop=True)
    # Make sure rows match number of cells in patient_cells
    if patient_cells.shape[0] != meta_filtered.shape[0]:
        print(f"Warning: Row mismatch for patient {patient_id}. Skipping...")
        continue
    #
    # Aggregate by celltype (cg_cov)
    patient_cells_agg = patient_cells.groupby(meta_filtered["cg_cov"]).mean()
    patient_cells_agg = patient_cells_agg.fillna(0)
    # Store in dict
    patient_agg_dict[patient_id] = patient_cells_agg

print(f"Processed {len(patient_agg_dict)} patients.")

all_ids = list(adata.obs["ind_cov"].unique())
meta = (
    adata.obs.groupby("ind_cov")[["Age", "Sex", "batch_cov", "Status", "SLE_status"]]
    .first()
    .loc[all_ids]
)
metadata = pd.read_csv("C:/Users/Inigo/Desktop/FloREN/Perez/samples_metadata_unique.csv")
metadata.columns = ["Unnamed: 0","0","patient_id","group","Age","Sex","Ethnicity","Pool","Batch","Age_Group"]
metadata["group"] = metadata["group"].astype("category").cat.codes
meta = meta.reindex(metadata["patient_id"])
meta.index = metadata["patient_id"]
groups = []
for pid in meta.index:
    if str(pid).startswith(("HC-", "IGTB")):
        groups.append("HC")
    else:
        groups.append("SLE")

#groups = meta["Status"].astype(str) + "_" + meta["SLE_status"].astype(str)
meta["Group"] = pd.Categorical(groups)
meta["Age"] = pd.to_numeric(meta["Age"], errors="coerce")
meta["Sex"] = meta["Sex"].astype(str)
meta["batch_cov"] = meta["batch_cov"].astype(str)

# Split patient IDs by group
hc_ids = [pid for pid in patient_agg_dict.keys() if meta.loc[pid, "Group"] == "HC"]
sle_ids = [pid for pid in patient_agg_dict.keys() if meta.loc[pid, "Group"] == "SLE"]

print(f"HC patients: {len(hc_ids)}")
print(f"SLE patients: {len(sle_ids)}")

def compute_group_mean(patient_ids):
    """
    Compute mean (celltype × 128) embedding across patients
    """
    dfs = []
    for pid in patient_ids:
        df = patient_agg_dict[pid].copy()
        df["patient_id"] = pid
        dfs.append(df)
    #
    # Concatenate all patients
    combined = pd.concat(dfs)
    # Compute mean per celltype
    group_mean = combined.iloc[:,:-1].groupby(combined.index).mean()
    return group_mean

# Compute group means
hc_mean = compute_group_mean(hc_ids)
sle_mean = compute_group_mean(sle_ids)
print("HC mean shape:", hc_mean.shape)
print("SLE mean shape:", sle_mean.shape)
# Add labels
hc_mean["group"] = "HC"
sle_mean["group"] = "SLE"
# Keep index as celltype
hc_mean["celltype"] = hc_mean.index
sle_mean["celltype"] = sle_mean.index
# Concatenate
df = pd.concat([hc_mean, sle_mean])
# Separate features and metadata
X = df.drop(columns=["group", "celltype"]).values
meta = df[["group", "celltype"]].reset_index(drop=True)

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import phate
embeddings = {}
# UMAP
embeddings["UMAP"] = umap.UMAP(
    n_neighbors=5,
    min_dist=0.3,
    n_components=2,
    random_state=42
).fit_transform(X)
# t-SNE
embeddings["tSNE"] = TSNE(
    n_components=2,
    perplexity=5,
    init="pca",
    random_state=42
).fit_transform(X)

# PHATE
embeddings["PHATE"] = phate.PHATE(
    n_components=2,
    knn=5,
    random_state=42
).fit_transform(X)
# PCA
embeddings["PCA"] = PCA(n_components=2).fit_transform(X)

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
pdf_path = "C:/Users/Inigo/Desktop/FloREN3.0/perez_output/cell_embeddings_hc_sle.pdf"
with PdfPages(pdf_path) as pdf:
    # ---------------- PAGE 1: GROUP (HC vs SLE) ----------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for i, (name, emb) in enumerate(embeddings.items()):
        ax = axes[i]
        sns.scatterplot(
            x=emb[:, 0],
            y=emb[:, 1],
            hue=meta["group"],
            palette={"HC": "gray", "SLE": "red"},
            s=80,
            ax=ax
        )
        ax.set_title(name)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.legend(title="Group")
    #
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
    # ---------------- PAGE 2: CELLTYPE ----------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    # Color palette for cell types
    celltypes = meta["celltype"].unique()
    palette = dict(zip(celltypes, sns.color_palette("tab10", len(celltypes))))
    for i, (name, emb) in enumerate(embeddings.items()):
        ax = axes[i]
        sns.scatterplot(
            x=emb[:, 0],
            y=emb[:, 1],
            hue=meta["celltype"],
            palette=palette,
            s=80,
            ax=ax
        )
        ax.set_title(name)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.legend(title="Celltype", bbox_to_anchor=(1.05, 1), loc='upper left')
    #
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

print(f"Saved to {pdf_path}")










##############################################################
#
#                 NEW RANKED CELL TYPE PLOT
#
##############################################################

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

plt.rcParams["figure.figsize"] = (14, 6)
plt.rcParams["font.size"] = 12

# 1. Build list of patients from your earlier processing

result_dir = '/Users/Inigo/Desktop/FloREN3.0/perez_output'# Sets parent directory for output
gene_dir = result_dir+'/gene/' # Sets directory for gene output
cell_dir = result_dir+'/cell/' # Sets directory for cell output
model_dir = result_dir+'/model/' # Sets directory for model output
att_dir = result_dir+'/att/' # Sets directory for attention output
loss_dir = result_dir+'/loss/' # Sets directory for loss output
inter_dir = result_dir+'/interpretability/' # Sets directory for loss output
gene_dir = "/Users/Inigo/Desktop/FloREN3.0/perez_output/floren_gene_embeddings/"
cell_dir = "/Users/Inigo/Desktop/FloREN3.0/perez_output/floren_cell_embeddings/"
att_dir = "/Users/Inigo/Desktop/FloREN3.0/perez_output/floren_attention_scores/"
patient_emb_dir = "/Users/Inigo/Desktop/FloREN3.0/perez_output/floren_patient_embeddings/"
patient_emb_dir_split = "/Users/Inigo/Desktop/FloREN3.0/perez_output/floren_patient_embeddings/split/"
plots_dir = "/Users/Inigo/Desktop/FloREN3.0/perez_output/plots"
data_path = "C:/Users/Inigo/Desktop/FloREN/Perez/tfs"
#count_matrices_path = os.path.join(data_path, "count_matrices")
#count_matrices_path = "/home/inigo/Desktop/FloREN3.0/Binvignat/genes/"
print(f"Looking for CSV files in: {data_path}")
csv_files = glob.glob(os.path.join(data_path, "*.csv"))
print(f"Found {len(csv_files)} CSV files: {csv_files}")
# Load reference for gene names and concatenate all matrices
reference = pd.read_csv(csv_files[0])
gene_names = reference[reference.columns[0]].values
n_genes = reference.shape[0]

if n_genes < args.in_dim:
    h_n = n_genes
else:
    h_n = args.in_dim

output_path = "C:/Users/Inigo/Desktop/FloREN3.0/perez_output"
cell_names = pd.read_csv(os.path.join(output_path, "All_AUC_Cell_names.csv"))
samples_names = []
for name in cell_names['0']:
    samples_names.append(re.split('__', name, 2)[0])

files = list(set(samples_names))


patients = [d for d in os.listdir(inter_dir) if os.path.isdir(os.path.join(inter_dir, d))]
patients.sort()

print("Patients detected:", patients)

##############################################################
# 2. Cell metadata processing helper
##############################################################

meta = pd.read_csv("/Users/Inigo/Desktop/FloREN/Perez/cells_metadata.csv")
# Example: "classical monocyte__RA09_1_C_M__GGGACCTGTGCGAA-1"
meta.columns = ["Unnamed", "patient", "group", "celltype", "barcode"]
#meta_split = meta["x"].str.split("__", expand=True)
#meta["celltype"] = meta_split[0]
meta["barcode"]  = meta.patient+"__"+meta.barcode  # e.g. "GGGACCTGTGCGAA-1"
#meta["patient"]  = meta_split[1]  # e.g. "RA09_1_C_M"
meta = adata.obs.copy()
meta["barcode"] = meta.ind_cov.astype(str) + "__" + meta.index

##############################################################
# 3. Iterate patients and compute gene & cell saliency
##############################################################

all_patients_gene_df = []
all_patients_cell_df = []

for patient in patients:
    print(f"Processing patient:", patient)
    pdir = os.path.join(inter_dir, patient)
    #
    # Load gradients
    genes_grad = np.loadtxt(os.path.join(pdir, "genes_atts.csv"), delimiter=",")
    cells_grad = np.loadtxt(os.path.join(pdir, "cells_atts.csv"), delimiter=",")
    #
    # Reduce: take row mean = saliency per gene/cell
    gene_scores = genes_grad
    cell_scores = cells_grad
    #
    # -------------  GENES: no grouping needed -------------------
    #
    gene_df = pd.DataFrame({
        "gene": gene_names,
        "saliency": gene_scores,
        "patient": patient
    })
    #
    # Save optional
    #gene_df.to_csv(os.path.join(pdir, "Gene_att_saliency.csv"), index=False)
    #
    all_patients_gene_df.append(gene_df)
    #
    # -------------  CELLS: Match to metadata ---------------------
    #
    # Patient cell names from your patient_graphs construction
    cell_indices = [i for i, nm in enumerate(cell_names["0"]) if patient in nm]
    # patient_cell_barcodes = [re.split("__", cell_names["0"].iloc[i])[1] for i in cell_indices]
    patient_cell_barcodes = [cell_names["0"].iloc[i] for i in cell_indices]
    #
    # cell_scores is aligned with encoded2/cell order → same order as cell_names patient subset
    if len(patient_cell_barcodes) != len(cell_scores):
        print("WARNING: mismatch in cell counts")
    #
    # Build df
    cell_df = pd.DataFrame({
        "barcode": patient_cell_barcodes,
        "saliency": cell_scores
    })
    #
    # Merge with metadata to get cell type
    merged = cell_df.merge(meta[["barcode", "celltype"]], on="barcode", how="left")
    #
    # Group by cell type and compute mean saliency
    #grouped = merged.groupby("celltype")["saliency"].sum().sort_values(ascending=False)
    grouped = merged.groupby("celltype")["saliency"].mean().sort_values(ascending=False)
    #grouped = merged.groupby("celltype")["saliency"].quantile(0.75).sort_values(ascending=False)
    #
    celltype_df = grouped.reset_index()
    celltype_df["patient"] = patient
    #
    # Save
    #celltype_df.to_csv(os.path.join(pdir, "Celltype_LVL1_att_saliency.csv"), index=False)
    #
    all_patients_cell_df.append(celltype_df)

# ===================== 4. Combine Across Patients ====================
all_genes = pd.concat(all_patients_gene_df)
all_cells = pd.concat(all_patients_cell_df)

# Compute mean across patients
mean_gene_saliency = all_genes.groupby("gene")["saliency"].mean().sort_values(ascending=False)
mean_celltype_saliency = all_cells.groupby("celltype")["saliency"].mean().sort_values(ascending=False)
#q75_celltype_saliency = all_cells.groupby("celltype")["saliency"].quantile(0.75).sort_values(ascending=False)


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Prepare data
df_plot = mean_celltype_saliency.sort_values(ascending=True).reset_index()
# Publication settings
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.sans-serif'] = 'Arial'
# Increase height slightly for 31 categories, but keep width compact
fig, ax = plt.subplots(figsize=(4, 6))
# 1. Smaller lollipops (linewidth and markersize)
ax.hlines(y=df_plot['celltype'], xmin=43.20, xmax=df_plot['saliency'],
          color='#bcbcbc', linestyle='-', linewidth=0.8, zorder=1)
# Smaller dots (s=30) with a thinner edge
ax.scatter(df_plot['saliency'], df_plot['celltype'],
           color='#2c3e50', s=30, edgecolors='white', linewidth=0.4, zorder=2)
# 2. Refine Labels
ax.set_xlabel("Mean Percentile Saliency Score", fontsize=11, fontweight='bold')
ax.set_title("Ranked Cell Type Saliency", loc='left', fontsize=11, pad=10)
# 3. Handle the density of Y-labels
# Setting fontsize=7 or 8 is usually the 'sweet spot' for large lists in journals
ax.tick_params(axis='y', labelsize=7.5)
ax.tick_params(axis='x', labelsize=8)
# 4. Clean up spines and grid
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.xaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
# Adjust X-limit to give the data some 'breathing room'
# based on your new min (134) and max (207)
#ax.set_xlim(130, 215)
plt.tight_layout()
# To save for publication:
fig.savefig(os.path.join(
    plots_dir,"Attention_Ranking_NatureMethods_mean_mean.pdf"), dpi=300, bbox_inches='tight')










##############################################################
#
#                 CELLS ATTENTION UMAP
#
##############################################################

all_cells_attention = []
for patient in patients:
    print(f"Processing patient: {patient}")
    pdir = os.path.join(inter_dir, patient)
    # Load attention scores
    cells_grad = np.loadtxt(os.path.join(pdir, "cells_atts.csv"), delimiter=",")
    # Get indices for this patient
    cell_indices = [i for i, nm in enumerate(cell_names["0"]) if patient in nm]
    # Extract matching names
    patient_cell_names = [cell_names["0"].iloc[i] for i in cell_indices]
    # Sanity check
    if len(patient_cell_names) != len(cells_grad):
        print(f"WARNING mismatch for {patient}: names={len(patient_cell_names)} vs scores={len(cells_grad)}")
    #
    # Build dataframe
    df = pd.DataFrame({
        "full_id": patient_cell_names,   # patient__barcode
        "attention": cells_grad
    })
    all_cells_attention.append(df)

# Combine
attention_df = pd.concat(all_cells_attention, ignore_index=True)
attention_df[["patient", "barcode"]] = attention_df["full_id"].str.split("__", n=1, expand=True)
#meta["full_id"] = meta["patient"].astype(str) + "__" + meta.barcode
# Merge
meta = meta.merge(
    attention_df[["full_id", "attention"]],
    on="full_id",
    how="left"
)
# Assign back
adata.obs["attention"] = meta["attention"].values
adata.obs["attention_log"] = np.log1p(adata.obs["attention"])
adata.obs["attention_sqrt"] = np.sqrt(adata.obs["attention"])
att = adata.obs["attention"].values
adata.obs["attention_z"] = (att - att.mean()) / att.std()
adata.obs["attention_z"][adata.obs["attention_z"] < 0] = 0
adata.obs["attention_rank"] = adata.obs["attention"].rank(pct=True)
vmin = np.percentile(adata.obs["attention_z"], 5)
vmax = np.percentile(adata.obs["attention_z"], 95)

import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# Make plots PDF-friendly (vectorized text)
sc.set_figure_params(dpi=100, dpi_save=300, format='pdf', vector_friendly=True)
# Output path
pdf_path = "C:/Users/Inigo/Desktop/FloREN3.0/perez_output/umap_attention_summary.pdf"
with PdfPages(pdf_path) as pdf:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    # ── LEFT: SLE_status ──
    sc.pl.umap(
        adata,
        color="SLE_status",
        ax=axes[0],
        show=False,
        frameon=False,
        title="SLE Status"
    )
    # ── MIDDLE: cell type ──
    sc.pl.umap(
        adata,
        color="cg_cov",
        ax=axes[1],
        show=False,
        frameon=False,
        title="Cell Type"
    )
    # ── RIGHT: attention ──
    sc.pl.umap(
        adata,
        #color="attention",
        #color="attention_log",
        #color="attention_sqrt",
        color="attention_z",
        #color="attention_rank",
        ax=axes[2],
        show=False,
        frameon=False,
        title="Attention",
        cmap="viridis"
    )
    #sc.pl.umap(
    #    adata,
    #    color="attention",
    #    cmap="viridis",
    #    vmin=vmin,
    #    vmax=vmax,
    #    sort_order=True,
    #    ax=axes[2],
    #    show=False,
    #    frameon=False,
    #    title="Attention (clipped)"
    #)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)