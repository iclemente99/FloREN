################################################################
#
# MODULES
#
################################################################

import pandas as pd
import numpy as np
import os
import re
import argparse
import random

import torch

from warnings import filterwarnings
filterwarnings("ignore")

import scanpy as sc
import matplotlib.pyplot as plt
import umap
import matplotlib.cm as cm
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

from reduction_leaky import reduction, apply_AE
from utils import debuginfoStr, loadGAS, build_data, build_graph
from sub_sample import sub_sample
from pyHGT.model import GNN

#---------------------------------------------------------------
#
#         SEED AND DEVICE SETUP
#
#---------------------------------------------------------------

seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

# Arguments
parser = argparse.ArgumentParser(description='Visualization of FloREN results')

parser.add_argument('--n_batch', type=int, default=25, help='Number of batch (sampled graphs) for each epoch')

# Result
parser.add_argument('--data_name', type=str, default='FloREN', help='The name for dataset')
parser.add_argument('--n_hid', type=int, default=128, help='Number of hidden dimension')
parser.add_argument('--n_heads', type=int, default=8, help='Number of attention head')

# Must match the --epochs value used in floren_training.py
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs (must match training run)')

# Data directories
parser.add_argument('--data_path', default='./data/', type=str, help='Path to data directory')
parser.add_argument('--output_path', default='./floren_output/', type=str, help='Path to folder with FloREN outputs')
parser.add_argument('--result_dir', default=None, type=str, help='Path to folder with FloREN model outputs (defaults to output_path)')
parser.add_argument('--adata_path', default='./data/binvignat_example.h5ad', type=str, help='Path to adata object')
parser.add_argument('--patient_id', default='patient_id', type=str,
                    help='adata.obs column with the patient identifier')
parser.add_argument('--metadata_group', default='disease', type=str,
                    help='adata.obs column with the metadata group to differentiate')

args = parser.parse_args()

if args.result_dir is None:
    args.result_dir = args.output_path

plots_dir = args.result_dir + "/plots/"
os.makedirs(plots_dir, exist_ok=True)
loss_dir = args.result_dir + '/loss/'
patient_emb_dir = args.result_dir + '/floren_patient_embeddings/'

metadata_group = args.metadata_group
patient_id_col = args.patient_id  # column name — kept separate to avoid shadowing
data_path = args.data_path

adata = sc.read_h5ad(args.adata_path)
inds = np.unique(adata.obs[patient_id_col].values.astype(str))
metadata = adata.obs[[patient_id_col, metadata_group]]
metadata = metadata.groupby(patient_id_col).first().reset_index()
metadata["group_txt"] = metadata[metadata_group]
metadata["group"] = metadata[metadata_group].astype("category").cat.codes

gene_names = adata.var_names
n_genes = len(adata.var_names)

#---------------------------------------------------------------
#
#                    VISUALIZATION
#
#---------------------------------------------------------------

# === TRAINING LOSS PLOT ===
model0 = f'{args.data_name}_epoch_{args.epochs}_n_hid_{args.n_hid}_nheads_{args.n_heads}_lr_01_n_batch{args.n_batch}'
loss_csv = os.path.join(loss_dir, model0 + ".csv")
loss_history = pd.read_csv(loss_csv)
plt.figure(figsize=(10, 6))
plt.plot(
    loss_history["epoch"],
    loss_history["train_contrastive"],
    label="Contrastive Loss (Self-Supervised)",
    color="blue"
)
plt.plot(
    loss_history["epoch"],
    loss_history["train_classification"],
    label="Classification Loss (Supervised)",
    color="green"
)
plt.plot(
    loss_history["epoch"],
    loss_history["val_total"],
    label="Validation Loss",
    color="orange"
)
plt.plot(
    loss_history["epoch"],
    loss_history["train_total"],
    label="Total Weighted Loss",
    color="black",
    linestyle="--"
)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Evolution Over Training")
plt.legend()
plt.tight_layout()

image_path1 = os.path.join(plots_dir, "FloREN_training.pdf")
plt.savefig(image_path1, format="pdf", bbox_inches="tight", dpi=300)
plt.close()

# === UMAP SAMPLES PLOT ===
vector_path = patient_emb_dir
vector_files = [f for f in os.listdir(vector_path) if f.endswith(".csv")]
vector_files = [f for f in vector_files if f"n_hid_{args.n_hid}_nheads_{args.n_heads}" in f]
all_vectors = []
labels = []
for file in vector_files:
    # Extract patient name from filename: sample_<patient>_epoch_...
    m = re.search(r"sample_(.*?)_epoch", file)
    if not m:
        print(f"Could not parse patient ID from file: {file}")
        continue

    pid = m.group(1)
    row = metadata[metadata[patient_id_col] == pid]
    if row.empty:
        print(f"Patient {pid} not found in metadata, skipping...")
        continue

    label = row["group_txt"].iloc[0]
    labels.append(label)
    df = pd.read_csv(os.path.join(vector_path, file), header=None)
    all_vectors.append(df.values.squeeze())

all_vectors = np.array(all_vectors)
n_hid = args.n_hid

# Split all_vectors into embedding components.
# Layout: [emb_n_hid | g_32 | l_32 | g_64 | l_64 | g_128 | l_128 | ret_class]
emb_start = 0
emb_end   = n_hid
emb_nhid  = all_vectors[:, emb_start:emb_end]
g_32      = all_vectors[:, emb_end:emb_end+32]
l_32      = all_vectors[:, emb_end+32:emb_end+64]
g_64      = all_vectors[:, emb_end+64:emb_end+128]
l_64      = all_vectors[:, emb_end+128:emb_end+192]
g_128     = all_vectors[:, emb_end+192:emb_end+320]
l_128     = all_vectors[:, emb_end+320:emb_end+448]
ret_class = all_vectors[:, emb_end+448:]
softmax_ret_class = np.exp(ret_class) / np.sum(np.exp(ret_class), axis=1, keepdims=True)

# Create UMAP plots in a 5x2 grid
pdf_path = os.path.join(plots_dir, "FloREN_umap.pdf")

try:
    fig, axes = plt.subplots(5, 2, figsize=(12, 24), constrained_layout=True)
    axes = axes.flatten()

    plot_order = [
        (f"emb_{n_hid}", emb_nhid),
        ("l_128", l_128),
        ("g_128", g_128),
        ("l_64", l_64),
        ("g_64", g_64),
        ("l_32", l_32),
        ("g_32", g_32),
        ("ret_class", ret_class),
        ("softmax(ret_class)", softmax_ret_class)
    ]

    umap_model = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric="euclidean",
        random_state=42
    )

    labels_np = np.array(labels)
    unique_groups = np.unique(labels_np)
    cmap = cm.get_cmap("tab10", len(unique_groups))
    colors = {group: cmap(i) for i, group in enumerate(unique_groups)}

    for idx, (title, vectors) in enumerate(plot_order):
        if idx >= len(axes):
            break
        ax = axes[idx]
        try:
            if vectors.shape[1] > 2:
                embedding = umap_model.fit_transform(vectors)
                ax.set_title(f"{title} (UMAP)")
            else:
                embedding = vectors
                ax.set_title(title)
            for label in np.unique(labels_np):
                indices = np.where(labels_np == label)[0]
                ax.scatter(
                    embedding[indices, 0],
                    embedding[indices, 1],
                    label=label,
                    color=colors[label],
                    alpha=0.7
                )
            ax.set_xlabel("Dimension 1")
            ax.set_ylabel("Dimension 2")
            ax.legend()
            ax.grid(True)
        except Exception as e:
            print(f"Error plotting {title}: {e}")
            ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')

    for ax in axes:
        if not ax.has_data():
            ax.remove()

    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=300)
    print(f"Saved UMAP plots to {pdf_path}")

except Exception as e:
    print(f"Error saving PDF: {e}")
finally:
    plt.close('all')

# === MODEL EVALUATION ===
print("\n=== MODEL EVALUATION METRICS ===")

y_true = np.array([1 if l == metadata.group_txt.unique()[1] else 0 for l in labels])
y_pred_probs = softmax_ret_class[:, 1]
y_pred = (y_pred_probs > 0.5).astype(int)

f1_macro    = f1_score(y_true, y_pred, average="macro")
f1_micro    = f1_score(y_true, y_pred, average="micro")
f1_weighted = f1_score(y_true, y_pred, average="weighted")

try:
    auc_score = roc_auc_score(y_true, y_pred_probs)
except ValueError:
    auc_score = np.nan

cm_mat = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_mat,
    display_labels=[metadata.group_txt.unique()[0], metadata.group_txt.unique()[1]]
)

print(f"F1 (macro):     {f1_macro:.4f}")
print(f"F1 (micro):     {f1_micro:.4f}")
print(f"F1 (weighted):  {f1_weighted:.4f}")
print(f"ROC-AUC:        {auc_score:.4f}")
print(f"Confusion matrix:\n{cm_mat}")

plt.figure(figsize=(5, 4))
disp.plot(cmap='Blues', values_format='d')
plt.title(f"Confusion Matrix: {metadata.group_txt.unique()[0]} vs {metadata.group_txt.unique()[1]}")
plt.grid(False)
plt.tight_layout()
confmat_path = os.path.join(plots_dir, "FloREN_confusion_matrix.pdf")
plt.savefig(confmat_path, format="pdf", bbox_inches="tight", dpi=300)
plt.close()
print(f"Saved confusion matrix to {confmat_path}")
