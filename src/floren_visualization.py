################################################################
#
# MODULES
#
################################################################

import pandas as pd
import numpy as np
import os
import glob
import re
from sklearn.neighbors import kneighbors_graph
import csv

import os
import random
from timeit import default_timer as timer
import argparse
import dill as pickle

import torch
import torch.utils.data as data
from torch import nn, optim
from torch.nn import functional as F

#from reduction import reduction

from reduction_leaky import reduction, apply_AE
from utils import debuginfoStr,loadGAS ,build_data, build_graph
from sub_sample import sub_sample
from pyHGT.model import GNN

from warnings import filterwarnings
filterwarnings("ignore")
from sklearn.model_selection import train_test_split

import scipy
import dcor
from scipy import stats
import statsmodels.stats.multitest
import conorm
from numpy import savetxt
from sklearn.preprocessing import minmax_scale
from statsmodels.stats.multitest import multipletests
import copy
from sklearn.preprocessing import MinMaxScaler
import itertools




#---------------------------------------------------------------
#
#         DEEPMAPS PREPARATION FOR HGT IMPLEMENTATION
#
#---------------------------------------------------------------

# Protection for stochastic procedures
seed = 0
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print('cuda version: ', torch.version.cuda)

# Arguments
parser = argparse.ArgumentParser(description='Training GNN on gene cell graph')

# Sampling times
parser.add_argument('--n_batch', type=int, default=25, help='Number of batch (sampled graphs) for each epoch')
#parser.add_argument('--cell_rate', type=float, default=0.9)
#parser.add_argument('--gene_rate', type=float, default=0.3)

# Result
parser.add_argument('--data_name', type=str, default='FloREN', help='The name for dataset')
#parser.add_argument('--reduction', type=str, default='AE', help='the method for feature extraction, pca, raw, AE')
#parser.add_argument('--in_dim', type=int, default=256, help='Number of hidden dimension (AE)')
#parser.add_argument('--floren_grn', type=str, default='True', help='Decide if to run the floren published gene-gene connections inference')

# GAE
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--n_hid', type=int, default=128, help='Number of hidden dimension')
parser.add_argument('--n_heads', type=int, default=8 ,help='Number of attention head')
#parser.add_argument('--n_layers', type=int, default=2, help='Number of GNN layers')
#parser.add_argument('--dropout', type=float, default=0, help='Dropout ratio')
#parser.add_argument('--lr', type=float, default=0.0005 ,help='learning rate')
#parser.add_argument('--batch_size', type=int, help='Number of output nodes for training')
#parser.add_argument('--layer_type', type=str, default='hgt', help='the layer type for GAE')
#parser.add_argument('--loss', type=str, default='kl', help='the loss for GAE')
#parser.add_argument('--factor', type=float, default='0.5', help='the attenuation factor')
#parser.add_argument('--patience', type=int, default=5, help='patience')
#parser.add_argument('--rf', type=float, default='0.0', help='the weights of regularization')
#parser.add_argument('--cuda', type=int, default=1, help='cuda 0 use GPU0 else cpu')
#parser.add_argument('--rep', type=str, default='T', help='precision truncation')
#parser.add_argument('--AEtype', type=int, default=1, help='AEtype:1 embedding node autoencoder 2:HGT node autoencode')
#parser.add_argument('--optimizer', type=str, default='adamw', help='optimizer')

# Data directories
parser.add_argument('--data_path', default='~/data', type=str, help='Path to folder with graph construction outputs')
parser.add_argument('--output_path', default='~/hgt_input', type=str, help='Path to folder with graph construction outputs')
#parser.add_argument('--cell_comm_path', default=None, type=str, help='Path to folder with cell-cell communication adjacency matrix')
#parser.add_argument('--tfs', default=False, type=str, help='Option to work at tfs level')
parser.add_argument('--result_dir', default=None, type=str, help='Path to folder with FloREN model outputs')
parser.add_argument('--metadata_path', default=None, type=str, help='Path to metadata file')


args = parser.parse_args()
#args.data_path = "C:/Users/Inigo/Desktop/FloREN/Schafflick/genes"
#args.output_path = "C:/Users/Inigo/Desktop/FloREN3.0/schafflick_output"
#args.metadata_path = "C:/Users/Inigo/Desktop/FloREN/Schafflick/samples_metadata.csv"
plots_dir = args.output_path + "/plots/"
os.makedirs(plots_dir, exist_ok=True)
loss_dir = args.output_path + '/loss/'  # Sets directory for loss output
patient_emb_dir = args.output_path + '/floren_patient_embeddings/'

metadata = pd.read_csv(args.metadata_path)
#metadata = pd.read_csv("/Users/Inigo/Desktop/FloREN/Perez/samples_metadata_unique.csv")
#metadata.columns = ["Unnamed: 0", "0", "patient_id", "group", "Age", "Sex", "Ethnicity", "Pool", "Batch", "Age_Group"]
#metadata.columns = ["patient_id", "patient_num", "patient_ID", "group", "Age", "Sex", "Tissue"]
metadata["group_txt"] = metadata["group"]
metadata["group"] = metadata["group"].astype("category").cat.codes
#metadata["patient_id"] = "schafflick_" + metadata["patient_id"].astype(str)

data_path = args.data_path
#data_path = "C:/Users/Inigo/Desktop/FloREN/Perez/tfs"
#count_matrices_path = os.path.join(data_path, "count_matrices")
#count_matrices_path = "/home/inigo/Desktop/FloREN3.0/Binvignat/genes/"
print(f"Looking for CSV files in: {data_path}")
csv_files = glob.glob(os.path.join(data_path, "*.csv"))
print(f"Found {len(csv_files)} CSV files: {csv_files}")

# Load reference for gene names and concatenate all matrices
reference = pd.read_csv(csv_files[0])
gene_names = reference[reference.columns[0]].values
n_genes = reference.shape[0]

#if n_genes < args.in_dim:
#    h_n = n_genes
#else:
#    h_n = args.in_dim

#---------------------------------------------------------------
#
#                    VISUALIZATION
#
#---------------------------------------------------------------

import matplotlib.pyplot as plt
import umap
import smtplib
from email.message import EmailMessage
import mimetypes
import ssl

# === TRAINING LOSS PLOT ===
#plt.figure(figsize=(10, 6))
#plt.plot(contrastive_losses, label="Contrastive Loss (Self-Supervised)", color='blue')
#plt.plot(classification_losses, label="Classification Loss (Supervised)", color='green')
#plt.plot(validation_losses, label="Validation Loss", color='orange')
#plt.plot(total_losses, label="Total Weighted Loss", color='black', linestyle="--")
#plt.xlabel("Epochs")
#plt.ylabel("Loss")
#plt.title("Loss Evolution Over Training")
#plt.legend()
#plt.grid(True)

# Load loss history
model0 = f'{args.data_name}_epoch_{args.epoch}_n_hid_{args.n_hid}_nheads_{args.n_heads}_lr_01_n_batch{args.n_batch}'
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
#plt.grid(True)
plt.tight_layout()
#plt.show()

image_path1 = os.path.join(plots_dir, "FloREN_training.pdf")
plt.savefig(image_path1, format="pdf", bbox_inches="tight", dpi=300)
plt.close()

# === UMAP SAMPLES PLOT ===
# vector_path = "/Users/Inigo/Desktop/FloREN3.0/perez_output/floren_patient_embeddings/"  # Path to saved patient embeddings
# vector_files = [f for f in os.listdir(vector_path) if f.endswith(".csv")]
# vector_files = [f for f in vector_files if "n_hid_128_nheads_8" in f]
# all_vectors = []
# labels = []
# for file in vector_files:
#    df = pd.read_csv(os.path.join(vector_path, file), header=None)  # No index column
#    all_vectors.append(df.values.squeeze())  # Shape: [552]
#    label = "RA" if "RA" in file else "CNT"
#    labels.append(label)

vector_path = patient_emb_dir
vector_files = [f for f in os.listdir(vector_path) if f.endswith(".csv")]
vector_files = [f for f in vector_files if f"n_hid_{args.n_hid}_nheads_{args.n_heads}" in f]
all_vectors = []
labels = []
for file in vector_files:
    # Extract patient ID from filename: sample_IGTB645_IGTB645_epoch_100...
    m = re.search(r"sample_(.*?)_epoch", file)
    if not m:
        print(f"Could not parse patient ID from file: {file}")
        continue
    #
    patient_id = m.group(1)  # e.g. "IGTB645_IGTB645"
    # Look up metadata row
    row = metadata[metadata["patient_id"] == patient_id]
    if row.empty:
        print(f"Patient {patient_id} not found in metadata! Skipping...")
        continue
    #
    # Group: 1 → RA, 0 → CNT
    group = row["group_txt"].iloc[0]
    # label = "SLE" if group == 1 else "CNT"
    #if group == 0:
    #    label = "CNT"
    #else:
    #    # group == 1 → SLE-related
    #    if patient_id.startswith("FLARE"):
    #        label = "FLARE"
    #    else:
    #        label = "SLE"
    #
    label = group
    labels.append(label)
    # Load vector
    df = pd.read_csv(os.path.join(vector_path, file), header=None)
    all_vectors.append(df.values.squeeze())

all_vectors = np.array(all_vectors)  # Shape: [num_patients, 552]
num_groups = 2  # Control vs. RA
# Split all_vectors into components
emb_104 = all_vectors[:, :128]  # [36, 104]
g_32 = all_vectors[:, 128:160]  # [36, 32]
l_32 = all_vectors[:, 160:192]  # [36, 32]
g_64 = all_vectors[:, 192:256]  # [36, 64]
l_64 = all_vectors[:, 256:320]  # [36, 64]
g_128 = all_vectors[:, 320:448]  # [36, 128]
l_128 = all_vectors[:, 448:576]  # [36, 128]
ret_class = all_vectors[:, 576:]  # [36, 2]
softmax_ret_class = np.exp(ret_class) / np.sum(np.exp(ret_class), axis=1, keepdims=True)  # [36, 2]

# Create UMAP plots in a 5x2 grid
pdf_path = os.path.join(plots_dir, "FloREN_umap.pdf")

import matplotlib.cm as cm

try:
    # Create figure and axes
    fig, axes = plt.subplots(5, 2, figsize=(12, 24), constrained_layout=True)
    axes = axes.flatten()
    #
    # Define plot order
    plot_order = [
        ("emb_104", emb_104),
        ("l_128", l_128),
        ("g_128", g_128),
        ("l_64", l_64),
        ("g_64", g_64),
        ("l_32", l_32),
        ("g_32", g_32),
        ("ret_class", ret_class),
        ("softmax(ret_class)", softmax_ret_class)
    ]
    #
    # Initialize UMAP
    umap_model = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric="euclidean",
        random_state=42
    )
    #
    #colors = {"MS": "red", "PT": "blue"}
    labels_np = np.array(labels)
    #unique_groups = np.unique(labels_np)
    #cmap = cm.get_cmap("tab10", len(unique_groups))
    #colors = {
    #    group: cmap(i)
    #    for i, group in enumerate(unique_groups)
    #}
    #
    colors = {"SLE": "red", "HD": "blue"}
    # Generate plots
    for idx, (title, vectors) in enumerate(plot_order):
        if idx >= len(axes):  # Prevent overflow if plot_order < 10
            break
        #
        ax = axes[idx]
        try:
            # If more than 2 features → apply UMAP
            if vectors.shape[1] > 2:
                embedding = umap_model.fit_transform(vectors)
                ax.set_title(f"{title} (UMAP)")
            else:
                embedding = vectors
                ax.set_title(title)
            #
            # Plot
            for label in np.unique(labels_np):
                indices = np.where(labels_np == label)[0]
                ax.scatter(
                    embedding[indices, 0],
                    embedding[indices, 1],
                    label=label,
                    color=colors[label],
                    alpha=0.7
                )
            #
            ax.set_xlabel("Dimension 1")
            ax.set_ylabel("Dimension 2")
            ax.legend()
            ax.grid(True)
            #
        except Exception as e:
            print(f"Error plotting {title}: {e}")
            ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
        #
    # Remove unused axes
    for ax in axes:
        if not ax.has_data():
            ax.remove()
    #
    # Save as PDF
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=300)
    print(f"Saved UMAP plots to {pdf_path}")
    #
except Exception as e:
    print(f"Error saving PDF: {e}")
finally:
    plt.close('all')  # Clean up

from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

# === MODEL EVALUATION ===
print("\n=== MODEL EVALUATION METRICS ===")

# Convert labels (RA=1, CNT=0)
y_true = np.array([1 if l == metadata.group_txt.unique()[1] else 0 for l in labels])
#y_true = metadata["group"].to_numpy()
y_pred_probs = softmax_ret_class[:, 1]  # Probability of RA
y_pred = (y_pred_probs > 0.5).astype(int)  # Classify with 0.5 threshold

# Compute metrics
f1_macro = f1_score(y_true, y_pred, average="macro")
f1_micro = f1_score(y_true, y_pred, average="micro")
f1_weighted = f1_score(y_true, y_pred, average="weighted")

try:
    auc_score = roc_auc_score(y_true, y_pred_probs)
except ValueError:
    auc_score = np.nan  # Handles the case if only one class is predicted

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[metadata.group_txt.unique()[0], metadata.group_txt.unique()[1]])

# Print metrics
print(f"F1 (macro):     {f1_macro:.4f}")
print(f"F1 (micro):     {f1_micro:.4f}")
print(f"F1 (weighted):  {f1_weighted:.4f}")
print(f"ROC-AUC:        {auc_score:.4f}")
print(f"Confusion matrix:\n{cm}")

# Plot confusion matrix
plt.figure(figsize=(5, 4))
disp.plot(cmap='Blues', values_format='d')
plt.title(f"Confusion Matrix: {metadata.group_txt.unique()[0]} vs {metadata.group_txt.unique()[1]}")
plt.grid(False)
plt.tight_layout()
confmat_path = os.path.join(plots_dir, "FloREN_confusion_matrix.pdf")
plt.savefig(confmat_path, format="pdf", bbox_inches="tight", dpi=300)
plt.close()
print(f"Saved confusion matrix to {confmat_path}")
