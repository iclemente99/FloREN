################################################################
#
# MODULES
#
################################################################

import pandas as pd
import numpy as np
import os
import itertools
import random
import argparse
import copy

import torch

from warnings import filterwarnings
filterwarnings("ignore")

from reduction_leaky import reduction, apply_AE
from utils import debuginfoStr, build_graph
from sub_sample import sub_sample

import scipy
import dcor
from statsmodels.stats.multitest import multipletests
from scipy.sparse import csr_matrix, save_npz

import scanpy as sc

#---------------------------------------------------------------
#
#         SEED AND DEVICE SETUP
#
#---------------------------------------------------------------

seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Arguments
parser = argparse.ArgumentParser(description='Build heterogeneous graph input for FloREN')

# Sampling
parser.add_argument('--n_batch', type=int, default=25, help='Number of batch (sampled graphs) for each epoch')
parser.add_argument('--cell_rate', type=float, default=0.9)
parser.add_argument('--gene_rate', type=float, default=0.3)

# Result
parser.add_argument('--data_name', type=str, default='FloREN', help='The name for dataset')
parser.add_argument('--reduction', type=str, default='AE', help='Feature extraction method: pca, raw, AE')
parser.add_argument('--in_dim', type=int, default=256, help='Number of hidden dimensions (AE bottleneck)')
parser.add_argument('--dc_grn', type=str, default='False',
                    help='If True, use distance correlation for GRN inference (slow); else Pearson')
parser.add_argument('--grn_cutoff', type=float, default=0.7, help='Cutoff for gene-gene connection binarization')
parser.add_argument('--epochs', type=int, default=1000, help='Number of AE training epochs')
parser.add_argument('--cuda', type=int, default=1, help='0 = GPU0, else CPU')
parser.add_argument('--AEtype', type=int, default=1, help='AEtype: 1=embedding AE, 2=HGT node AE')
parser.add_argument('--optimizer', type=str, default='adamw', help='Optimizer')

# Data directories
parser.add_argument('--data_path', default='./data/', type=str, help='Path to data directory')
parser.add_argument('--output_path', default='./floren_output/', type=str,
                    help='Path to folder for graph construction outputs')
parser.add_argument('--cell_comm_path', default=None, type=str,
                    help='Path to folder with cell-cell communication adjacency matrices')
parser.add_argument('--patient_id', default='patient_id', type=str,
                    help='adata.obs column with the patient identifier')
parser.add_argument('--count_layer', default='logcounts', type=str,
                    help='adata.layer name with the log-normalized counts')
parser.add_argument('--adata_path', default='./data/binvignat_example.h5ad', type=str,
                    help='Path to adata object')
parser.add_argument('--save_intermediate_grn', action='store_true', default=False,
                    help='Save intermediate GRN matrices (correlation, bayesian, prior knowledge) to CSV. Large files.')

args = parser.parse_args()
sample_name = args.data_name

output_path = args.output_path
embeddings_path = os.path.join(output_path, "embeddings")
gene_embeddings_path = os.path.join(embeddings_path, "gene_embeddings")
cell_embeddings_path = os.path.join(embeddings_path, "cell_embeddings")
connections_path = os.path.join(output_path, "connections")
os.makedirs(gene_embeddings_path, exist_ok=True)
os.makedirs(cell_embeddings_path, exist_ok=True)
os.makedirs(connections_path, exist_ok=True)

if args.cuda == 0:
    device = torch.device("cuda:0")
    print("Using CUDA GPU0")
else:
    device = torch.device("cpu")
    print("Using CPU")

#---------------------------------------------------------------
#
#                 DATA PROCESSING
#
#---------------------------------------------------------------

print("Loading adata object")
adata = sc.read_h5ad(args.adata_path)

gene_names = adata.var_names
n_genes = len(adata.var_names)
patient_id = args.patient_id
count_layer = args.count_layer

h_n = min(n_genes, args.in_dim)

inds = np.unique(adata.obs[patient_id].values.astype(str))

# D1 — PASS 1: count cells per patient and collect names (no data copies)
cells = []
cell_counts = []
for ind in range(len(inds)):
    adata_subset = adata[adata.obs[patient_id].isin([inds[ind]])]
    n_c = adata_subset.shape[0]
    patient_name = inds[ind]
    cells.append([patient_name + '__' + col for col in adata_subset.obs_names])
    cell_counts.append(n_c)

total_cells = sum(cell_counts)

# D1 — PASS 2: fill pre-allocated float32 array (eliminates O(n_patients) np.concatenate copies)
gene_cell = np.zeros((n_genes, total_cells), dtype=np.float32)
col = 0
for ind in range(len(inds)):
    adata_subset = adata[adata.obs[patient_id].isin([inds[ind]])]
    f_matrix = adata_subset.layers[count_layer].A.astype(np.float32).T  # [n_genes, n_cells]
    n_c = f_matrix.shape[1]
    gene_cell[:, col:col + n_c] = f_matrix
    col += n_c

print(f"    Total genes: {gene_cell.shape[0]}")
print(f"    Total cells: {gene_cell.shape[1]}")

#---------------------------------------------------------------
#
#            AUTOENCODER TRAINING
#
#---------------------------------------------------------------

print("\nRUNNING AE")
epochs = args.epochs
if args.reduction == 'AE':
    model_gene, model_cell, losses, losses2 = reduction(args.reduction, gene_cell, device, h_n, epochs=epochs)
    pd.DataFrame(losses).to_csv(os.path.join(embeddings_path, "All_AE_Genes_loss.csv"))
    pd.DataFrame(losses2).to_csv(os.path.join(embeddings_path, "All_AE_Cells_loss.csv"))
else:
    model_gene = None
    model_cell = None
    print("Modify FloREN to accept your dimensionality reduction method or object")

#---------------------------------------------------------------
#
#            PER-PATIENT EMBEDDINGS
#
#---------------------------------------------------------------

print("Processing samples individually")

all_cells = list(itertools.chain.from_iterable(cells))
pd.DataFrame(all_cells).to_csv(os.path.join(output_path, "All_AUC_Cell_names.csv"))

patient_gene_embeddings = []
patient_cell_embeddings = []
if args.reduction == 'AE':
    cell_encoded = apply_AE(gene_cell, device, h_n, model_cell, cell=True)

    # I1 — Allocate patient_gene_cell once outside the per-patient loop
    patient_gene_cell = np.zeros(gene_cell.shape, dtype=np.float32)

    start_idx = 0
    for i, (patient_cells, n_cells) in enumerate(zip(cells, cell_counts)):
        patient_name = inds[i]
        print(f"    Processing patient: {patient_name}")

        patient_cell_emb = cell_encoded[start_idx:start_idx + n_cells]
        patient_cell_embeddings.append(patient_cell_emb)
        cell_embedding_file = os.path.join(cell_embeddings_path, f"{patient_name}_AE_Emb_Cells.csv")
        pd.DataFrame(patient_cell_emb.cpu().detach().numpy(), index=patient_cells).to_csv(cell_embedding_file)
        start_idx += n_cells

        # I1 — fill patient columns, use, then reset (reuses buffer instead of allocating each iteration)
        col_start = start_idx - n_cells
        col_end   = start_idx
        patient_gene_cell[:, col_start:col_end] = gene_cell[:, col_start:col_end]
        gene_encoded = apply_AE(patient_gene_cell, device, h_n, model_gene, cell=False)
        patient_gene_cell[:, col_start:col_end] = 0.0  # reset for next patient

        patient_gene_embeddings.append(gene_encoded)
        gene_embedding_file = os.path.join(gene_embeddings_path, f"{patient_name}_AE_Emb_Genes.csv")
        pd.DataFrame(gene_encoded.cpu().detach().numpy(), index=gene_names).to_csv(gene_embedding_file)
else:
    start_idx = 0
    for i, (patient_cells, n_cells) in enumerate(zip(cells, cell_counts)):
        patient_name = inds[i]
        print(f"    Processing patient (raw): {patient_name}")

        patient_cell_emb = torch.tensor(gene_cell[:, start_idx:start_idx + n_cells].T, dtype=torch.float32).to(device)
        patient_cell_embeddings.append(patient_cell_emb)
        cell_embedding_file = os.path.join(cell_embeddings_path, f"{patient_name}_Raw_Emb_Cells.csv")
        pd.DataFrame(patient_cell_emb.cpu().detach().numpy(), index=patient_cells).to_csv(cell_embedding_file)

        patient_gene_emb = torch.tensor(gene_cell[:, start_idx:start_idx + n_cells], dtype=torch.float32).to(device)
        patient_gene_embeddings.append(patient_gene_emb)
        gene_embedding_file = os.path.join(gene_embeddings_path, f"{patient_name}_Raw_Emb_Genes.csv")
        pd.DataFrame(patient_gene_emb.cpu().detach().numpy(), index=gene_names).to_csv(gene_embedding_file)

        start_idx += n_cells

debuginfoStr('Feature extraction finished')

#---------------------------------------------------------------
#
#            GLOBAL NORMALIZATION OF ALL EMBEDDINGS
#
#---------------------------------------------------------------

gene_embeddings_norm_path = os.path.join(embeddings_path, "gene_embeddings_norm")
cell_embeddings_norm_path = os.path.join(embeddings_path, "cell_embeddings_norm")
os.makedirs(gene_embeddings_norm_path, exist_ok=True)
os.makedirs(cell_embeddings_norm_path, exist_ok=True)

print("Starting GLOBAL normalization of all patient embeddings...")

all_gene_np = [emb.cpu().detach().numpy() for emb in patient_gene_embeddings]
all_cell_np = [emb.cpu().detach().numpy() for emb in patient_cell_embeddings]

all_genes_concat = np.vstack(all_gene_np)
all_cells_concat = np.vstack(all_cell_np)
all_nodes_concat = np.vstack([all_genes_concat, all_cells_concat])

print(f"Total nodes for normalization: {all_nodes_concat.shape[0]}")


def normalize_embeddings(X):
    return (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)


all_nodes_normalized = normalize_embeddings(all_nodes_concat)

total_genes = sum(g.shape[0] for g in all_gene_np)
normalized_genes = all_nodes_normalized[:total_genes, :]
normalized_cells = all_nodes_normalized[total_genes:, :]

normalized_patient_gene_embeddings = []
normalized_patient_cell_embeddings = []

g_idx = 0
c_idx = 0
for i in range(len(patient_gene_embeddings)):
    patient_name = inds[i]

    g_end = g_idx + patient_gene_embeddings[i].shape[0]
    c_end = c_idx + patient_cell_embeddings[i].shape[0]

    gene_tensor = torch.tensor(normalized_genes[g_idx:g_end], dtype=torch.float32).to(device)
    cell_tensor = torch.tensor(normalized_cells[c_idx:c_end], dtype=torch.float32).to(device)

    normalized_patient_gene_embeddings.append(gene_tensor)
    normalized_patient_cell_embeddings.append(cell_tensor)

    pd.DataFrame(gene_tensor.cpu().numpy(), index=gene_names).to_csv(
        os.path.join(gene_embeddings_norm_path, f"{patient_name}_AE_Emb_Genes.csv"))
    pd.DataFrame(cell_tensor.cpu().numpy(), index=cells[i]).to_csv(
        os.path.join(cell_embeddings_norm_path, f"{patient_name}_AE_Emb_Cells.csv"))

    g_idx = g_end
    c_idx = c_end

print("Global normalization complete.")
debuginfoStr('Global normalization finished')

#---------------------------------------------------------------
#
#            GENE-GENE CONNECTIONS CALCULATION
#
#---------------------------------------------------------------

prior_knowledge_path = os.path.join(args.data_path, "Prior_Knowledge_PRECISEADS.csv")
PK8_PRECIESADS = pd.read_csv(prior_knowledge_path)
PK8_PRECIESADS.set_index('1', inplace=True)

for i in range(len(inds)):
    patient_name = inds[i]
    print(f"    Calculating gene-gene connections for patient: {patient_name}")

    data_corr = patient_gene_embeddings[i].cpu().detach().numpy()

    if args.dc_grn == 'True':
        dcorr_matrix = np.zeros((data_corr.shape[0], data_corr.shape[0]))
        for row in range(data_corr.shape[0]):
            print(f"Calculating dcor for {row}/{data_corr.shape[0]}")
            for column in range(data_corr.shape[0]):
                dcorr_matrix[row, column] = dcor.distance_correlation(data_corr[row, :], data_corr[column, :])
    else:
        # D3-lite — use float32 to halve memory usage
        dcorr_matrix = np.corrcoef(data_corr).astype(np.float32)
        dcorr_matrix = np.absolute(dcorr_matrix)
        np.fill_diagonal(dcorr_matrix, 0)

    if args.save_intermediate_grn:
        pd.DataFrame(dcorr_matrix, index=gene_names, columns=gene_names).to_csv(
            os.path.join(connections_path, f"{patient_name}_distance_correlation.csv"))

    DC8_PRECIESADS = pd.DataFrame(dcorr_matrix, index=gene_names, columns=gene_names)
    inter_names = list(set(DC8_PRECIESADS.columns).intersection(set(PK8_PRECIESADS.columns)))
    PK8_subset = PK8_PRECIESADS[PK8_PRECIESADS.index.isin(inter_names)][inter_names]
    notinter_names = set(inter_names) ^ set(DC8_PRECIESADS.columns)
    empties = pd.DataFrame(np.zeros((len(notinter_names), PK8_subset.shape[0])), columns=PK8_subset.index)
    PK8_subset = pd.concat([PK8_subset, empties])
    empties = pd.DataFrame(np.zeros((PK8_subset.shape[0], len(notinter_names))), index=PK8_subset.index)
    PK8_subset = pd.concat([PK8_subset, empties], axis=1)
    PK8_subset.columns = list(inter_names) + list(notinter_names)
    PK8_subset.index = list(inter_names) + list(notinter_names)
    PK8_subset = PK8_subset.reindex(list(DC8_PRECIESADS.columns))
    PK8_subset = PK8_subset.T.reindex(list(DC8_PRECIESADS.columns))
    PK_PRECIESADS = (0.01 + PK8_subset.values.astype(float)) / (2.01)
    # D3-lite — ensure float32 for DC_PRECIESADS
    DC_PRECIESADS = DC8_PRECIESADS.values.astype(np.float32)
    np.fill_diagonal(DC_PRECIESADS, 0)

    LIKELIHOOD_PRECIESADS = copy.deepcopy(DC_PRECIESADS)
    LIKELIHOOD_PRECIESADS[PK_PRECIESADS < 0.091] = 0
    row_sums = np.sum(LIKELIHOOD_PRECIESADS, axis=1)

    # D2 — Vectorized Bayesian GRN row computation (replaces per-row loop)
    rowsums = row_sums[:, None]  # broadcast [n_genes, 1]
    Edge_lh = (0.01 + DC_PRECIESADS) / (0.01 + DC_PRECIESADS + rowsums)
    np.fill_diagonal(Edge_lh, 0)

    if args.save_intermediate_grn:
        pd.DataFrame(Edge_lh, index=PK8_subset.columns, columns=PK8_subset.columns).to_csv(
            os.path.join(connections_path, f"{patient_name}_bayesian_probability.csv"))
        pd.DataFrame(PK_PRECIESADS, index=PK8_subset.columns, columns=PK8_subset.columns).to_csv(
            os.path.join(connections_path, f"{patient_name}_prior_knowledge.csv"))

    gene_scores = PK_PRECIESADS + (DC_PRECIESADS * Edge_lh)
    gene_scores = gene_scores / np.max(gene_scores)
    gene_scores = np.nan_to_num(gene_scores, nan=0)
    np.fill_diagonal(gene_scores, 0)
    print(f"Patient {patient_name}: Gene-gene connections distribution: "
          f"{np.min(gene_scores):.4f} - {np.max(gene_scores):.4f}, mean {np.mean(gene_scores):.4f}")
    gene_scores[gene_scores <= args.grn_cutoff] = 0
    gene_scores[gene_scores > args.grn_cutoff] = 1
    print(f"Patient {patient_name}: Total gene-gene connections: "
          f"{int(np.sum(gene_scores))} from {gene_scores.shape[0]} genes")

    # Save CSV for backward compatibility
    pd.DataFrame(gene_scores, index=DC8_PRECIESADS.columns, columns=DC8_PRECIESADS.columns).to_csv(
        os.path.join(connections_path, f"{patient_name}_gene_gene_connections.csv"))

    # E1 — Also save as scipy sparse .npz (much smaller for sparse binarized matrices)
    save_npz(
        os.path.join(connections_path, f"{patient_name}_gene_gene_connections.npz"),
        csr_matrix(gene_scores.astype(np.float32))
    )

debuginfoStr('Gene-gene connections calculation finished')
