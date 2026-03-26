
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

# from reduction import reduction

from reduction_leaky import reduction, apply_AE
from utils import debuginfoStr, loadGAS, build_data, build_graph
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

import scanpy as sc

# ---------------------------------------------------------------
#
#         DEEPMAPS PREPARATION FOR HGT IMPLEMENTATION
#
# ---------------------------------------------------------------

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
parser.add_argument('--cell_rate', type=float, default=0.9)
parser.add_argument('--gene_rate', type=float, default=0.3)

# Result
parser.add_argument('--data_name', type=str, default='FloREN', help='The name for dataset')
parser.add_argument('--reduction', type=str, default='AE', help='the method for feature extraction, pca, raw, AE')
parser.add_argument('--in_dim', type=int, default=256, help='Number of hidden dimension (AE)')
parser.add_argument('--dc_grn', type=str, default='False',
                    help='Decide if to run the floren gene-gene connections inference on distance correlation')
#parser.add_argument('--tfs', default=False, type=str, help='Option to work at tfs level')
parser.add_argument('--grn_cutoff', type=float, default=0.7, help='Cutoff for gene-gene connections inference')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs (AE)')

# GAE
#parser.add_argument('--epoch', type=int, default=100)
#parser.add_argument('--n_hid', type=int, help='Number of hidden dimension', default=128)
#parser.add_argument('--n_heads', type=int, help='Number of attention head', default=8)
#parser.add_argument('--n_layers', type=int, default=2, help='Number of GNN layers')
#parser.add_argument('--dropout', type=float, default=0, help='Dropout ratio')
#parser.add_argument('--lr', type=float, help='learning rate', default=0.005)
#parser.add_argument('--batch_size', type=int, help='Number of output nodes for training')
#parser.add_argument('--layer_type', type=str, default='hgt', help='the layer type for GAE')
#parser.add_argument('--loss', type=str, default='kl', help='the loss for GAE')
#parser.add_argument('--factor', type=float, default='0.5', help='the attenuation factor')
#parser.add_argument('--patience', type=int, default=5, help='patience')
#parser.add_argument('--rf', type=float, default='0.0', help='the weights of regularization')
parser.add_argument('--cuda', type=int, default=1, help='cuda 0 use GPU0 else cpu')
#parser.add_argument('--rep', type=str, default='T', help='precision truncation')
parser.add_argument('--AEtype', type=int, default=1, help='AEtype:1 embedding node autoencoder 2:HGT node autoencode')
parser.add_argument('--optimizer', type=str, default='adamw', help='optimizer')

# Data directories
parser.add_argument('--data_path', default='~/data', type=str, help='Path to adata object')
parser.add_argument('--output_path', default='~/hgt_input', type=str,
                    help='Path to folder with graph construction outputs')
parser.add_argument('--cell_comm_path', default=None, type=str,
                    help='Path to folder with cell-cell communication adjacency matrix')

args = parser.parse_args()
#args.epoch = 100
#args.n_hid = 128
#args.n_heads = 13
#args.lr = 0.1
#args.n_batch = 32
sample_name = args.data_name

# Set up result directories
# output_path = os.path.abspath(os.path.expanduser(args.output_path))
#output_path = "C:/Users/Inigo/Desktop/FloREN3.0/perez_output"
output_path = args.output_path
embeddings_path = os.path.join(output_path, "embeddings")
gene_embeddings_path = os.path.join(embeddings_path, "gene_embeddings")
cell_embeddings_path = os.path.join(embeddings_path, "cell_embeddings")
connections_path = os.path.join(output_path, "connections")
os.makedirs(gene_embeddings_path, exist_ok=True)
os.makedirs(cell_embeddings_path, exist_ok=True)
os.makedirs(connections_path, exist_ok=True)

#args.cuda = 1
# Set device
if args.cuda == 0:
    device = torch.device("cuda:0")
    print("cuda>>>")
else:
    device = torch.device("cpu")

print(device)

# ---------------------------------------------------------------
#
#                 DATA PROCESSING BEFORE HGT
#
# ---------------------------------------------------------------

data_path = args.data_path
#data_path = "C:/Users/Inigo/Desktop/FloREN/Perez/tfs"
# count_matrices_path = os.path.join(data_path, "count_matrices")
# count_matrices_path = "/home/inigo/Desktop/FloREN3.0/Binvignat/genes/"
#print(f"Looking for CSV files in: {data_path}")
#csv_files = glob.glob(os.path.join(data_path, "*.csv"))
#print(f"Found {len(csv_files)} CSV files: {csv_files}")
adata = sc.read_h5ad(data_path)

# Load reference for gene names and concatenate all matrices
#reference = pd.read_csv(csv_files[0])
#print(f"Attention! Your input files have shape: {reference.shape}")
#gene_names = reference[reference.columns[0]].values
#n_genes = reference.shape[0]
gene_names = adata.var_names
n_genes = len(adata.var_names)

if n_genes < args.in_dim:
    h_n = n_genes
else:
    h_n = args.in_dim

inds = np.unique(adata.obs["patient_id"].values.astype(str))
gene_cell = np.zeros((n_genes, 1))
cells = []
cell_counts = []
for f in csv_files:
    #file = pd.read_csv(f)
    #patient_name = str.split(str.split(f, 'genes')[1], '.csv')[0]
    #patient_name = str.split(f, '.csv')[0]
    #patient_name = str.split(str.split(f, str.split(data_path, "/")[-1] + "\\")[1], '.csv')[0]
    #cells.append([patient_name + '__' + col for col in file.columns[1:]])
    #cell_counts.append(file.shape[1] - 1)  # Number of cells for this patient
    #file = file.iloc[:, 1:].to_numpy()
    #gene_cell = np.concatenate((gene_cell, file), axis=1)
    adata_subset = adata[adata.obs['donor_id'].isin([inds[ind]])]
    f_matrix = adata_subset.layers['logcounts'].A.T
    file = pd.DataFrame(f_matrix, columns=adata_subset.obs_names, index=adata_subset.var_names)
    patient_name = [inds[ind]]
    cells.append([patient_name[0] + '__' + col for col in file.columns])
    cell_counts.append(file.shape[1])
    file = file.values
    gene_cell = np.concatenate((gene_cell, file), axis=1)

gene_cell = gene_cell[:, 1:]  # Shape: [n_genes, total_cells]

epochs = args.epochs
# Train autoencoders on merged matrix
if args.reduction == 'AE':
    model_gene, model_cell, losses, losses2 = reduction(args.reduction, gene_cell, device, h_n, epochs=epochs)
    pd.DataFrame(losses).to_csv(os.path.join(embeddings_path, "All_AE_Genes_loss.csv"))
    pd.DataFrame(losses2).to_csv(os.path.join(embeddings_path, "All_AE_Cells_loss.csv"))
else:
    model_gene = None
    model_cell = None
    print("Modify FloREN to accept your dimensionality reduction method or object")

# ---------------------------------------------------------------
#
#            PER-PATIENT EMBEDDINGS
#
# ---------------------------------------------------------------

import itertools

all_cells = list(itertools.chain.from_iterable(cells))
pd.DataFrame(all_cells).to_csv(os.path.join(output_path, "All_AUC_Cell_names.csv"))

# Split cell embeddings and generate patient-specific gene embeddings
patient_gene_embeddings = []
patient_cell_embeddings = []
if args.reduction == 'AE':
    # Get cell embeddings from merged matrix
    cell_encoded = apply_AE(gene_cell, device, h_n, model_cell, cell=True)  # Shape: [total_cells, h_n]
    #
    # Split cell embeddings by patient
    start_idx = 0
    for i, (patient_cells, n_cells) in enumerate(zip(cells, cell_counts)):
        #patient_name = str.split(str.split(csv_files[i], 'perez_')[1], '.csv')[0]
        #patient_name = str.split(str.split(csv_files[i], str.split(data_path, "/")[-1]+"\\")[1], '.csv')[0]
        #patient_name = str.split(csv_files[i], '.csv')[0]
        patient_name = inds[i]
        print(f"Processing patient: {patient_name}")
        #
        # Extract cell embeddings
        patient_cell_emb = cell_encoded[start_idx:start_idx + n_cells]
        patient_cell_embeddings.append(patient_cell_emb)
        cell_embedding_file = os.path.join(cell_embeddings_path, f"{patient_name}_AE_Emb_Cells.csv")
        pd.DataFrame(patient_cell_emb.cpu().detach().numpy(), index=patient_cells).to_csv(cell_embedding_file)
        start_idx += n_cells
        #
        # Create patient-specific gene matrix (zero out other patients' cells)
        patient_gene_cell = np.zeros_like(gene_cell)  # Shape: [n_genes, total_cells]
        patient_gene_cell[:, start_idx - n_cells:start_idx] = gene_cell[:, start_idx - n_cells:start_idx]
        #
        # Get gene embeddings
        gene_encoded = apply_AE(patient_gene_cell, device, h_n, model_gene, cell=False)  # Shape: [n_genes, h_n]
        patient_gene_embeddings.append(gene_encoded)
        gene_embedding_file = os.path.join(gene_embeddings_path, f"{patient_name}_AE_Emb_Genes.csv")
        pd.DataFrame(gene_encoded.cpu().detach().numpy(), index=gene_names).to_csv(gene_embedding_file)
else:
    # Raw method
    start_idx = 0
    for i, (patient_cells, n_cells) in enumerate(zip(cells, cell_counts)):
        #patient_name = str.split(str.split(csv_files[i], 'perez_')[1], '.csv')[0]
        #patient_name = str.split(str.split(csv_files[i], str.split(data_path, "/")[-1]+"\\")[1], '.csv')[0]
        patient_name = inds[i]
        print(f"Processing patient (raw): {patient_name}")
        #
        # Extract raw cell data
        patient_cell_emb = torch.tensor(gene_cell[:, start_idx:start_idx + n_cells].T, dtype=torch.float32).to(device)
        patient_cell_embeddings.append(patient_cell_emb)
        cell_embedding_file = os.path.join(cell_embeddings_path, f"{patient_name}_Raw_Emb_Cells.csv")
        pd.DataFrame(patient_cell_emb.cpu().detach().numpy(), index=patient_cells).to_csv(cell_embedding_file)
        #
        # Use raw gene data for this patient
        patient_gene_emb = torch.tensor(gene_cell[:, start_idx:start_idx + n_cells], dtype=torch.float32).to(device)
        patient_gene_embeddings.append(patient_gene_emb)
        gene_embedding_file = os.path.join(gene_embeddings_path, f"{patient_name}_Raw_Emb_Genes.csv")
        pd.DataFrame(patient_gene_emb.cpu().detach().numpy(), index=gene_names).to_csv(gene_embedding_file)
        #
        start_idx += n_cells


debuginfoStr('Feature extraction finished')

# ---------------------------------------------------------------
#
#            GLOBAL NORMALIZATION OF ALL EMBEDDINGS
#
# ---------------------------------------------------------------

gene_embeddings_norm_path = os.path.join(embeddings_path, "gene_embeddings_norm")
cell_embeddings_norm_path = os.path.join(embeddings_path, "cell_embeddings_norm")
os.makedirs(gene_embeddings_norm_path, exist_ok=True)
os.makedirs(cell_embeddings_norm_path, exist_ok=True)

print("Starting GLOBAL normalization of all patient embeddings...")

# Step 1: Convert to numpy and collect
all_gene_np = [emb.cpu().detach().numpy() for emb in patient_gene_embeddings]
all_cell_np = [emb.cpu().detach().numpy() for emb in patient_cell_embeddings]

# Step 2: Concatenate all
all_genes_concat = np.vstack(all_gene_np)  # [total_genes, h_n]
all_cells_concat = np.vstack(all_cell_np)  # [total_cells, h_n]
all_nodes_concat = np.vstack([all_genes_concat, all_cells_concat])  # [total_nodes, h_n]

print(f"Total nodes for normalization: {all_nodes_concat.shape[0]}")


def normalize_embeddings(X):
    # Normalize each row (embedding vector) to zero mean, unit variance
    return (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)


# Step 3: Global min-max normalization (per feature)
# all_nodes_normalized = minmax_scale(all_nodes_concat, axis=0)  # [0,1] per column
all_nodes_normalized = normalize_embeddings(all_nodes_concat)

# Step 4: Split back
total_genes = sum(g.shape[0] for g in all_gene_np)
normalized_genes = all_nodes_normalized[:total_genes, :]
normalized_cells = all_nodes_normalized[total_genes:, :]

# Step 5: Re-split into per-patient tensors
normalized_patient_gene_embeddings = []
normalized_patient_cell_embeddings = []

g_idx = 0
c_idx = 0
for i in range(len(patient_gene_embeddings)):
    #patient_name = str.split(str.split(csv_files[i], 'perez_')[1], '.csv')[0]
    #patient_name = str.split(str.split(csv_files[i], str.split(data_path, "/")[-1]+"\\")[1], '.csv')[0]
    patient_name = inds[i]
    #
    g_end = g_idx + patient_gene_embeddings[i].shape[0]
    c_end = c_idx + patient_cell_embeddings[i].shape[0]
    #
    gene_tensor = torch.tensor(normalized_genes[g_idx:g_end], dtype=torch.float32).to(device)
    cell_tensor = torch.tensor(normalized_cells[c_idx:c_end], dtype=torch.float32).to(device)
    #
    normalized_patient_gene_embeddings.append(gene_tensor)
    normalized_patient_cell_embeddings.append(cell_tensor)
    #
    print(patient_name)
    pd.DataFrame(gene_tensor.cpu().numpy(), index=gene_names).to_csv(
        os.path.join(gene_embeddings_norm_path, f"{patient_name}_AE_Emb_Genes.csv"))
    pd.DataFrame(cell_tensor.cpu().numpy(), index=cells[i]).to_csv(
        os.path.join(cell_embeddings_norm_path, f"{patient_name}_AE_Emb_Cells.csv"))
    #
    g_idx = g_end
    c_idx = c_end

print("Global normalization complete. All patients now in shared embedding space.")
debuginfoStr('Global normalization finished')

# ---------------------------------------------------------------
#
#            GENE-GENE CONNECTIONS CALCULATION
#
# ---------------------------------------------------------------

# Load prior knowledge
PK8_PRECIESADS = pd.read_csv(os.path.join(data_path, "Prior_Knowledge_PRECISEADS.csv"))
#PK8_PRECIESADS = pd.read_csv("C:/Users/Inigo/Documents/Prior_Knowledge_PRECISEADS.csv")
PK8_PRECIESADS.set_index('1', inplace=True)

#args.dc_grn = 'False'
# Process gene-gene connections for each patient
for i, f in enumerate(csv_files):
    #patient_name = str.split(str.split(f, 'perez_')[1], '.csv')[0]
    #patient_name = str.split(str.split(f, str.split(data_path, "/")[-1] + "\\")[1], '.csv')[0]
    patient_name = inds[i]
    print(f"Calculating gene-gene connections for patient: {patient_name}")
    #
    # Load gene embeddings
    data_corr = patient_gene_embeddings[i].cpu().detach().numpy()
    #
    # Calculate distance correlation or Pearson correlation
    if args.dc_grn == 'True':
        dcorr_matrix = np.zeros((data_corr.shape[0], data_corr.shape[0]))
        for row in range(data_corr.shape[0]):
            print(f"Calculating dcor for {row}/{data_corr.shape[0]}")
            for column in range(data_corr.shape[0]):
                dcorr_matrix[row, column] = dcor.distance_correlation(data_corr[row, :], data_corr[column, :])
    else:
        dcorr_matrix = np.corrcoef(data_corr)
        dcorr_matrix = np.absolute(dcorr_matrix)
        np.fill_diagonal(dcorr_matrix, 0)
    #
    # Save distance correlation matrix
    pd.DataFrame(dcorr_matrix, index=gene_names, columns=gene_names).to_csv(
        os.path.join(connections_path, f"{patient_name}_distance_correlation.csv"))
    #
    # Process prior knowledge for this patient
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
    DC_PRECIESADS = DC8_PRECIESADS.values
    np.fill_diagonal(DC_PRECIESADS, 0)
    #
    # Compute likelihood
    LIKELIHOOD_PRECIESADS = copy.deepcopy(DC_PRECIESADS)
    LIKELIHOOD_PRECIESADS[PK_PRECIESADS < 0.091] = 0
    row_sums = np.sum(LIKELIHOOD_PRECIESADS, axis=1)
    Edge_lh = np.zeros_like(DC_PRECIESADS)
    for row in range(len(Edge_lh)):
        rowsum = row_sums[row]
        data = DC_PRECIESADS[row, :]
        likelihood = (0.01 + data) / (0.01 + data + rowsum)
        Edge_lh[row, :] = likelihood
    np.fill_diagonal(Edge_lh, 0)
    #
    # Save Bayesian probability and prior knowledge
    pd.DataFrame(Edge_lh, index=PK8_subset.columns, columns=PK8_subset.columns).to_csv(
        os.path.join(connections_path, f"{patient_name}_bayesian_probability.csv"))
    pd.DataFrame(PK_PRECIESADS, index=PK8_subset.columns, columns=PK8_subset.columns).to_csv(
        os.path.join(connections_path, f"{patient_name}_prior_knowledge.csv"))
    #
    # Compute gene-gene connection scores
    gene_scores = PK_PRECIESADS + (DC_PRECIESADS * Edge_lh)
    gene_scores = gene_scores / np.max(gene_scores)
    gene_scores = np.nan_to_num(gene_scores, nan=0)
    np.fill_diagonal(gene_scores, 0)
    print(f"Patient {patient_name}: Gene-gene connections have a distribution between "
          f"{np.min(gene_scores)} : {np.max(gene_scores)} with mean {np.mean(gene_scores)}")
    gene_scores[gene_scores <= args.grn_cutoff] = 0
    gene_scores[gene_scores > args.grn_cutoff] = 1
    print(
        f"Patient {patient_name}: Total gene-gene connections: {np.sum(gene_scores)} from {gene_scores.shape[0]} genes")
    pd.DataFrame(gene_scores, index=DC8_PRECIESADS.columns, columns=DC8_PRECIESADS.columns).to_csv(
        os.path.join(connections_path, f"{patient_name}_gene_gene_connections.csv"))

debuginfoStr('Gene-gene connections calculation finished')
