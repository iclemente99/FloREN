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
parser.add_argument('--cell_rate', type=float, default=0.9)
parser.add_argument('--gene_rate', type=float, default=0.3)

# Result
parser.add_argument('--data_name', type=str, default='FloREN-hgt', help='The name for dataset')
parser.add_argument('--reduction', type=str, default='AE', help='the method for feature extraction, pca, raw, AE')
parser.add_argument('--in_dim', type=int, default=256, help='Number of hidden dimension (AE)')
parser.add_argument('--floren_grn', type=str, default='True', help='Decide if to run the floren published gene-gene connections inference')

# GAE
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--n_hid', type=int, help='Number of hidden dimension')
parser.add_argument('--n_heads', type=int, help='Number of attention head')
parser.add_argument('--n_layers', type=int, default=2, help='Number of GNN layers')
parser.add_argument('--dropout', type=float, default=0, help='Dropout ratio')
parser.add_argument('--lr', type=float, help='learning rate')
parser.add_argument('--batch_size', type=int, help='Number of output nodes for training')
parser.add_argument('--layer_type', type=str, default='hgt', help='the layer type for GAE')
parser.add_argument('--loss', type=str, default='kl', help='the loss for GAE')
parser.add_argument('--factor', type=float, default='0.5', help='the attenuation factor')
parser.add_argument('--patience', type=int, default=5, help='patience')
parser.add_argument('--rf', type=float, default='0.0', help='the weights of regularization')
parser.add_argument('--cuda', type=int, default=1, help='cuda 0 use GPU0 else cpu')
parser.add_argument('--rep', type=str, default='T', help='precision truncation')
parser.add_argument('--AEtype', type=int, default=1, help='AEtype:1 embedding node autoencoder 2:HGT node autoencode')
parser.add_argument('--optimizer', type=str, default='adamw', help='optimizer')

# Data directories
parser.add_argument('--data_path', default='~/data', type=str, help='Path to folder with graph construction outputs')
parser.add_argument('--output_path', default='~/hgt_input', type=str, help='Path to folder with graph construction outputs')

args = parser.parse_args()
args.epoch = 100
args.n_hid = 128
args.n_heads = 13
args.lr = 0.1
args.n_batch = 32
sample_name = args.data_name

# Set up result directories
#output_path = os.path.abspath(os.path.expanduser(args.output_path))
output_path = "C:/Users/Inigo/Desktop/FloREN3.0/binvignat_output"
embeddings_path = os.path.join(output_path, "embeddings")
gene_embeddings_path = os.path.join(embeddings_path, "gene_embeddings")
cell_embeddings_path = os.path.join(embeddings_path, "cell_embeddings")
connections_path = os.path.join(output_path, "connections")
os.makedirs(gene_embeddings_path, exist_ok=True)
os.makedirs(cell_embeddings_path, exist_ok=True)
os.makedirs(connections_path, exist_ok=True)

args.cuda = 1
# Set device
if args.cuda == 0:
    device = torch.device("cuda:0")
    print("cuda>>>")
else:
    device = torch.device("cpu")

print(device)





#---------------------------------------------------------------
#
#                 DATA PROCESSING BEFORE HGT
#
#---------------------------------------------------------------

data_path = "C:/Users/Inigo/Desktop/FloREN/Binvignat/genes"
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

gene_embeddings_norm_path = os.path.join(embeddings_path, "gene_embeddings_norm")
cell_embeddings_norm_path = os.path.join(embeddings_path, "cell_embeddings_norm")

# ---------------------------------------------------------------
#
#            GENERATE HETEROGENEOUS KNN PRELIMINAR ADJACENCY MATRIX
#
# ---------------------------------------------------------------

# Get patient names
# cell_names = pd.read_csv(os.path.join(output_path, "All_AUC_Cell_names.csv"))
cell_names = pd.read_csv(os.path.join(output_path, "All_AUC_Cell_names.csv"))
samples_names = []
for name in cell_names['0']:
    samples_names.append(re.split('__', name, 2)[0])

files = list(set(samples_names))

patient_graphs = []

# for file in range(len(files)):
for file_idx, patient_name in enumerate(files):
    # patient_name = files[file]
    print(f"Processing patient: {patient_name}")
    #
    # -------------------------
    # Gene-cell adjacency
    # -------------------------
    load_file = [f for f in csv_files if patient_name in f][0]
    transformed_matrix = pd.read_csv(load_file)
    transformed_matrix = transformed_matrix.iloc[:, 1:].values
    gene_cell = transformed_matrix
    gene_cell[gene_cell > 0] = 1
    # gene_cell[gene_cell <= 5] = 0
    #
    # -------------------------
    # Gene-gene adjacency (per patient)
    # -------------------------
    gene_gene = pd.read_csv(os.path.join(connections_path, f"{patient_name}_gene_gene_connections.csv"))
    gene_gene.drop(columns=gene_gene.columns[0], axis=1, inplace=True)
    gene_gene.set_index(gene_gene.columns, inplace=True)
    genes_intrsct = list(set(gene_names) & set(gene_gene.columns))
    genes_f = gene_gene[genes_intrsct]
    genes_f = genes_f.loc[genes_intrsct]
    genes_f = np.concatenate(
        [np.array(genes_f), np.zeros((genes_f.shape[0], len(set(genes_intrsct) ^ set(gene_names))))], axis=1)
    genes_f = pd.DataFrame(
        np.concatenate([np.array(genes_f), np.zeros((len(set(genes_intrsct) ^ set(gene_names)), genes_f.shape[1]))],
                       axis=0))
    genes_f.columns = genes_intrsct + list(set(genes_intrsct) ^ set(gene_names))
    genes_f.set_index(genes_f.columns, inplace=True)
    genes_f = genes_f.reindex(sorted(genes_f.columns), axis=1)
    genes_f = genes_f.reindex(sorted(genes_f.index), axis=0)
    np.fill_diagonal(genes_f.values, 0)
    genes_f[genes_f > 0] = 1
    #
    # -------------------------
    # Cell-cell adjacency (per patient)
    # -------------------------
    cells_f = pd.read_csv(
        f"/Users/Inigo/Desktop/FloREN/Binvignat/cell_connections/"
        + str.split(load_file, '/genes')[1]
    )
    cells_f.set_index(cells_f.columns[0], inplace=True)
    #
    # -------------------------
    # Gene embeddings (per patient)
    # -------------------------
    # genes_files = pd.read_csv(os.path.join(gene_embeddings_path, f"{patient_name}_AE_Emb_Genes.csv"))
    # genes_files.drop(genes_files.columns[0], axis=1, inplace=True)
    # encoded = torch.tensor(genes_files.values, dtype=torch.float32).to(device)
    genes_files = pd.read_csv(os.path.join(gene_embeddings_norm_path, f"{patient_name}_AE_Emb_Genes.csv"))
    genes_files.drop(genes_files.columns[0], axis=1, inplace=True)
    encoded = torch.tensor(genes_files.values, dtype=torch.float32).to(device)
    print(encoded.shape)
    #
    # -------------------------
    # Cell embeddings (per patient)
    # -------------------------
    # cells_files = pd.read_csv(os.path.join(cell_embeddings_path, f"{patient_name}_AE_Emb_Cells.csv"))
    # cells_files.drop(cells_files.columns[0], axis=1, inplace=True)
    # encoded2 = torch.tensor(cells_files.values, dtype=torch.float32).to(device)
    cells_files = pd.read_csv(os.path.join(cell_embeddings_norm_path, f"{patient_name}_AE_Emb_Cells.csv"))
    cells_files.drop(cells_files.columns[0], axis=1, inplace=True)
    encoded2 = torch.tensor(cells_files.values, dtype=torch.float32).to(device)
    print(encoded2.shape)
    #
    # encoded = normalized_patient_gene_embeddings[file_idx]   # [n_genes, h_n]
    # encoded2 = normalized_patient_cell_embeddings[file_idx]
    # print(file_idx)
    # -------------------------
    # Build patient graph
    # -------------------------
    graph = build_graph(gene_cell, genes_f, cells_f, encoded, encoded2)
    #
    # Store
    patient_graphs.append({
        "name": patient_name,
        "graph": graph,
        "cells_idx": cell_names[cell_names['0'].str.contains(patient_name)].index.values
    })

print(f"Built {len(patient_graphs)} patient graphs")
debuginfoStr('Build Graph finished')

args.epoch = 100  # Hyperparameter specification
args.n_hid = 128  # Hyperparameter specification
args.n_heads = 8  # Hyperparameter specification
# args.n_heads = 13 # Hyperparameter specification
# args.lr = 0.1 # Hyperparameter specification
args.lr = 0.0005  # Hyperparameter specification
args.lr.txt = '01'
# args.n_batch = 32 # Hyperparameter specification
# sample_name = re.split('TFs_',tfs_files[file],2)[-1][:re.split('TFs_',tfs_files[file],2)[-1].index(".")]
sample_name = "Binvignat"
file0 = f'sample_{sample_name}_epoch_{args.epoch}_n_hid_{args.n_hid}_nheads_{args.n_heads}_lr_01_n_batch{args.n_batch}'  # Text saving hypermarametrization
# print(f'\n{file0}') #Print hyperparametrization
args.result_dir = '/Users/Inigo/Desktop/FloREN3.0/binvignat_output'  # Sets parent directory for output
gene_dir = args.result_dir + '/gene/'  # Sets directory for gene output
cell_dir = args.result_dir + '/cell/'  # Sets directory for cell output
model_dir = args.result_dir + '/model/'  # Sets directory for model output
att_dir = args.result_dir + '/att/'  # Sets directory for attention output
loss_dir = args.result_dir + '/loss/'  # Sets directory for loss output
inter_dir = args.result_dir + '/interpretability/'  # Sets directory for loss output
os.makedirs(gene_dir, exist_ok=True)
os.makedirs(cell_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(att_dir, exist_ok=True)
os.makedirs(loss_dir, exist_ok=True)
os.makedirs(inter_dir, exist_ok=True)
# args.cuda = 1
# if args.cuda == 0:
#    device = torch.device("cuda:" + "0") # Sets space to GPU
#    print("cuda>>>")
# else:
#    device = torch.device("cpu") # Sets space to CPU


print(device)  # Print device. IMPORTANT for incompatibilities.

metadata = pd.read_csv("/Users/Inigo/Desktop/FloREN/Binvignat/metadata_updated.csv")


def stratified_split_graphs(patient_graphs, metadata_df, test_size=0.15, val_size=0.15, random_state=42):
    # Make lookup from metadata
    meta_lookup = dict(zip(metadata_df["patient_id"], metadata_df["group"]))
    patient_names = [pg["name"] for pg in patient_graphs]
    labels = [meta_lookup[name] for name in patient_names]
    train_val_graphs, test_graphs, train_val_labels, test_labels = train_test_split(
        patient_graphs, labels,
        test_size=test_size, stratify=labels, random_state=random_state
    )
    relative_val_size = val_size / (1 - test_size)
    train_graphs, val_graphs, train_labels, val_labels = train_test_split(
        train_val_graphs, train_val_labels,
        test_size=relative_val_size, stratify=train_val_labels, random_state=random_state
    )
    return train_graphs, val_graphs, test_graphs


train_graphs, val_graphs, test_graphs = stratified_split_graphs(
    patient_graphs, metadata, test_size=0.05, val_size=0.05
)

print("Sampling end!")
debuginfoStr('Cell Graph constructed and pruned')

gnn = GNN(conv_name=args.layer_type, in_dim=256,
          n_hid=args.n_hid, n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout,
          num_types=2, num_relations=7, use_RTE=False, n_labels=(np.max(metadata.group) + 1)
          ).to(device)  # Loads HGT function
# gnn = GNN_class(conv_name=args.layer_type, in_dim=256,
#                n_hid=args.n_hid, n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout,
#                num_types=2, num_relations=6, use_RTE=False,
#                h_out=patient_loss.shape[1]
#                ).to(device) # Loads HGT function
# Options to use different optimization algorithms. Default: adamw
if args.optimizer == 'adamw':
    optimizer = torch.optim.AdamW(gnn.parameters(), lr=args.lr)
elif args.optimizer == 'adam':
    optimizer = torch.optim.Adam(gnn.parameters(), lr=args.lr)
elif args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(gnn.parameters(), lr=args.lr)
elif args.optimizer == 'adagrad':
    optimizer = torch.optim.Adagrad(gnn.parameters(), lr=args.lr)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', factor=args.factor, patience=args.patience, verbose=True)

import time
import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split

# --- DEVICE: CPU fallback if CUDA not available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)

# --- Loss fns (conservative choice: BCE for multi-hot/one-hot, you can change if needed)
# contrastive_loss_fn = nn.BCEWithLogitsLoss(reduction='sum')   # for logits vs lbl (0/1)
# classification_loss_fn = nn.BCEWithLogitsLoss(reduction='sum') # for multi-class one-hot labels (if you have softmax/CELoss you can swap)
contrastive_loss_fn = nn.BCEWithLogitsLoss()  # Load Self-supervised loss for classification learning
classification_loss_fn = nn.CrossEntropyLoss()  # Load Cross Entropy loss for classification learning

# --- Hyperparams for dynamic multi-tasking & early stopping
nb_epochs = 150
weight_contrastive = 0.1
weight_classification = 0.9
first_cutoff = False
first_time = False
contrastive_patience = 0
best_contrastive = float('inf')

patience = 70  # patience for early stopping (validation)
counter = 0
best_model_loss = float('inf')
best_model_state = None

# optionally freeze gnn later: we will check a flag
# freeze_gcn_when = None  # if you want to explicitly set epoch to freeze GCN; else keep None

# function that returns a list of jobs for a given patient graph (tries to be compatible with your preprocessing)
# def get_jobs_for_patient(patient_graph, n_batch=args.n_batch, cell_rate=args.cell_rate, gene_rate=args.gene_rate):
#    """
#    Return the list `jobs` needed by your inner training loop.
#    If `patient_graph` already has a 'jobs' key (created previously), use it.
#    Otherwise, call `sub_sample` (assumed to exist in your pipeline) to generate jobs.
#    Each job should be (feature, time, edge_list, indxs) like your original loop expects.
#    """
#    if isinstance(patient_graph, dict) and 'jobs' in patient_graph:
#        return patient_graph['jobs']
#    # fallback: attempt to call sub_sample on full graph if available
#    if isinstance(patient_graph, dict) and 'graph' in patient_graph:
#        try:
#            # sub_sample should accept graph and return a list of jobs (as in your prior pipeline)
#            return sub_sample(patient_graph['graph'], n_batch=n_batch,
#                              cell_rate=cell_rate, gene_rate=gene_rate)
#        except Exception as e:
#            raise RuntimeError("Couldn't generate jobs for patient. Please precompute 'jobs' or ensure sub_sample() works.") from e
#    raise RuntimeError("patient_graph must be dict with either 'jobs' or 'graph' key.")

# --- Prepare train/val/test lists (you already produced these previously)
# train_graphs, val_graphs, test_graphs = stratified_split_graphs(patient_graphs, metadata_df, ...)

# Example dynamic-multi-task schedule variables (you can tune)
weight_decay_step = 0.001
weight_decay_min = 0.2
weight_inc_step = 0.001
weight_inc_max = 0.8

# Some bookkeeping arrays for plotting later
training_losses = []
validation_losses = []
contrastive_losses = []
classification_losses = []
total_losses = []

args.gene_rate = 0.1
args.cell_rate = 0.1

# Set random seed for reproducibility
random_seed = 42
random.seed(random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)


def pregenerate_jobs(graphs, csv_files, gene_rate, cell_rate, device):
    jobs_dict = {}
    for patient in graphs:
        patient_name = patient['name']
        # Get graph and gene_cell matrix
        graph = patient['graph']
        load_file = [f for f in csv_files if patient_name in f][0]
        transformed_matrix = pd.read_csv(load_file)
        transformed_matrix = transformed_matrix.iloc[:, 1:].values
        gene_cell = transformed_matrix
        #
        # Generate jobs
        jobs = []
        gene_num = int(gene_cell.shape[0] * gene_rate)
        cell_num = int(gene_cell.shape[1] * cell_rate)
        jobs.append(
            sub_sample(graph, gene_cell, gene_cell.shape[1], gene_cell.shape[0], gene_cell.shape[0], gene_cell.shape[1],
                       query=True))
        jobs.append(sub_sample(graph, gene_cell, cell_num, gene_num, gene_cell.shape[0], gene_cell.shape[1]))
        jobs.append(sub_sample(graph, gene_cell, cell_num, gene_num, gene_cell.shape[0], gene_cell.shape[1]))
        #
        # Format jobs into gnn_input tensors
        formatted_jobs = []
        for job in jobs:
            feature, time_info, edge_list, indxs, og_gene_indxs, og_cell_indxs = job
            node_dict = {}
            node_feature = []
            node_type = []
            node_time = []
            edge_index = []
            edge_type = []
            edge_time = []
            # print(f"Jobs: edge_index - {edge_index}")
            node_num = 0
            types = graph.get_types()
            for t in types:
                node_dict[t] = [node_num, len(node_dict)]
                node_num += len(feature[t])
            #
            for t in types:
                # t_i = node_dict[t][1]
                # node_feature.append(torch.tensor(feature[t], dtype=torch.float32).to(device))
                # node_time += list(time_info[t])
                # node_type += [node_dict[t][1] for _ in range(len(feature[t]))]
                t_i = node_dict[t][1]
                node_feature.insert(t_i, torch.tensor(feature[t], dtype=torch.float32).to(
                    device))  # Saves cells or genes matrices
                node_time += list(time_info[t])  # Saves node time
                node_type += [node_dict[t][1] for _ in range(len(feature[t]))]  # Saves node type
            #
            edge_dict = {e[2]: i for i, e in enumerate(graph.get_meta_graph())}
            edge_dict['self'] = len(edge_dict)
            # if formatted_jobs == []:
            #    print(edge_dict)
            for target_type in edge_list:
                for source_type in edge_list[target_type]:
                    for relation_type in edge_list[target_type][source_type]:
                        for ti, si in edge_list[target_type][source_type][relation_type]:
                            tid = ti + node_dict[target_type][0]
                            sid = si + node_dict[source_type][0]
                            edge_index.append([sid, tid])
                            edge_type.append(edge_dict.get(relation_type, 0))
                            edge_time.append(120)
            #
            # node_feature = torch.cat(tuple(node_feature), 0).to(device)
            node_feature = torch.cat((node_feature[0], node_feature[1]), 0)
            node_type = torch.LongTensor(node_type).to(device)
            edge_time = torch.LongTensor(edge_time).to(device)
            edge_index = torch.LongTensor(edge_index).t().to(device)
            edge_type = torch.LongTensor(edge_type).to(device)
            # print(f"Jobs: edge_index - {edge_index.max()}")
            formatted_jobs.append(
                [node_feature, node_type, edge_time, edge_index, edge_type, og_gene_indxs, og_cell_indxs])
        #
        jobs_dict[patient_name] = formatted_jobs
    #
    print(edge_dict)
    return jobs_dict


# Generate and format jobs
print("Pre-generating and formatting jobs...")
train_jobs = pregenerate_jobs(train_graphs, csv_files, args.gene_rate, args.cell_rate, device)
val_jobs = pregenerate_jobs(val_graphs, csv_files, args.gene_rate, args.cell_rate, device)
# print("Pre-generated and formatted jobs for all patients")
debuginfoStr('Pre-generated and formatted jobs for all patients')



################################################################
#
# APPLY HGT TO WHOLE GRAPH
#
################################################################

# Directories for saving results
gene_dir = "/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/floren_gene_embeddings/"
cell_dir = "/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/floren_cell_embeddings/"
att_dir = "/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/floren_attention_scores/"
patient_emb_dir = "/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/floren_patient_embeddings/"
patient_emb_dir_split = "/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/floren_patient_embeddings/split/"
os.makedirs(gene_dir, exist_ok=True)
os.makedirs(cell_dir, exist_ok=True)
os.makedirs(att_dir, exist_ok=True)
os.makedirs(patient_emb_dir, exist_ok=True)
os.makedirs(patient_emb_dir_split, exist_ok=True)

# Load metadata
#metadata = pd.read_csv("/home/inigo/Desktop/FloREN3.0/Binvignat/metadata_updated.csv")

# Device setup
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print("Using device:", device)

# Load trained model
#model_name = f'Binvignat_epoch_{args.epoch}_n_hid_{args.n_hid}_nheads_{args.n_heads}_lr_01_n_batch{args.n_batch}'
#model_path = os.path.join(model_dir, model_name)
model0=f'Binvignat_epoch_{args.epoch}_n_hid_{args.n_hid}_nheads_{args.n_heads}_lr_01_n_batch{args.n_batch}'
state = torch.load(model_dir+model0, map_location=lambda storage, loc: storage) # Loads model
print(f"Loaded model from {model_dir+model0}")

test_jobs = pregenerate_jobs(test_graphs, csv_files, args.gene_rate, args.cell_rate, device)

# Initialize GNN model
gnn = GNN(
    conv_name=args.layer_type,
    in_dim=args.in_dim,  # Should match training (e.g., 256)
    n_hid=args.n_hid,    # Should match training (e.g., 128)
    n_heads=args.n_heads,  # Should match training (e.g., 8)
    n_layers=args.n_layers,
    dropout=args.dropout,
    num_types=2,
    num_relations=7,
    n_labels=(np.max(metadata.group)+1),  # Should match training (e.g., 2 for Control vs RA)
    use_RTE=False
).to(device)
gnn.load_state_dict(state['model'])
gnn.eval()

# Combine train and validation graphs for inference
all_graphs = train_graphs + val_graphs + test_graphs
all_jobs = {**train_jobs, **val_jobs, **test_jobs}  # Combine train_jobs and val_jobs















#-----------------------------------------------------
#
# UMAP CLEAN
#
#-----------------------------------------------------

import matplotlib.pyplot as plt
import umap
import smtplib
from email.message import EmailMessage
import mimetypes
import ssl
plots_dir = "/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/plots"

# === UMAP SAMPLES PLOT ===
vector_path = "/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/floren_patient_embeddings/"  # Path to saved patient embeddings
vector_files = [f for f in os.listdir(vector_path) if f.endswith(".csv")]
vector_files = [f for f in vector_files if "n_hid_128_nheads_8" in f]
all_vectors = []
labels = []
for file in vector_files:
    df = pd.read_csv(os.path.join(vector_path, file), header=None)  # No index column
    all_vectors.append(df.values.squeeze())  # Shape: [552]
    label = "RA" if "RA" in file else "HD"
    labels.append(label)

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
pdf_path = os.path.join(plots_dir, "FloREN_umap_clean.pdf")

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
    colors = {"RA": "red", "HD": "blue"}
    labels_np = np.array(labels)
    #
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
            #ax.grid(True)
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















#-----------------------------------------------------
#
# ALL CELL TYPES SUM SUP FIG
#
#-----------------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import glob

result_dir = '/Users/Inigo/Desktop/FloREN3.0/binvignat_output'  # Sets parent directory for output
inter_dir = result_dir + '/interpretability/'  # Sets directory for loss output
att_dir = "/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/floren_attention_scores/"
data_path = "C:/Users/Inigo/Desktop/FloREN/Binvignat/genes"
csv_files = glob.glob(os.path.join(data_path, "*.csv"))
reference = pd.read_csv(csv_files[0])
gene_names = reference[reference.columns[0]].values
n_genes = reference.shape[0]
output_path = "C:/Users/Inigo/Desktop/FloREN3.0/binvignat_output"
cell_names = pd.read_csv(os.path.join(output_path, "All_AUC_Cell_names.csv"))

plt.rcParams["figure.figsize"] = (14, 6)
plt.rcParams["font.size"] = 12

##############################################################
# 1. Build list of patients from your earlier processing
##############################################################

patients = [d for d in os.listdir(inter_dir) if os.path.isdir(os.path.join(inter_dir, d))]
patients.sort()

print("Patients detected:", patients)

##############################################################
# 2. Cell metadata processing helper
##############################################################

meta = pd.read_csv("/Users/Inigo/Desktop/FloREN/Binvignat/cells_metadata.csv")
# Example: "classical monocyte__RA09_1_C_M__GGGACCTGTGCGAA-1"
meta_split = meta["x"].str.split("__", expand=True)
meta["celltype"] = meta_split[0]
meta["barcode"] = meta_split[1] + "__" + meta_split[2]  # e.g. "GGGACCTGTGCGAA-1"
meta["patient"] = meta_split[1]  # e.g. "RA09_1_C_M"

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
    grouped = merged.groupby("celltype")["saliency"].sum().sort_values(ascending=False)
    #
    celltype_df = grouped.reset_index()
    celltype_df["patient"] = patient
    #
    # Save
    #celltype_df.to_csv(os.path.join(pdir, "Celltype_att_saliency.csv"), index=False)
    #
    all_patients_cell_df.append(celltype_df)

# ===================== 4. Combine Across Patients ====================
all_genes = pd.concat(all_patients_gene_df)
all_cells = pd.concat(all_patients_cell_df)

# Compute mean across patients
mean_gene_saliency = all_genes.groupby("gene")["saliency"].mean().sort_values(ascending=False)
mean_celltype_saliency = all_cells.groupby("celltype")["saliency"].mean().sort_values(ascending=False)

from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

# Output
plots_dir = "/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/plots"
output_pdf = os.path.join(
    plots_dir,
    "FloREN_Celltype_Saliency_Violinplots.pdf"
)

# Order cell types by mean saliency (already computed)
ordered_cells = mean_celltype_saliency.index.tolist()

# Keep only ordered cells
plot_df = all_cells.copy()
plot_df = plot_df.reset_index(drop=True)
#plot_df = plot_df.dropna(subset=["celltype"])
plot_df["celltype"] = pd.Categorical(
    plot_df["celltype"],
    categories=ordered_cells,
    ordered=True
)
#plot_df["celltype"] = pd.Categorical(
#    plot_df["celltype"].astype(str),
#    categories=ordered_cells,
#    ordered=True
#)

# Color palette (one color per cell type)
palette = sns.color_palette("tab20", n_colors=len(ordered_cells))

with PdfPages(output_pdf) as pdf:
    # =========================================================
    # PAGE 1 — Linear scale
    # =========================================================
    fig, ax = plt.subplots(figsize=(22, 8))
    sns.violinplot(
        data=plot_df,
        x="celltype",
        y="saliency",
        palette=palette,
        inner=None,      # no internal bars
        cut=0,
        linewidth=1,
        ax=ax
    )
    sns.stripplot(
        data=plot_df,
        x="celltype",
        y="saliency",
        color="black",
        size=3,
        jitter=0.25,
        alpha=0.6,
        ax=ax
    )
    ax.set_title("Cell Type Saliency Distribution (All Patients)", fontsize=16)
    ax.set_xlabel("")
    ax.set_ylabel("Saliency Score")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
    # =========================================================
    # PAGE 2 — Log scale
    # =========================================================
    fig, ax = plt.subplots(figsize=(22, 8))
    sns.violinplot(
        data=plot_df,
        x="celltype",
        y="saliency",
        palette=palette,
        inner=None,
        cut=0,
        linewidth=1,
        ax=ax
    )
    sns.stripplot(
        data=plot_df,
        x="celltype",
        y="saliency",
        color="black",
        size=3,
        jitter=0.25,
        alpha=0.6,
        ax=ax
    )
    ax.set_yscale("log")
    ax.set_title("Cell Type Saliency Distribution (Log Scale)", fontsize=16)
    ax.set_xlabel("")
    ax.set_ylabel("Log Saliency Score")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
















#-----------------------------------------------------
#
# ALL CELL TYPES MEAN SUP FIG
#
#-----------------------------------------------------

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
    grouped = merged.groupby("celltype")["saliency"].mean().sort_values(ascending=False)
    #
    celltype_df = grouped.reset_index()
    celltype_df["patient"] = patient
    #
    # Save
    #celltype_df.to_csv(os.path.join(pdir, "Celltype_att_saliency.csv"), index=False)
    #
    all_patients_cell_df.append(celltype_df)

# ===================== 4. Combine Across Patients ====================
all_genes = pd.concat(all_patients_gene_df)
all_cells = pd.concat(all_patients_cell_df)

# Compute mean across patients
mean_gene_saliency = all_genes.groupby("gene")["saliency"].mean().sort_values(ascending=False)
mean_celltype_saliency = all_cells.groupby("celltype")["saliency"].mean().sort_values(ascending=False)

from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

# Output
plots_dir = "/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/plots"
output_pdf = os.path.join(
    plots_dir,
    "FloREN_Celltype_Saliency_Violinplots_mean.pdf"
)

# Order cell types by mean saliency (already computed)
ordered_cells = mean_celltype_saliency.index.tolist()

# Keep only ordered cells
plot_df = all_cells.copy()
plot_df = plot_df.reset_index(drop=True)
#plot_df = plot_df.dropna(subset=["celltype"])
plot_df["celltype"] = pd.Categorical(
    plot_df["celltype"],
    categories=ordered_cells,
    ordered=True
)
#plot_df["celltype"] = pd.Categorical(
#    plot_df["celltype"].astype(str),
#    categories=ordered_cells,
#    ordered=True
#)

# Color palette (one color per cell type)
palette = sns.color_palette("tab20", n_colors=len(ordered_cells))

with PdfPages(output_pdf) as pdf:
    # =========================================================
    # PAGE 1 — Linear scale
    # =========================================================
    fig, ax = plt.subplots(figsize=(22, 8))
    sns.violinplot(
        data=plot_df,
        x="celltype",
        y="saliency",
        palette=palette,
        inner=None,      # no internal bars
        cut=0,
        linewidth=1,
        ax=ax
    )
    sns.stripplot(
        data=plot_df,
        x="celltype",
        y="saliency",
        color="black",
        size=3,
        jitter=0.25,
        alpha=0.6,
        ax=ax
    )
    ax.set_title("Cell Type Saliency Distribution (All Patients)", fontsize=16)
    ax.set_xlabel("")
    ax.set_ylabel("Saliency Score")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
    # =========================================================
    # PAGE 2 — Log scale
    # =========================================================
    fig, ax = plt.subplots(figsize=(22, 8))
    sns.violinplot(
        data=plot_df,
        x="celltype",
        y="saliency",
        palette=palette,
        inner=None,
        cut=0,
        linewidth=1,
        ax=ax
    )
    sns.stripplot(
        data=plot_df,
        x="celltype",
        y="saliency",
        color="black",
        size=3,
        jitter=0.25,
        alpha=0.6,
        ax=ax
    )
    ax.set_yscale("log")
    ax.set_title("Cell Type Saliency Distribution (Log Scale)", fontsize=16)
    ax.set_xlabel("")
    ax.set_ylabel("Log Saliency Score")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
















#-----------------------------------------------------
#
# ALL CELL TYPES MEAN TOP 5
#
#-----------------------------------------------------

# ============================
# USER CONTROLS
# ============================
TOP_N = 5
output_pdf = os.path.join(
    plots_dir,
    f"FloREN_Celltype_Saliency_Top{TOP_N}_ra_hd_mean.pdf"
)

# ============================
# PREPARE DATA
# ============================
top_celltypes = mean_celltype_saliency.head(TOP_N).index.tolist()

plot_df = all_cells[all_cells["celltype"].isin(top_celltypes)].copy()
plot_df = plot_df.reset_index(drop=True)
plot_df["celltype"] = pd.Categorical(
    plot_df["celltype"],
    categories=top_celltypes,
    ordered=True
)

# Define condition
plot_df["condition"] = plot_df["patient"].apply(
    lambda p: "RA" if "RA" in p else "HD"
)

# ============================
# STYLE
# ============================
sns.set_theme(style="white")
plt.rcParams.update({
    "font.size": 12,
    "axes.linewidth": 1.2,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

# Cell-type palette (violins)
violin_palette = sns.color_palette("Set2", n_colors=TOP_N)

# Condition colors (distinct from violins)
condition_palette = {
    "RA": "#B22222",   # firebrick (muted red)
    "HD": "#1F4E79"    # steel blue (muted blue)
}

# ============================
# PLOT
# ============================
with PdfPages(output_pdf) as pdf:
    fig, ax = plt.subplots(figsize=(5 + TOP_N * 1.2, 5))
    # Violin plots (cell-type colored)
    sns.violinplot(
        data=plot_df,
        x="celltype",
        y="saliency",
        palette=violin_palette,
        inner=None,
        cut=0,
        width=0.8,
        linewidth=1.2,
        ax=ax
    )
    # Jittered points (condition colored)
    sns.stripplot(
        data=plot_df,
        x="celltype",
        y="saliency",
        hue="condition",
        palette=condition_palette,
        size=4,
        jitter=0.25,
        alpha=0.75,
        dodge=False,
        ax=ax
    )
    # Median markers (overall)
    medians = plot_df.groupby("celltype")["saliency"].mean()
    ax.scatter(
        range(len(medians)),
        medians.values,
        color="black",
        s=60,
        zorder=5,
        label="Median"
    )
    # Log scale
    ax.set_yscale("log")
    # Labels & title
    ax.set_title(
        f"Top {TOP_N} Cell Types by Attention Saliency",
        fontsize=14,
        pad=10
    )
    ax.set_xlabel("")
    ax.set_ylabel("Attention Saliency (log scale)")
    # Axis cleanup
    #ax.tick_params(axis="x", rotation=30)
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=30,
        ha="right"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Legend cleanup
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[:2],
        labels[:2],
        title="Condition",
        frameon=False,
        loc="upper right",
        fontsize=5,
        title_fontsize=5
    )
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)















#-----------------------------------------------------
#
# CELL TYPES GENE SIGNATURES
#
#-----------------------------------------------------

# Assuming these are already defined in your environment:
# - patients
# - inter_dir
# - att_dir
# - cell_names
# - meta
# - n_genes
# - gene_names   ← very important! list/array of gene names in order (length = n_genes)

# List of all unique cell types (from your output)
cell_types = mean_celltype_saliency.index.unique().tolist()

# We'll collect one aggregated vector per cell type
celltype_vectors = {}

for target_celltype in cell_types:
    print(f"\n=== Processing cell type: {target_celltype} ===")
    all_patients_gene_df = []
    for patient in patients:
        print(f"  Patient: {patient}")
        pdir = os.path.join(inter_dir, patient)
        # Load edge attention for this patient
        df2 = pd.read_csv(os.path.join(att_dir, f"{patient}_edge_att.csv"), delimiter=",").iloc[:, 1:]
        # Get cells for this patient
        patient_mask = cell_names["0"].str.startswith(patient + "__")
        patient_cell_names = cell_names.loc[patient_mask, "0"].reset_index(drop=True)
        cell_df = pd.DataFrame({
            "local_cell_index": np.arange(len(patient_cell_names)),
            "barcode_full": patient_cell_names
        })
        cell_df["barcode"] = cell_df["barcode_full"]
        cell_df["patient"] = patient
        # Map to global node index (cells start after genes)
        cell_df["node_index"] = cell_df["local_cell_index"] + n_genes
        # Merge with metadata to get celltype
        cell_df = cell_df.merge(
            meta[["barcode", "celltype", "patient"]],
            on=["barcode", "patient"],
            how="left"
        )
        # Select only target cell type cells
        target_cells = cell_df[
            (cell_df["celltype"] == target_celltype) &
            (cell_df["patient"] == patient)
            ]["node_index"].values
        # Filter edges connected to target cells and genes
        df2.columns = ["src", "tgt", "att"]
        mask = (
                ((df2["src"].isin(target_cells)) & (df2["tgt"] < n_genes)) |
                ((df2["tgt"].isin(target_cells)) & (df2["src"] < n_genes))
        )
        df_celltype_edges = df2[mask].copy()
        # Accumulate attention per gene
        gene_attention = np.zeros(n_genes, dtype=np.float64)
        # Gene as source
        gene_mask_src = df_celltype_edges["src"] < n_genes
        np.add.at(
            gene_attention,
            df_celltype_edges.loc[gene_mask_src, "src"].values,
            df_celltype_edges.loc[gene_mask_src, "att"].values
        )
        # Gene as target
        gene_mask_tgt = df_celltype_edges["tgt"] < n_genes
        np.add.at(
            gene_attention,
            df_celltype_edges.loc[gene_mask_tgt, "tgt"].values,
            df_celltype_edges.loc[gene_mask_tgt, "att"].values
        )
        # ✅ NEW: normalize by number of cells of this type in this patient
        #n_target_cells = len(target_cells)
        #if n_target_cells > 0:
        #    gene_attention /= n_target_cells
        #
        # Save per-patient file (optional but useful)
        out_dir = os.path.join(inter_dir, patient, "celltype_specific")
        os.makedirs(out_dir, exist_ok=True)
        #np.savetxt(
        #    os.path.join(out_dir, f"genes_att_{target_celltype.replace(' ', '_')}.csv"),
        #    gene_attention,
        #    delimiter=","
        #)
        # Create per-patient gene DataFrame
        gene_df = pd.DataFrame({
            "gene": gene_names,
            "saliency": gene_attention,
            "patient": patient,
            "celltype": target_celltype
        })
        all_patients_gene_df.append(gene_df)
    #
    # ── Aggregate across patients for this cell type ──
    if all_patients_gene_df:
        all_genes_celltype = pd.concat(all_patients_gene_df, ignore_index=True)
        # Compute mean saliency per gene across patients
        aggregated = all_genes_celltype.groupby('gene')['saliency'].mean().reset_index()
        aggregated.columns = ['gene', 'mean_attention']
        # Pivot to wide format (genes as columns)
        vector_wide = aggregated.set_index('gene')['mean_attention'].to_frame().T
        vector_wide.index = [target_celltype]
        celltype_vectors[target_celltype] = vector_wide
        print(f"  → Aggregated for {target_celltype}: {len(aggregated)} genes")
    else:
        print(f"  → No data for {target_celltype}")

# ── 5. Combine all cell types into one big matrix (like NMF W) ──
if celltype_vectors:
    celltype_attention_matrix = pd.concat(celltype_vectors.values(), axis=0)
    # Ensure all genes are present (fill missing with 0)
    celltype_attention_matrix = celltype_attention_matrix.reindex(columns=gene_names, fill_value=0.0)
    # Sort rows by cell type (optional)
    celltype_attention_matrix = celltype_attention_matrix.reindex(cell_types)
    print("\nFinal cell-type × gene attention matrix shape:", celltype_attention_matrix.shape)
    print(celltype_attention_matrix.iloc[:5, :8])  # preview
    # Save
    output_path = "/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/celltype_attention_matrix.csv"
    #celltype_attention_matrix.to_csv(output_path)
    print(f"Saved to: {output_path}")
else:
    print("No cell types had valid data.")

import mygene
mg = mygene.MyGeneInfo()
gene_map = mg.querymany(
    celltype_attention_matrix.columns.tolist(),
    scopes="ensembl.gene",
    fields="symbol",
    species="human",
    as_dataframe=True
)
map_series = gene_map["symbol"].fillna(gene_map.index.to_series())
celltype_attention_matrix.columns = celltype_attention_matrix.columns.map(map_series)

subset_matrix = celltype_attention_matrix.loc[
    celltype_attention_matrix.index.intersection(top_celltypes)
]
gene_sums = subset_matrix.sum(axis=0)
top_ngenes = 20
top_genes = gene_sums.sort_values(ascending=False).head(top_ngenes)
















#-----------------------------------------------------
#
# TOP N CELL TYPE GENE SIGNATURES - SUM
#
#-----------------------------------------------------

all_patients_gene_df = []
for target_celltype in top_celltypes:
    print(f"Processing cell type: {target_celltype}")
    for patient in patients:
        print(f"  Patient: {patient}")
        pdir = os.path.join(inter_dir, patient)
        df2 = pd.read_csv(
            os.path.join(att_dir, f"{patient}_edge_att.csv"),
            delimiter=","
        ).iloc[:, 1:]
        patient_mask = cell_names["0"].str.startswith(patient + "__")
        patient_cell_names = cell_names.loc[patient_mask, "0"].reset_index(drop=True)
        cell_df = pd.DataFrame({
            "local_cell_index": np.arange(len(patient_cell_names)),
            "barcode_full": patient_cell_names
        })
        cell_df["barcode"] = cell_df["barcode_full"]
        cell_df["patient"] = patient
        cell_df["node_index"] = cell_df["local_cell_index"] + n_genes
        cell_df = cell_df.merge(
            meta[["barcode", "celltype", "patient"]],
            on=["barcode", "patient"],
            how="left"
        )
        target_cells = cell_df.loc[
            cell_df["celltype"] == target_celltype,
            "node_index"
        ].values
        if len(target_cells) == 0:
            continue
        #
        df2.columns = ["src", "tgt", "att"]
        mask = (
            ((df2["src"].isin(target_cells)) & (df2["tgt"] < n_genes)) |
            ((df2["tgt"].isin(target_cells)) & (df2["src"] < n_genes))
        )
        df_celltype_edges = df2[mask]
        gene_attention = np.zeros(n_genes, dtype=np.float64)
        # src
        gene_mask = df_celltype_edges["src"] < n_genes
        np.add.at(
            gene_attention,
            df_celltype_edges.loc[gene_mask, "src"].values,
            df_celltype_edges.loc[gene_mask, "att"].values
        )
        # tgt
        gene_mask = df_celltype_edges["tgt"] < n_genes
        np.add.at(
            gene_attention,
            df_celltype_edges.loc[gene_mask, "tgt"].values,
            df_celltype_edges.loc[gene_mask, "att"].values
        )
        gene_df = pd.DataFrame({
            "gene": gene_names,
            "saliency": gene_attention,
            "patient": patient,
            "celltype": target_celltype
        })
        all_patients_gene_df.append(gene_df)


all_genes = pd.concat(all_patients_gene_df, ignore_index=True)
all_genes["gene"] = all_genes["gene"].map(lambda g: map_series.get(g, g))
plot_df = all_genes.copy()
ranked_genes = (
    plot_df
    .groupby("gene")["saliency"]
    .mean()
    .sort_values(ascending=False)
)
top_ngenes = 50
top_genes = ranked_genes.head(top_ngenes)
plot_df = plot_df[plot_df["gene"].isin(top_genes.index)].copy()
plot_df = plot_df.reset_index(drop=True)
plot_df = (
    plot_df
    .groupby(["gene", "patient"], as_index=False)["saliency"]
    .sum()
)
plot_df["gene"] = pd.Categorical(
    plot_df["gene"],
    categories=top_genes.index,
    ordered=True
)
plot_df["condition"] = plot_df["patient"].apply(
    lambda p: "RA" if "RA" in p else "HD"
)

output_pdf = os.path.join(
    plots_dir,
    f"FloREN_Top{TOP_N}_CellTypes_GeneSignature_Top{top_ngenes}_Genes_sum_log.pdf"
)
# ============================
# STYLE — Nature Methods
# ============================
sns.set_theme(style="white")

plt.rcParams.update({
    "font.size": 11,
    "axes.linewidth": 1.2,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

# 🎨 Colors
violin_color = "#F4A261"   # refined, soft orange
condition_palette = {
    "RA": "#8B1E3F",       # deep muted red (distinct from violin)
    "HD": "#264653"        # muted navy blue
}

# ============================
# PLOT
# ============================
from matplotlib.backends.backend_pdf import PdfPages

with PdfPages(output_pdf) as pdf:
    fig, ax = plt.subplots(
        #figsize=(8.2, 4.6)  # compact & journal-friendly
    )
    # ── Violin plots ─────────────────────────────
    sns.violinplot(
        data=plot_df,
        x="gene",
        y="saliency",
        color=violin_color,
        inner=None,
        cut=0,
        linewidth=1.1,
        ax=ax
    )
    # ── Per-patient points ───────────────────────
    sns.stripplot(
        data=plot_df,
        x="gene",
        y="saliency",
        hue="condition",
        palette=condition_palette,
        size=4,
        jitter=0.22,
        alpha=0.75,
        dodge=False,
        ax=ax
    )
    # ── Mean dots (USED FOR RANKING → CONSISTENT) ─
    mean_vals = (
        plot_df
        .groupby("gene")["saliency"]
        .mean()
        .loc[top_genes.index]
    )
    ax.scatter(
        np.arange(len(mean_vals)),
        mean_vals.values,
        color="black",
        s=42,
        zorder=5,
        label="Mean"
    )
    # ── Log scale ────────────────────────────────
    ax.set_yscale("log")
    # ── Labels & title ───────────────────────────
    ax.set_title(
        f"{target_celltype} — Top {top_ngenes} Unique Genes\n"
        "Ranked by specificity-weighted attention",
        fontsize=13,
        pad=8
    )
    ax.set_xlabel("")
    ax.set_ylabel("Attention saliency (log scale)")
    # ── Axis aesthetics ──────────────────────────
    #ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        ha="right"
    )
    ax.tick_params(axis="x", labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # ── Clean legend ─────────────────────────────
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[:2],
        labels[:2],
        title="Condition",
        frameon=False,
        loc="upper right",
        fontsize=5,
        title_fontsize=5
    )
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

















#-----------------------------------------------------
#
# SUPER EXTRA: PAGA-GRN ANALYSIS
#
#-----------------------------------------------------

# ── Compute top 250 genes by mean saliency across top 5 cell types ──
ranked_genes = all_genes.groupby("gene")["saliency"].mean().sort_values(ascending=False)
top_250_genes = ranked_genes.head(250).index.tolist()

print(f"Top 250 genes selected (by mean attention in top {len(top_celltypes)} cell types):")
print(top_250_genes[:10])  # preview

ensembl_to_symbol = map_series.to_dict()
symbol_to_ensembl = {sym: ens for ens, sym in ensembl_to_symbol.items()}
# Handle cases where multiple Ensembl map to same symbol (rare, but possible)
# We'll take the first one for simplicity
symbol_to_ensembl = {}
for ens, sym in ensembl_to_symbol.items():
    if sym not in symbol_to_ensembl:
        symbol_to_ensembl[sym] = ens

# Now map your top 250 symbols to their original Ensembl IDs
top_250_ensembl = []
for sym in top_250_genes:
    ens = symbol_to_ensembl.get(sym)
    if ens:
        top_250_ensembl.append(ens)
    else:
        print(f"Warning: No Ensembl ID found for symbol '{sym}'")

print(f"Found {len(top_250_ensembl)} / {len(top_250_genes)} Ensembl IDs for top symbols")

# Map gene → index (for filtering edges)
gene_to_index = {gene: i for i, gene in enumerate(gene_names)}
#top_250_indices = [gene_to_index.get(g, -1) for g in top_250_genes]
top_250_indices = [gene_to_index.get(ens, -1) for ens in top_250_ensembl]
top_250_indices = [i for i in top_250_indices if i != -1]  # remove any missing

# Optional: add condition column for nodes
def patient_condition(p):
    if "RA" in p:
        return "RA"
    elif "Control" in p or "HD" in p or "HC" in p:
        return "Control"  # adjust labels as needed
    else:
        return "Other"

all_genes["condition"] = all_genes["patient"].apply(patient_condition)

# ===================== GRN: Limited to Top 250 Genes from Top Cell Types =====================

# ── 1. Aggregate gene saliency (nodes) — from celltype attention, limited to top 250 ──
gene_condition_mean = all_genes.groupby(["condition", "gene"], as_index=False)["saliency"].mean()

# Filter nodes to top 250 genes
ra_nodes = gene_condition_mean[
    (gene_condition_mean["condition"] == "RA") &
    (gene_condition_mean["gene"].isin(top_250_genes))
    ].copy()
ctrl_nodes = gene_condition_mean[
    (gene_condition_mean["condition"] == "Control") &
    (gene_condition_mean["gene"].isin(top_250_genes))
    ].copy()

# Optional: rename saliency column for consistency
#ra_nodes = ra_nodes.rename(columns={"saliency": "mean_attention"})
#ctrl_nodes = ctrl_nodes.rename(columns={"saliency": "mean_attention"})

print(f"RA nodes (top 250): {len(ra_nodes)}")
print(f"Control nodes (top 250): {len(ctrl_nodes)}")

# ── 2. Aggregate edges — gene-gene only, limited to pairs in top 250 ──
all_edges = []

for patient in patients:
    condition = patient_condition(patient)
    if condition not in ["RA", "Control"]:
        continue
    #
    df2 = pd.read_csv(os.path.join(att_dir, f"{patient}_edge_att.csv"), delimiter=",").iloc[:, 1:]
    df2.columns = ["src", "tgt", "att"]
    # Keep only gene-gene edges
    df2 = df2[(df2["src"] < n_genes) & (df2["tgt"] < n_genes)].copy()
    # Critical filter: only edges between top 250 genes
    df2 = df2[
        df2["src"].isin(top_250_indices) &
        df2["tgt"].isin(top_250_indices)
        ].copy()
    df2["patient"] = patient
    df2["condition"] = condition
    all_edges.append(df2)

all_edges = pd.concat(all_edges, ignore_index=True)

# Mean attention per edge per condition
edge_condition_mean = all_edges.groupby(["condition", "src", "tgt"], as_index=False)["att"].mean()

ra_edges = edge_condition_mean.query("condition == 'RA'")
ctrl_edges = edge_condition_mean.query("condition == 'Control'")

# Remove self-loops
ra_edges_noself = ra_edges[ra_edges["src"] != ra_edges["tgt"]].copy()
ctrl_edges_noself = ctrl_edges[ctrl_edges["src"] != ctrl_edges["tgt"]].copy()

print(f"RA edges (top 250 genes): {len(ra_edges_noself)}")
print(f"Control edges (top 250 genes): {len(ctrl_edges_noself)}")

# ── 3. Optional Cumulative Plot (adapted to top 250) ──
# Sum saliency across top 250 genes (from celltype attention)
top250_saliency = all_genes[all_genes["gene"].isin(top_250_genes)].groupby("gene")["saliency"].sum()
top250_sum_df = pd.DataFrame({
    "gene": top250_saliency.index,
    "saliency": top250_saliency.values
}).sort_values("saliency", ascending=True).reset_index(drop=True)

top250_sum_df["cum_sum"] = top250_sum_df["saliency"].cumsum()
total_top250 = top250_sum_df["saliency"].sum()
top250_sum_df["cum_frac"] = top250_sum_df["cum_sum"] / total_top250

# Plot cumulative for top 250
plt.figure(figsize=(7, 5))
plt.plot(top250_sum_df.index + 1, top250_sum_df["cum_frac"], linewidth=2)
plt.axhline(0.95, linestyle="--", color="gray")
plt.xlabel("Number of top genes (sorted by importance)")
plt.ylabel("Cumulative fraction of total attention (top 250)")
plt.title("Cumulative Attention in Top 250 Genes from Top 5 Cell Types")
plt.tight_layout()
cum_pdf = os.path.join(plots_dir, "FloREN_Top250_cumsum.pdf")
plt.savefig(cum_pdf, format="pdf")
plt.close()

# ── 4. Save files (limited to top 250) ──
ra_edges_noself.to_csv("/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/plots/ra_edges_noself_top250.csv")
ctrl_edges_noself.to_csv("/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/plots/ctrl_edges_noself_top250.csv")
ra_nodes.to_csv("/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/plots/ra_nodes_top250.csv")
ctrl_nodes.to_csv("/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/plots/ctrl_nodes_top250.csv")

print("GRN files saved (limited to top 250 genes from top 5 cell types)")

import pandas as pd
import igraph as ig
from collections import defaultdict
import mygene

# Load data (top 250 filtered)
ra_nodes = pd.read_csv("/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/plots/ra_nodes_top250.csv").iloc[:, 1:]
ctrl_nodes = pd.read_csv("/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/plots/ctrl_nodes_top250.csv").iloc[:, 1:]
ra_edges_noself = pd.read_csv("/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/plots/ra_edges_noself_top250.csv").iloc[
                  :, 1:]
ctrl_edges_noself = pd.read_csv(
    "/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/plots/ctrl_edges_noself_top250.csv").iloc[:, 1:]

# Rename saliency column if needed (from mean_attention)
if 'mean_attention' in ra_nodes.columns:
    ra_nodes = ra_nodes.rename(columns={'mean_attention': 'saliency'})
    ctrl_nodes = ctrl_nodes.rename(columns={'mean_attention': 'saliency'})

# Assume 'gene' column contains symbols (from previous mapping)
# If any are still Ensembl, convert them
ensembl_list = [g for g in set(ra_nodes['gene'].tolist() + ctrl_nodes['gene'].tolist()) if
                isinstance(g, str) and g.startswith('ENSG')]

ensembl_to_symbol = {}
if ensembl_list:
    mg = mygene.MyGeneInfo()
    results = mg.querymany(ensembl_list, scopes='ensembl.gene', fields='symbol', species='human')
    for r in results:
        if 'notfound' not in r:
            ens = r['query']
            sym = r.get('symbol', ens)
            ensembl_to_symbol[ens] = sym
    print(f"Converted {len(ensembl_to_symbol)} remaining Ensembl IDs to symbols")
else:
    print("All genes appear to be symbols already")

# Apply mapping to gene column (safe: symbols stay as is)
ra_nodes['gene'] = ra_nodes['gene'].map(lambda g: ensembl_to_symbol.get(g, g))
ctrl_nodes['gene'] = ctrl_nodes['gene'].map(lambda g: ensembl_to_symbol.get(g, g))

# Create local index maps: gene symbol → local index (0 to len-1)
ra_gene_to_local = {row['gene']: i for i, row in ra_nodes.iterrows()}
ctrl_gene_to_local = {row['gene']: i for i, row in ctrl_nodes.iterrows()}


# Remap edges: convert original src/tgt (0-1999) to local indices (0-249) based on gene symbols
def remap_edges(edges_df, nodes_df, gene_to_local):
    edges_remapped = []
    for _, row in edges_df.iterrows():
        # Get genes from original indices (using nodes_df)
        src_idx = int(row['src'])
        tgt_idx = int(row['tgt'])
        if src_idx >= len(nodes_df) or tgt_idx >= len(nodes_df):
            continue
        src_gene = nodes_df.iloc[src_idx]['gene']
        tgt_gene = nodes_df.iloc[tgt_idx]['gene']
        src_local = gene_to_local.get(src_gene, -1)
        tgt_local = gene_to_local.get(tgt_gene, -1)
        if src_local != -1 and tgt_local != -1:
            edges_remapped.append({
                'src': src_local,
                'tgt': tgt_local,
                'att': row['att']
            })
    return pd.DataFrame(edges_remapped)


ra_edges_local = remap_edges(ra_edges_noself, ra_nodes, ra_gene_to_local)
ctrl_edges_local = remap_edges(ctrl_edges_noself, ctrl_nodes, ctrl_gene_to_local)

print(f"RA remapped edges: {len(ra_edges_local)}")
print(f"Control remapped edges: {len(ctrl_edges_local)}")

# Gene map: local index → gene symbol
ra_gene_map = {i: gene for i, gene in enumerate(ra_nodes['gene'])}
ctrl_gene_map = {i: gene for i, gene in enumerate(ctrl_nodes['gene'])}


def build_and_detect_infomap(edges_df_local, nodes_df, gene_map, condition):
    n_vertices = len(nodes_df)
    if n_vertices == 0:
        print(f"No nodes for {condition}")
        return []
    #
    G = ig.Graph(directed=True)
    G.add_vertices(n_vertices)
    #
    G.vs['saliency'] = nodes_df['saliency'].tolist()
    G.vs['gene'] = nodes_df['gene'].tolist()
    #
    edges = list(zip(edges_df_local['src'], edges_df_local['tgt']))
    weights = edges_df_local['att'].tolist()
    G.add_edges(edges)
    G.es['weight'] = weights
    #
    communities = G.community_infomap(edge_weights='weight', trials=10)
    #
    modules = defaultdict(list)
    for node_idx, mod_id in enumerate(communities.membership):
        gene = gene_map.get(node_idx, f"Unknown_{node_idx}")
        sal = G.vs[node_idx]['saliency']
        modules[mod_id].append((gene, sal))
    #
    sorted_modules = sorted(modules.items(), key=lambda x: len(x[1]), reverse=True)
    #
    print(f"\nInfomap communities for {condition} (top 250 genes):")
    for mod_id, gene_list in sorted_modules:
        print(f"Module {mod_id} (size: {len(gene_list)}):")
        for gene, sal in sorted(gene_list, key=lambda x: x[1], reverse=True):
            print(f"  - {gene}: {sal:.4f}")
    #
    return sorted_modules


# Run
ra_modules = build_and_detect_infomap(ra_edges_local, ra_nodes, ra_gene_map, 'RA')
ctrl_modules = build_and_detect_infomap(ctrl_edges_local, ctrl_nodes, ctrl_gene_map, 'Control')


# Jaccard similarity example
def module_jaccard(mod1, mod2):
    set1 = set([g for g, s in mod1])
    set2 = set([g for g, s in mod2])
    return len(set1 & set2) / len(set1 | set2) if len(set1 | set2) > 0 else 0

print("\nTop RA module vs all CTRL modules Jaccard:")
if ra_modules:
    top_ra = ra_modules[0][1]
    for i, (mod_id, ctrl_mod) in enumerate(ctrl_modules):
        sim = module_jaccard(top_ra, ctrl_mod)
        print(f"RA top vs CTRL {mod_id}: {sim:.4f}")
else:
    print("No RA modules detected")

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# ── Assuming you already ran the previous blocks and have: ──
# ra_nodes, ctrl_nodes, ra_edges_local, ctrl_edges_local
# ra_modules, ctrl_modules (lists of sorted modules from Infomap)

# ── 1. Create module mappings: symbol → module ID ──
ra_symbol_to_module = {}
for mod_id, gene_list in ra_modules:
    for gene, _ in gene_list:
        ra_symbol_to_module[gene] = mod_id

ctrl_symbol_to_module = {}
for mod_id, gene_list in ctrl_modules:
    for gene, _ in gene_list:
        ctrl_symbol_to_module[gene] = mod_id


# ── 2. Build full NetworkX graphs ──
def build_full_graph(nodes_df, edges_df, symbol_to_module):
    G = nx.DiGraph()
    # Add nodes with saliency and module
    for i, row in nodes_df.iterrows():
        symbol = row['gene']
        mod_id = symbol_to_module.get(symbol, -1)
        G.add_node(symbol, saliency=row['saliency'], module=mod_id)
    #
    # Add all edges
    for _, row in edges_df.iterrows():
        src_idx = int(row['src'])
        tgt_idx = int(row['tgt'])
        if src_idx < len(nodes_df) and tgt_idx < len(nodes_df):
            src_gene = nodes_df.iloc[src_idx]['gene']
            tgt_gene = nodes_df.iloc[tgt_idx]['gene']
            G.add_edge(src_gene, tgt_gene, weight=row['att'])
    #
    return G


G_ra_full = build_full_graph(ra_nodes, ra_edges_local, ra_symbol_to_module)
G_ctrl_full = build_full_graph(ctrl_nodes, ctrl_edges_local, ctrl_symbol_to_module)

print(f"RA full graph: {G_ra_full.number_of_nodes()} nodes, {G_ra_full.number_of_edges()} edges")
print(f"Control full graph: {G_ctrl_full.number_of_nodes()} nodes, {G_ctrl_full.number_of_edges()} edges")


# ── 3. Filter out small modules (≤ 3 nodes) ──
def filter_small_modules(G, min_size=4):
    if len(G.nodes()) == 0:
        return G
    #
    # Count size per module
    module_sizes = {}
    for node in G.nodes():
        mod = G.nodes[node]['module']
        if mod not in module_sizes:
            module_sizes[mod] = 0
        module_sizes[mod] += 1
    #
    # Keep only nodes in modules with size >= 4
    kept_nodes = set()
    for node in G.nodes():
        mod = G.nodes[node]['module']
        if module_sizes.get(mod, 0) >= min_size:
            kept_nodes.add(node)
    #
    return G.subgraph(kept_nodes).copy()

min_size=4
G_ra = filter_small_modules(G_ra_full, min_size=min_size)
G_ctrl = filter_small_modules(G_ctrl_full, min_size=min_size)

print(f"After filtering small modules:")
print(f"RA: {G_ra.number_of_nodes()} nodes, {G_ra.number_of_edges()} edges")
print(f"Control: {G_ctrl.number_of_nodes()} nodes, {G_ctrl.number_of_edges()} edges")

# ── 4. Plotting ──
# Prepare colors (normalize across all modules from both graphs)
all_modules = set(ra_symbol_to_module.values()) | set(ctrl_symbol_to_module.values())
if not all_modules:
    all_modules = {-1}  # fallback if no modules

module_norm = mcolors.Normalize(vmin=min(all_modules), vmax=max(all_modules))
cmap = cm.get_cmap('tab20')  # good for up to ~20 modules
#cmap = cm.get_cmap('tab20b')  # good for up to ~20 modules
#cmap = cm.get_cmap('Set1')
#cmap = cm.get_cmap('Paired')
#cmap = cm.get_cmap('Dark2')
#cmap = cm.get_cmap('Accent')

def plot_graph_clean(G, ax, title, layout_name='kamada_kawai'):
    if len(G.nodes()) == 0:
        ax.text(0.5, 0.5, 'No nodes after filtering', ha='center', va='center')
        ax.set_title(title)
        ax.axis('off')
        return
    #
    # Layout
    if layout_name == 'spring':
        pos = nx.spring_layout(G, seed=42, k=0.2, iterations=150)
    elif layout_name == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    elif layout_name == 'fruchterman_reingold':
        pos = nx.fruchterman_reingold_layout(G, seed=42, k=0.25)
    else:
        pos = nx.spring_layout(G, seed=42)
    #
    # Node sizes
    #node_sizes = [max(40, min(1200, G.nodes[n]['saliency'] * 180)) for n in G.nodes()]
    node_sizes = [max(40, min(900, G.nodes[n]['saliency'] * 140)) for n in G.nodes()]
    #
    # Node colors by module
    node_colors = [cmap(module_norm(G.nodes[n]['module'])) for n in G.nodes()]
    #
    # Edge widths (thinner for all edges)
    edge_widths = [max(0.4, G[u][v]['weight'] * 4.5) for u, v in G.edges()]
    #
    # Draw
    nx.draw_networkx_nodes(G, pos,
                           node_size=node_sizes,
                           node_color=node_colors,
                           ax=ax,
                           edgecolors='gray',
                           linewidths=0.4,
                           alpha=0.95)
    nx.draw_networkx_edges(G, pos,
                           width=edge_widths,
                           arrows=True,
                           ax=ax,
                           alpha=0.55,
                           arrowsize=8,
                           connectionstyle='arc3,rad=0.08')
    nx.draw_networkx_labels(G, pos,
                            labels={n: n for n in G.nodes()},
                            font_size=5.8,
                            font_weight='normal',
                            ax=ax,
                            font_family='Arial')
    ax.set_title(f"{title} – All edges – Modules ≥ 4 genes – Layout: {layout_name}", fontsize=14, pad=15)
    ax.axis('off')


# ── Choose layout ──
layout_choice = 'kamada_kawai'

# ── Save ──
output_path = f"/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/plots/gene_graphs_top250_all_edges_filtered_{layout_choice}_v2.pdf"

with PdfPages(output_path) as pdf:
    fig, axs = plt.subplots(1, 2, figsize=(26, 13))
    plot_graph_clean(G_ra, axs[0], 'RA Graph (top 250 genes, all edges, modules ≥ 4)', layout_choice)
    plot_graph_clean(G_ctrl, axs[1], 'Control Graph (top 250 genes, all edges, modules ≥ 4)', layout_choice)
    plt.tight_layout()
    pdf.savefig(fig, dpi=180)
    plt.close(fig)

print(f"PDF saved: {output_path}")
print(f"RA after filtering: {G_ra.number_of_nodes()} nodes, {G_ra.number_of_edges()} edges")
print(f"Control after filtering: {G_ctrl.number_of_nodes()} nodes, {G_ctrl.number_of_edges()} edges")

# ── 1. Helper: Get module → list of (symbol, saliency) for each condition ──
def get_modules_from_graph(G, condition_prefix):
    module_dict = defaultdict(list)
    for node in G.nodes():
        mod_id = G.nodes[node]['module']
        saliency = G.nodes[node]['saliency']
        module_dict[mod_id].append((node, saliency))  # node = symbol here
    #
    # Sort each module by saliency descending
    sorted_modules = {}
    for mod_id, genes in module_dict.items():
        sorted_genes = sorted(genes, key=lambda x: x[1], reverse=True)
        sorted_modules[mod_id] = [gene[0] for gene in sorted_genes]  # only symbols
    #
    # Create nicely named columns
    module_data = {}
    for old_mod_id, genes in sorted(sorted_modules.items(), key=lambda x: x[0]):
        new_col_name = f"{condition_prefix}_GP_{old_mod_id}"
        module_data[new_col_name] = genes
    #
    return module_data

# ── 2. Extract modules from both graphs ──
ra_module_data = get_modules_from_graph(G_ra, "RA")
ctrl_module_data = get_modules_from_graph(G_ctrl, "CTRL")
# ── 3. Combine into one big dictionary ──
all_module_data = {**ra_module_data, **ctrl_module_data}
# ── 4. Create DataFrame ──
# Find the longest module to set the number of rows
max_length = max(len(genes) for genes in all_module_data.values()) if all_module_data else 0
# Create DataFrame with NaN padding
modules_df = pd.DataFrame(
    {col: pd.Series(genes) for col, genes in all_module_data.items()}
)
# Optional: sort columns by condition then by module number (RA first, then CTRL)
ra_cols = sorted([c for c in modules_df.columns if c.startswith("RA_")],
                 key=lambda x: int(x.split('_')[-1]))
ctrl_cols = sorted([c for c in modules_df.columns if c.startswith("CTRL_")],
                   key=lambda x: int(x.split('_')[-1]))
sorted_columns = ra_cols + ctrl_cols
modules_df = modules_df[sorted_columns]
# Optional: make it look nicer by filling NaN with empty string
modules_df = modules_df.fillna("")
print(f"Module DataFrame created with {len(sorted_columns)} columns and {len(modules_df)} rows")
print(modules_df.head(12))  # preview first 12 rows
# ── 5. Save to file (choose your preferred path) ──
output_csv = f"/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/plots/grn_modules_min{min_size}_new.csv"
modules_df.to_csv(output_csv, index=False)
print(f"Saved to: {output_csv}")

import seaborn as sns
# ── 1. Separate RA and CTRL modules ──
ra_cols = [col for col in modules_df.columns if col.startswith('RA_')]
ctrl_cols = [col for col in modules_df.columns if col.startswith('CTRL_')]
# Get gene sets for each module (drop empty/NaN)
ra_module_sets = {col: set(modules_df[col].dropna()) - {''} for col in ra_cols}
ctrl_module_sets = {col: set(modules_df[col].dropna()) - {''} for col in ctrl_cols}
# ── 2. Compute Jaccard Index for every pair ──
def jaccard_index(set1, set2):
    if not set1 and not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0

# Create matrix: rows=RA modules, columns=CTRL modules
ra_labels = sorted(ra_cols, key=lambda x: int(x.split('_')[-1]))
ctrl_labels = sorted(ctrl_cols, key=lambda x: int(x.split('_')[-1]))
jaccard_matrix = np.zeros((len(ra_labels), len(ctrl_labels)))
for i, ra_mod in enumerate(ra_labels):
    for j, ctrl_mod in enumerate(ctrl_labels):
        jaccard_matrix[i, j] = jaccard_index(ra_module_sets[ra_mod], ctrl_module_sets[ctrl_mod])

# ── 3. Create DataFrame for heatmap ──
jaccard_df = pd.DataFrame(jaccard_matrix, index=ra_labels, columns=ctrl_labels)
# ── 4. Plot Heatmap ──
output_pdf = f"/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/plots/jaccard_heatmap_min{min_size}_new.pdf"
with PdfPages(output_pdf) as pdf:
    fig, ax = plt.subplots(figsize=(max(8, len(ctrl_labels) * 0.5), max(6, len(ra_labels) * 0.5)))
    sns.heatmap(jaccard_df,
                annot=True,  # Show values in cells
                fmt=".2f",  # Format to 2 decimals
                vmin= 0,
                vmax= 1,
                cmap="YlGnBu",  # Color map (yellow low → blue high)
                cbar_kws={'label': 'Jaccard Index'},
                linewidths=0.5,  # Grid lines
                ax=ax)
    ax.set_title("Jaccard Similarity Between RA and Control Gene Modules", fontsize=14)
    ax.set_xlabel("Control Modules")
    ax.set_ylabel("RA Modules")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)

print(f"Jaccard heatmap saved to: {output_pdf}")

# Optional: Print top similar pairs (for quick insight)
#print("\nTop 10 similar module pairs (Jaccard > 0):")
#similar_pairs = [(ra, ctrl, val) for ra, ctrl in product(ra_labels, ctrl_labels)
#                 for val in [jaccard_df.at[ra, ctrl]] if val > 0]
#similar_pairs.sort(key=lambda x: x[2], reverse=True)
#for ra, ctrl, jac in similar_pairs[:10]:
#    print(f"{ra} vs {ctrl}: {jac:.3f}")



# ── 1. Get all unique genes across both filtered graphs ──
all_genes = sorted(set(G_ra.nodes()) | set(G_ctrl.nodes()))
# ── 2. Prepare module → genes mapping (already sorted by saliency from previous code) ──
# We'll reuse the structure from earlier module extraction
def get_module_gene_attention(G, modules_dict, condition_prefix):
    module_attention = defaultdict(dict)  # module_id → {gene: avg_attention}
    # For each module, collect all edges where both endpoints are in the same module
    membership = {node: G.nodes[node]['module'] for node in G.nodes()}
    for u, v, data in G.edges(data=True):
        mod_u = membership.get(u, -1)
        mod_v = membership.get(v, -1)
        if mod_u == mod_v and mod_u != -1:
            # Edge inside the same module → contribute to both genes
            att = data['weight']
            if u in module_attention[mod_u]:
                module_attention[mod_u][u].append(att)
            else:
                module_attention[mod_u][u] = [att]
            #
            if v in module_attention[mod_v]:
                module_attention[mod_v][v].append(att)
            else:
                module_attention[mod_v][v] = [att]
    #
    # Compute average attention per gene per module
    module_avg = {}
    for mod_id, gene_atts in module_attention.items():
        col_name = f"{condition_prefix}_GP_{mod_id}"
        module_avg[col_name] = {gene: np.mean(atts) for gene, atts in gene_atts.items()}
    #
    return module_avg


# ── 3. Get attention dictionaries for RA and CTRL ──
ra_module_attention = get_module_gene_attention(G_ra, ra_modules, "RA")
ctrl_module_attention = get_module_gene_attention(G_ctrl, ctrl_modules, "CTRL")
# Combine all programs
all_programs = {}
all_programs.update(ra_module_attention)
all_programs.update(ctrl_module_attention)
# ── 4. Build the final matrix ──
# Initialize empty matrix (programs × genes)
program_names = sorted(all_programs.keys(),
                       key=lambda x: (x.startswith('CTRL'), int(x.split('_')[-1])))
attention_matrix = pd.DataFrame(0.0, index=program_names, columns=all_genes)
# Fill in values
for program in program_names:
    gene_att = all_programs.get(program, {})
    for gene, avg_att in gene_att.items():
        if gene in attention_matrix.columns:
            attention_matrix.at[program, gene] = avg_att

# Optional: round for readability
attention_matrix = attention_matrix.round(4)
print(f"Attention matrix shape: {attention_matrix.shape}")
print(f"Programs: {len(program_names)}")
print(f"Genes: {len(all_genes)}")
print("\nPreview (first 6 programs × first 8 genes):")
print(attention_matrix.iloc[:6, :8])
# ── 5. Save ──
output_path = f"/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/plots/GPs_matrix_att_min{min_size}_new.csv"
attention_matrix.to_csv(output_path)
print(f"\nMatrix saved to: {output_path}")















#-----------------------------------------------------
#
# SUPER SUPER EXTRA: PROGRAMNS VALIDATION T-TEST
#
#-----------------------------------------------------

import glob
import os

data_path = "C:/Users/Inigo/Desktop/FloREN/Binvignat/genes"
#count_matrices_path = os.path.join(data_path, "count_matrices")
#count_matrices_path = "/home/inigo/Desktop/FloREN3.0/Binvignat/genes/"
print(f"Looking for CSV files in: {data_path}")
csv_files = glob.glob(os.path.join(data_path, "*.csv"))
print(f"Found {len(csv_files)} CSV files: {csv_files}")

# Load reference for gene names and concatenate all matrices
reference = pd.read_csv(csv_files[0])
gene_names = reference[reference.columns[0]].values
n_genes = reference.shape[0]
in_dim = 256
if n_genes < in_dim:
    h_n = n_genes
else:
    h_n = in_dim

gene_cell = np.zeros((n_genes, 1))
cells = []
cell_counts = []
for f in csv_files:
    file = pd.read_csv(f)
    patient_name = str.split(str.split(f, 'binvignat_')[1], '.csv')[0]
    cells.append([patient_name + '__' + col for col in file.columns[1:]])
    cell_counts.append(file.shape[1] - 1)  # Number of cells for this patient
    file = file.iloc[:, 1:].to_numpy()
    gene_cell = np.concatenate((gene_cell, file), axis=1)

gene_cell = gene_cell[:, 1:]  # Shape: [n_genes, total_cells]

attention_matrix_df = pd.read_csv(f"/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/plots/GPs_matrix_att_min{min_size}_new.csv")
attention_matrix_df = attention_matrix_df.set_index("Unnamed: 0")
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── 1. Flatten cell names (in order matching gene_cell columns) ──
all_cell_names = [cell for patient_cells in cells for cell in patient_cells]
print(f"Total cells: {len(all_cell_names)}, gene_cell shape: {gene_cell.shape}")
# Sanity check
assert len(all_cell_names) == gene_cell.shape[1], "Cell names count mismatch with gene_cell columns"

# ── 2. Normalize gene_cell (critical!) ──
# Option A: log1p + library size normalization (common for scRNA-seq)
#gene_cell_norm = np.log1p(gene_cell)
# Option B: + per-cell scaling (unit norm or z-score per cell)
#scaler = StandardScaler(with_mean=False)  # or with_mean=True for z-score
#gene_cell_norm = scaler.fit_transform(gene_cell_norm.T).T  # scale genes (rows)
# Or per-cell norm: gene_cell_norm /= np.linalg.norm(gene_cell_norm, axis=0) + 1e-10

# ── 3. Align attention_matrix genes to gene_names order ──
# Assume attention_matrix columns are symbols; map to Ensembl order if needed
# If columns are already in gene_names order, skip; else reindex
mg = mygene.MyGeneInfo()
results = mg.querymany(attention_matrix.columns, scopes='symbol', fields='ensembl.gene', species='human')
symbol_to_ensembl = {}
for r in results:
    if 'notfound' in r:
        continue
    #
    sym = r['query']
    ens = None
    # Case 1: single dict
    if 'ensembl' in r and isinstance(r['ensembl'], dict):
        ens = r['ensembl'].get('gene')
    #
    # Case 2: list of dicts
    elif 'ensembl' in r and isinstance(r['ensembl'], list):
        for item in r['ensembl']:
            if isinstance(item, dict) and 'gene' in item:
                ens = item['gene']
                break  # take first
    #
    # Fallback
    symbol_to_ensembl[sym] = ens if ens else sym

# Report
print(f"Successfully mapped {sum(1 for v in symbol_to_ensembl.values() if v.startswith('ENSG'))} / {len(symbol_to_ensembl)} symbols to Ensembl IDs")
print("First few mappings:")
# Handle any symbols in attention_matrix not in mapping (rare)
new_attention_matrix = pd.DataFrame(
    0.0,
    index=attention_matrix.index,
    columns=gene_names  # 2000 Ensembl IDs
)

for sym in attention_matrix.columns:
    ens = symbol_to_ensembl.get(sym)
    if ens and ens in gene_names:
        new_attention_matrix[ens] = attention_matrix[sym]

print(f"Expanded attention matrix shape: {new_attention_matrix.shape}")
print(f"Non-zero entries: {(new_attention_matrix != 0).sum().sum()} (out of {new_attention_matrix.size})")

# ── 4. Compute activations: cells x GPs ──
# gene_cell_norm.T (cells x genes) @ attention_matrix.T (genes x GPs) = cells x GPs
activations = gene_cell.T @ new_attention_matrix.values.T

# Create DataFrame
gp_names = attention_matrix.index.tolist()  # e.g., ['RA_GP_3', 'RA_GP_4', ...]
activation_df = pd.DataFrame(activations, index=all_cell_names, columns=gp_names)

print(f"Activation matrix shape: {activation_df.shape}")

# Optional: add condition (RA/Control) from patient in cell name
def get_condition(cell_name):
    patient = cell_name.split('__')[0]
    if 'RA' in patient:
        return 'RA'
    elif 'Control' in patient or 'HD' in patient or 'HC' in patient:
        return 'Control'
    else:
        return 'Other'

activation_df['condition'] = [get_condition(cell) for cell in activation_df.index]

# Preview
print(activation_df.head())
print("\nMean activation per GP by condition:")
print(activation_df.groupby('condition').mean())

# Save
activation_df.to_csv("/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/plots/cell_gp_activations.csv")


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.backends.backend_pdf import PdfPages
from statannotations.Annotator import Annotator

# ── 1. Pseudobulk: mean activation per patient ──
# Extract patient ID from cell name
activation_df['patient'] = activation_df.index.map(lambda x: x.split('__')[0])
# Group by patient and condition, take mean per GP
pseudobulk = activation_df.groupby(['patient', 'condition']).mean(numeric_only=True).reset_index()
# Drop any 'Other' if present
pseudobulk = pseudobulk[pseudobulk['condition'].isin(['RA', 'Control'])]
print(f"Pseudobulk shape: {pseudobulk.shape}")
print(pseudobulk.head())

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.backends.backend_pdf import PdfPages
import math

gp_columns = [col for col in pseudobulk.columns if col not in ['patient', 'condition']]
# ── Number of plots & grid layout ──
n_plots = len(gp_columns)
n_cols = 3  # adjust if you prefer 2 or 4
n_rows = math.ceil(n_plots / n_cols)
# Figure size: scale with number of rows
fig_height = max(4, n_rows * 3.5)  # roughly 3.5 inches per row
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, fig_height),
                         sharex=False, sharey=False, squeeze=False)
# Flatten axes for easy iteration
axes_flat = axes.flatten()
# ── Statistical testing & plotting loop ──
for i, gp in enumerate(gp_columns):
    ax = axes_flat[i]
    ra_vals = pseudobulk[pseudobulk['condition'] == 'RA'][gp].dropna()
    ctrl_vals = pseudobulk[pseudobulk['condition'] == 'Control'][gp].dropna()
    if len(ra_vals) < 3 or len(ctrl_vals) < 3:
        ax.text(0.5, 0.5, 'Too few samples', ha='center', va='center')
        ax.axis('off')
        continue
    #
    # Normality
    _, p_ra = stats.shapiro(ra_vals)
    _, p_ctrl = stats.shapiro(ctrl_vals)
    both_normal = (p_ra > 0.05) and (p_ctrl > 0.05)
    # Test
    if both_normal:
        stat, pval = stats.ttest_ind(ra_vals, ctrl_vals, equal_var=False)
        test_name = "t-test"
    else:
        stat, pval = stats.mannwhitneyu(ra_vals, ctrl_vals, alternative='two-sided')
        test_name = "MW U"
    #
    # p-value text with stars
    if pval < 0.001:
        p_text = f'p < 0.001 ***'
    elif pval < 0.01:
        p_text = f'p < 0.01 **'
    elif pval < 0.05:
        p_text = f'p < 0.05 *'
    else:
        p_text = f'p = {pval:.3f} ns'
    #
    # ── Plot ──
    sns.boxplot(
        data=pseudobulk,
        x='condition',
        y=gp,
        palette={'RA': '#d95f02', 'Control': '#1b9e77'},
        width=0.45,
        ax=ax,
        fliersize=0
    )
    sns.stripplot(
        data=pseudobulk,
        x='condition',
        y=gp,
        hue='condition',
        palette={'RA': '#d95f02', 'Control': '#1b9e77'},
        size=5,
        jitter=0.18,
        alpha=0.85,
        ax=ax,
        legend=False
    )
    # p-value bracket
    x1, x2 = 0, 1
    y_max = pseudobulk[gp].max() * 1.12
    ax.plot([x1, x1, x2, x2], [y_max * 0.9, y_max, y_max, y_max * 0.9], lw=1.2, c='k')
    ax.text((x1 + x2) / 2, y_max * 1.03, p_text, ha='center', va='bottom', fontsize=9)
    # Smaller title
    ax.set_title(f"{gp} ({test_name})", fontsize=10, pad=6)
    ax.set_xlabel("") if i // n_cols < n_rows - 1 else ax.set_xlabel("Condition")
    ax.set_ylabel("Mean Activation") if i % n_cols == 0 else ax.set_ylabel("")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=8)

# Hide unused subplots
for j in range(i + 1, len(axes_flat)):
    axes_flat[j].axis('off')

fig.suptitle("GP Activation per Patient (Pseudobulk) – RA vs Control", fontsize=14, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])

# ── Save ──
output_pdf = "/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/plots/gp_activation_pseudobulk_all_in_one.pdf"
fig.savefig(output_pdf, dpi=250, bbox_inches='tight')
plt.close(fig)

print(f"All GPs plotted on one page and saved to: {output_pdf}")














#-----------------------------------------------------
#
# SUPER SUPER EXTRA: PROGRAMNS VALIDATION T-TEST SCANPY
#
#-----------------------------------------------------

import pandas as pd
import numpy as np
import scanpy as sc  # assuming you have Scanpy imported
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


adata = sc.read_h5ad("C:/Users/Inigo/Desktop/FloREN/Binvignat/output.h5ad")
adata.obs["sample_id"] = adata.obs["donor_id"].astype(str) + "__" + adata.obs["disease"].astype(str)
min_size = 4
attention_matrix_df = pd.read_csv(f"/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/plots/GPs_matrix_att_min{min_size}_new.csv")
attention_matrix_df = attention_matrix_df.set_index("Unnamed: 0")

# ── 1. Extract barcode from activation_df index ──
activation_df = pd.read_csv("/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/plots/cell_gp_activations.csv")
#activation_df = activation_df.copy()  # avoid modifying original
activation_df = activation_df.set_index("Unnamed: 0")
#activation_df.rename(columns={activation_df.columns[0]: "cell_full"}, inplace=True)
activation_df['cell_full'] = activation_df.index
#activation_df = activation_df.set_index("cell_full")
activation_df['barcode'] = activation_df['cell_full'].str.split('__').str[-1]

# Set barcode as index
activation_df = activation_df.set_index('barcode')

# Drop helper columns (keep only GP columns + condition if needed)
gp_columns = [col for col in activation_df.columns if col not in ['cell_full', 'condition', 'patient']]
activation_df_aligned = activation_df[gp_columns]

# ── 2. Reindex to exactly match adata.obs_names ──
# This ensures perfect alignment (raises error if mismatch)
# Inner join: only keep cells present in both
activation_df_aligned = activation_df_aligned.loc[adata.obs_names].copy()

print(f"Aligned activations shape: {activation_df_aligned.shape}")
print(f"Number of matching cells: {len(activation_df_aligned)}")
if len(activation_df_aligned) < len(adata):
    print("Warning: Some adata cells missing activations → filled with NaN or 0 below")

# Fill any missing cells with 0 (or NaN)
activation_df_aligned = activation_df_aligned.reindex(adata.obs_names, fill_value=0.0)

# ── 3. Add to adata as .obsm (recommended) ──
adata.obsm['gp_activation'] = activation_df_aligned.values

# Optional: store GP names for reference
adata.uns['gp_names'] = activation_df_aligned.columns.tolist()

## Also add condition/patient to adata.obs if not already there
#if 'condition' not in adata.obs.columns:
#    adata.obs['condition'] = activation_df['condition'].reindex(adata.obs_names, fill_value='Unknown')
#
#if 'patient' not in adata.obs.columns:
#    adata.obs['patient'] = activation_df['patient'].reindex(adata.obs_names, fill_value='Unknown')
#
#print("Added to adata:")
#print("- .obsm['gp_activation']: shape", adata.obsm['gp_activation'].shape)
#print("- .uns['gp_names']: GP list")
#print("- .obs['condition'] and .obs['patient'] updated if missing")
# Quick check: first few rows
#print(adata.obsm['gp_activation'][:5, :5])  # preview values

# ── Assumptions ──
# adata already has:
#   - .obsm['UMAP']
#   - .obsm['gp_activation']  (cells × GPs)
#   - .uns['gp_names']        (list of GP names in order of columns)
#   - .obs['cell_type']
#   - .obs['disease']         (with values like "normal", "rheumatoid arthritis")

gp_names = adata.uns['gp_names']  # e.g. ['RA_GP_3', 'RA_GP_4', ...]
adata.obsm["X_umap"] = np.asarray(adata.obsm["UMAP"])
plots_dir = "/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/plots"
sc._settings.settings._vector_friendly=True
# ── Output PDF ──
for gp_idx, gp_name in enumerate(gp_names):
    print(f"Plotting {gp_name} ({gp_idx + 1}/{len(gp_names)})")
    # Store GP activation in obs
    adata.obs["gp_value"] = adata.obsm["gp_activation"][:, gp_idx]
    # Shared color scale (robust)
    gp_vals = adata.obs["gp_value"].values
    #vmin, vmax = np.percentile(gp_vals, [1, 99])
    vmin = 0
    vmax = gp_vals.max()
    output_pdf = os.path.join(
        plots_dir,
        f"GP_UMAP_{gp_name}.pdf"
    )
    with PdfPages(output_pdf) as pdf:
        fig, axs = plt.subplots(
            1, 3, figsize=(15, 5),
            gridspec_kw={"wspace": 0.35}
        )
        # ── 1. All cells – cell types ─────────────────────────
        sc._settings.settings._vector_friendly = True
        sc.pl.umap(
            adata,
            color="cell_type",
            ax=axs[0],
            show=False,
            frameon=False,
            size=15,
            legend_loc="right margin",
            legend_fontsize=2,
            legend_fontoutline=0,
            title="All cells – Cell types"
        )
        # ── 2. Normal cells ───────────────────────────────────
        normal_mask = adata.obs["disease"] == "normal"
        if normal_mask.sum() == 0:
            axs[1].text(0.5, 0.5, "No normal cells",
                        ha="center", va="center")
            axs[1].axis("off")
        else:
            sc._settings.settings._vector_friendly = True
            sc.pl.umap(
                adata[normal_mask],
                color="gp_value",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                ax=axs[1],
                show=False,
                frameon=False,
                size=25,
                title=f"Normal – {gp_name}"
            )
        #
        # ── 3. RA cells ───────────────────────────────────────
        ra_mask = adata.obs["disease"] == "rheumatoid arthritis"
        if ra_mask.sum() == 0:
            axs[2].text(0.5, 0.5, "No RA cells",
                        ha="center", va="center")
            axs[2].axis("off")
        else:
            sc._settings.settings._vector_friendly = True
            sc.pl.umap(
                adata[ra_mask],
                color="gp_value",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                ax=axs[2],
                show=False,
                frameon=False,
                size=25,
                title=f"RA – {gp_name}"
            )
        #
        fig.suptitle(
            f"Gene Program: {gp_name}",
            fontsize=14,
            y=0.98,
            weight="bold"
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, dpi=200)
        plt.close(fig)

#------------------------------------
# plot all together
#------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
from matplotlib.backends.backend_pdf import PdfPages

plots_dir = "/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/plots"
output_pdf = os.path.join(plots_dir, "RA_GPs_UMAP_12x3.pdf")

# Vector-friendly Scanpy
sc._settings.settings._vector_friendly = True

# Select the 12 RA GPs explicitly
ra_gp_names = [g for g in gp_names if g.startswith("RA_GP")][:12]
n_gps = len(ra_gp_names)

# Masks + subsets
normal_mask = adata.obs["disease"] == "normal"
ra_mask = adata.obs["disease"] == "rheumatoid arthritis"

adata_normal = adata[normal_mask].copy()
adata_ra = adata[ra_mask].copy()

with PdfPages(output_pdf) as pdf:
    fig, axs = plt.subplots(
        nrows=n_gps,
        ncols=3,
        figsize=(12, 2.6 * n_gps),
        gridspec_kw={"wspace": 0.25, "hspace": 0.25}
    )
    for row, gp_name in enumerate(ra_gp_names):
        gp_idx = gp_names.index(gp_name)
        print(f"Plotting {gp_name} ({row + 1}/{n_gps})")
        # Add GP values
        adata.obs["gp_value"] = adata.obsm["gp_activation"][:, gp_idx]
        adata_normal.obs["gp_value"] = adata.obs.loc[normal_mask, "gp_value"].values
        adata_ra.obs["gp_value"] = adata.obs.loc[ra_mask, "gp_value"].values
        # Shared scale (Normal + RA)
        gp_vals = np.concatenate([
            adata_normal.obs["gp_value"].values,
            adata_ra.obs["gp_value"].values
        ])
        vmin = 0
        #vmax = np.percentile(gp_vals, 99)
        vmax = np.max(gp_vals)
        # ── Column 1: All cells ───────────────────────────────
        sc.pl.umap(
            adata,
            color="cell_type",
            ax=axs[row, 0],
            show=False,
            frameon=False,
            size=10,
            legend_loc="right margin" if row == 0 else None,
            legend_fontsize=5,
            legend_fontoutline=0
        )
        axs[row, 0].set_title(
            "All cells – Cell types" if row == 0 else "",
            fontsize=10
        )
        # ── Column 2: Normal ──────────────────────────────────
        sc.pl.umap(
            adata_normal,
            color="gp_value",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            ax=axs[row, 1],
            show=False,
            frameon=False,
            size=10
        )
        axs[row, 1].set_title(
            "Normal" if row == 0 else "",
            fontsize=10
        )
        # ── Column 3: RA ──────────────────────────────────────
        sc.pl.umap(
            adata_ra,
            color="gp_value",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            ax=axs[row, 2],
            show=False,
            frameon=False,
            size=10
        )
        axs[row, 2].set_title(
            "Rheumatoid arthritis" if row == 0 else "",
            fontsize=10
        )
        # Row label (GP name)
        axs[row, 0].set_ylabel(gp_name, fontsize=9, rotation=0, labelpad=45, va="center")
        # Rasterize points only
        for col in range(3):
            for coll in axs[row, col].collections:
                coll.set_rasterized(True)
    #
    fig.suptitle(
        "RA Gene Programs — UMAP overview",
        fontsize=16,
        y=0.995,
        weight="bold"
    )
    pdf.savefig(fig, dpi=200)
    plt.close(fig)

print(f"Saved: {output_pdf}")

#------------------------------------
# plot top 5 subtypes
#------------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
from matplotlib.backends.backend_pdf import PdfPages
# ── Subset to specific cell types ──
target_cell_types = [
    "plasmablast",
    "gamma-delta T cell",
    "effector memory CD4-positive, alpha-beta T cell",
    "natural killer cell",
    "central memory CD4-positive, alpha-beta T cell"
]
# Subset adata
subset_mask = adata.obs["cell_type"].isin(target_cell_types)
adata_subset = adata[subset_mask].copy()
print(f"Subset shape: {adata_subset.shape} (from original {adata.shape})")
# ── Select the 12 RA GPs explicitly ──
ra_gp_names = [g for g in gp_names if g.startswith("RA_GP")][:12]
n_gps = len(ra_gp_names)
# ── Masks + subsets on the subset adata ──
normal_mask = adata_subset.obs["disease"] == "normal"
ra_mask = adata_subset.obs["disease"] == "rheumatoid arthritis"
adata_subset_normal = adata_subset[normal_mask].copy()
adata_subset_ra = adata_subset[ra_mask].copy()
# ── Output PDF ──
output_pdf = os.path.join(plots_dir, "RA_GPs_UMAP_12x3_subset_celltypes.pdf")
with PdfPages(output_pdf) as pdf:
    fig, axs = plt.subplots(
        nrows=n_gps,
        ncols=3,
        figsize=(12, 2.6 * n_gps),
        gridspec_kw={"wspace": 0.25, "hspace": 0.25}
    )
    for row, gp_name in enumerate(ra_gp_names):
        gp_idx = gp_names.index(gp_name)
        print(f"Plotting {gp_name} ({row + 1}/{n_gps})")
        # Add GP values to obs (on subset)
        adata_subset.obs["gp_value"] = adata_subset.obsm["gp_activation"][:, gp_idx]
        adata_subset_normal.obs["gp_value"] = adata_subset.obs.loc[normal_mask, "gp_value"].values
        adata_subset_ra.obs["gp_value"] = adata_subset.obs.loc[ra_mask, "gp_value"].values
        # Shared scale (Normal + RA in subset)
        gp_vals = np.concatenate([
            adata_subset_normal.obs["gp_value"].values,
            adata_subset_ra.obs["gp_value"].values
        ])
        vmin = 0
        vmax = np.max(gp_vals)
        # ── Column 1: All subset cells – Cell types ──
        sc.pl.umap(
            adata_subset,
            color="cell_type",
            ax=axs[row, 0],
            show=False,
            frameon=False,
            size=10,
            legend_loc="right margin" if row == 0 else None,
            legend_fontsize=5,
            legend_fontoutline=0
        )
        axs[row, 0].set_title(
            "All cells – Cell types" if row == 0 else "",
            fontsize=10
        )
        # ── Column 2: Normal (subset) ──
        sc.pl.umap(
            adata_subset_normal,
            color="gp_value",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            ax=axs[row, 1],
            show=False,
            frameon=False,
            size=10
        )
        axs[row, 1].set_title(
            "Normal" if row == 0 else "",
            fontsize=10
        )
        # ── Column 3: RA (subset) ──
        sc.pl.umap(
            adata_subset_ra,
            color="gp_value",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            ax=axs[row, 2],
            show=False,
            frameon=False,
            size=10
        )
        axs[row, 2].set_title(
            "Rheumatoid arthritis" if row == 0 else "",
            fontsize=10
        )
        # Row label (GP name)
        axs[row, 0].set_ylabel(gp_name, fontsize=9, rotation=0, labelpad=45, va="center")
        # Rasterize points only
        for col in range(3):
            for coll in axs[row, col].collections:
                coll.set_rasterized(True)
    #
    fig.suptitle(
        "RA Gene Programs — UMAP overview (subset cell types)",
        fontsize=16,
        y=0.995,
        weight="bold"
    )
    pdf.savefig(fig, dpi=200)
    plt.close(fig)

#------------------------------------
# DEG 5 SUBTYPES
#------------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
from matplotlib.backends.backend_pdf import PdfPages

plots_dir = "/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/plots"
output_pdf = os.path.join(plots_dir, "RA_GPs_UMAP_12x3_all_and_per_celltype.pdf")

sc._settings.settings._vector_friendly = True

# Target cell types (in order for consistent pages)
target_celltypes = [
    "plasmablast",
    "gamma-delta T cell",
    "effector memory CD4-positive, alpha-beta T cell",
    "natural killer cell",
    "central memory CD4-positive, alpha-beta T cell"
]

# Subset adata to these cell types (done once)
subset_mask = adata.obs["cell_type"].isin(target_celltypes)
adata_subset = adata[subset_mask].copy()
print(f"Subset shape (all 5 types): {adata_subset.shape}")

# Select 12 RA GPs
ra_gp_names = [g for g in gp_names if g.startswith("RA_GP")][:12]
n_gps = len(ra_gp_names)

with PdfPages(output_pdf) as pdf:
    # ── PAGE 1: All 5 cell types together ──
    print("Generating page 1/6: all cell types")
    fig, axs = plt.subplots(
        nrows=n_gps,
        ncols=3,
        figsize=(12, 2.6 * n_gps),
        gridspec_kw={"wspace": 0.25, "hspace": 0.25}
    )
    for row, gp_name in enumerate(ra_gp_names):
        gp_idx = gp_names.index(gp_name)
        # Fresh masks
        normal_mask = adata_subset.obs["disease"] == "normal"
        ra_mask = adata_subset.obs["disease"] == "rheumatoid arthritis"
        adata_normal = adata_subset[normal_mask].copy()
        adata_ra = adata_subset[ra_mask].copy()
        # Add GP value
        adata_normal.obs["gp_value"] = adata_subset.obsm["gp_activation"][normal_mask, gp_idx]
        adata_ra.obs["gp_value"] = adata_subset.obsm["gp_activation"][ra_mask, gp_idx]
        # Shared scale
        gp_vals = np.concatenate([
            adata_normal.obs["gp_value"].values,
            adata_ra.obs["gp_value"].values
        ])
        vmin = 0
        vmax = np.max(gp_vals) if len(gp_vals) > 0 else 1
        # Column 1: All cells – cell types
        sc.pl.umap(
            adata_subset,
            color="cell_type",
            ax=axs[row, 0],
            show=False,
            frameon=False,
            size=10,
            legend_loc="right margin" if row == 0 else None,
            legend_fontsize=5,
            legend_fontoutline=0
        )
        axs[row, 0].set_title("All cells – Cell types" if row == 0 else "", fontsize=10)
        # Column 2: Normal
        sc.pl.umap(
            adata_normal,
            color="gp_value",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            ax=axs[row, 1],
            show=False,
            frameon=False,
            size=10
        )
        axs[row, 1].set_title("Normal" if row == 0 else "", fontsize=10)
        # Column 3: RA
        sc.pl.umap(
            adata_ra,
            color="gp_value",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            ax=axs[row, 2],
            show=False,
            frameon=False,
            size=10
        )
        axs[row, 2].set_title("Rheumatoid arthritis" if row == 0 else "", fontsize=10)
        axs[row, 0].set_ylabel(gp_name, fontsize=9, rotation=0, labelpad=45, va="center")
        for col in range(3):
            for coll in axs[row, col].collections:
                coll.set_rasterized(True)
    #
    fig.suptitle("RA Gene Programs — All 5 cell types", fontsize=16, y=0.995, weight="bold")
    pdf.savefig(fig, dpi=200, bbox_inches='tight')
    plt.close(fig)
    # ── PAGES 2–6: One cell type per page ──
    for ct_idx, ct in enumerate(target_celltypes, start=2):
        print(f"Generating page {ct_idx}/6: {ct}")
        # Subset to only this cell type
        ct_mask = adata_subset.obs["cell_type"] == ct
        adata_ct = adata_subset[ct_mask].copy()
        if adata_ct.n_obs == 0:
            print(f"  Skipping {ct}: no cells")
            continue
        #
        fig, axs = plt.subplots(
            nrows=n_gps,
            ncols=3,
            figsize=(12, 2.6 * n_gps),
            gridspec_kw={"wspace": 0.25, "hspace": 0.25}
        )
        for row, gp_name in enumerate(ra_gp_names):
            gp_idx = gp_names.index(gp_name)
            # Fresh masks for this cell-type subset
            normal_mask_ct = adata_ct.obs["disease"] == "normal"
            ra_mask_ct = adata_ct.obs["disease"] == "rheumatoid arthritis"
            adata_normal_ct = adata_ct[normal_mask_ct].copy()
            adata_ra_ct = adata_ct[ra_mask_ct].copy()
            # Add GP value
            adata_normal_ct.obs["gp_value"] = adata_ct.obsm["gp_activation"][normal_mask_ct, gp_idx]
            adata_ra_ct.obs["gp_value"] = adata_ct.obsm["gp_activation"][ra_mask_ct, gp_idx]
            # Shared scale for this cell type
            gp_vals_ct = np.concatenate([
                adata_normal_ct.obs["gp_value"].values,
                adata_ra_ct.obs["gp_value"].values
            ])
            vmin_ct = 0
            vmax_ct = np.max(gp_vals_ct) if len(gp_vals_ct) > 0 else 1
            # Column 1: All cells of this type – cell type label
            sc.pl.umap(
                adata_ct,
                color="cell_type",  # will be constant, but shows the type
                ax=axs[row, 0],
                show=False,
                frameon=False,
                size=10,
                legend_loc=None  # no legend needed
            )
            axs[row, 0].set_title("All cells" if row == 0 else "", fontsize=10)
            # Column 2: Normal of this type
            sc.pl.umap(
                adata_normal_ct,
                color="gp_value",
                cmap="viridis",
                vmin=vmin_ct,
                vmax=vmax_ct,
                ax=axs[row, 1],
                show=False,
                frameon=False,
                size=10
            )
            axs[row, 1].set_title("Normal" if row == 0 else "", fontsize=10)
            # Column 3: RA of this type
            sc.pl.umap(
                adata_ra_ct,
                color="gp_value",
                cmap="viridis",
                vmin=vmin_ct,
                vmax=vmax_ct,
                ax=axs[row, 2],
                show=False,
                frameon=False,
                size=10
            )
            axs[row, 2].set_title("Rheumatoid arthritis" if row == 0 else "", fontsize=10)
            # Row label
            axs[row, 0].set_ylabel(gp_name, fontsize=9, rotation=0, labelpad=45, va="center")
            for col in range(3):
                for coll in axs[row, col].collections:
                    coll.set_rasterized(True)
        #
        fig.suptitle(f"RA Gene Programs — {ct}", fontsize=16, y=0.995, weight="bold")
        pdf.savefig(fig, dpi=200, bbox_inches='tight')
        plt.close(fig)

print(f"PDF with 6 pages saved to: {output_pdf}")

#------------------------------------
# DEG GPs 5 SUBTYPES
#------------------------------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import mannwhitneyu

# Assume adata_subset is already created (from previous code)
# ra_gp_names = the 12 RA GPs
ra_gp_names = [g for g in gp_names if g.startswith("RA_GP")][:12]
ra_gp_indices = [gp_names.index(gp) for gp in ra_gp_names]
# Slice only RA columns from the activation array
ra_activation = adata_subset.obsm['gp_activation'][:, ra_gp_indices]

# Target cell types (for rows 2–6)
target_cell_types = [
    "plasmablast",
    "gamma-delta T cell",
    "effector memory CD4-positive, alpha-beta T cell",
    "natural killer cell",
    "central memory CD4-positive, alpha-beta T cell"
]

# ── Pseudobulk: mean GP activation per patient per cell type ──
gp_df = pd.DataFrame(
    ra_activation,
    index=adata_subset.obs_names,
    columns=ra_gp_names  # only RA GPs
)

gp_df['patient'] = adata_subset.obs['donor_id']
gp_df['cell_type'] = adata_subset.obs['cell_type']
gp_df['condition'] = adata_subset.obs['disease'].map({
    'normal': 'HD',
    'rheumatoid arthritis': 'RA'
})  # assume this mapping; adjust if needed

# Group by patient + cell_type for means
pseudobulk_ct = gp_df.groupby(['patient', 'cell_type', 'condition']).mean().reset_index()


# Group by patient for "all" means (across all 5 types)
pseudobulk_all = gp_df.drop(columns=['cell_type']).groupby(['patient', 'condition']).mean().reset_index()
pseudobulk_all['cell_type'] = 'All types'

# Combine both
pseudobulk = pd.concat([pseudobulk_all, pseudobulk_ct], ignore_index=True)

# ── Plot 6x12 grid ──
plots_dir = "/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/plots"
output_pdf = os.path.join(plots_dir, "pseudobulk_gp_activation_6x12.pdf")

with PdfPages(output_pdf) as pdf:
    fig, axs = plt.subplots(
        nrows=6,  # 1 all + 5 individual
        ncols=12,  # 12 GPs
        figsize=(24, 12),  # large for 6x12; adjust as needed
        sharey='row',  # share y per row
        gridspec_kw={'wspace': 0.3, 'hspace': 0.4}
    )
    row_titles = ['All types'] + target_cell_types
    for row_idx, ct in enumerate(row_titles):
        data_row = pseudobulk[pseudobulk['cell_type'] == ct]
        for col_idx, gp in enumerate(ra_gp_names):
            ax = axs[row_idx, col_idx]
            # RA/HD values
            ra_vals = data_row[data_row['condition'] == 'RA'][gp]
            ra_vals = ra_vals.dropna()
            hd_vals = data_row[data_row['condition'] == 'HD'][gp]
            hd_vals = hd_vals.dropna()
            # Boxplot
            sns.boxplot(
                data=data_row,
                x='condition',
                y=gp,
                palette={'RA': '#d95f02', 'HD': '#1b9e77'},
                width=0.45,
                ax=ax,
                fliersize=0
            )
            # Scatter
            sns.stripplot(
                data=data_row,
                x='condition',
                y=gp,
                hue='condition',
                palette={'RA': '#d95f02', 'HD': '#1b9e77'},
                size=6,
                jitter=0.18,
                alpha=0.85,
                ax=ax,
                legend=False
            )
            # Wilcoxon test
            if len(ra_vals) > 1 and len(hd_vals) > 1:  # need ≥2 per group for test
                stat, pval = mannwhitneyu(ra_vals, hd_vals, alternative='two-sided')
                sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
                p_text = f'p={pval:.2e}\n{sig}'
            else:
                p_text = 'N/A'
            #
            # Annotate p-value
            x1, x2 = 0, 1
            y_max = data_row[gp].max() * 1.15
            ax.plot([x1, x1, x2, x2], [y_max * 0.92, y_max, y_max, y_max * 0.92], lw=1.2, c='k')
            ax.text((x1 + x2) / 2, y_max * 1.02, p_text, ha='center', va='bottom', fontsize=8)
            # Titles/labels
            if row_idx == 0:
                ax.set_title(gp, fontsize=10, pad=6)
            if col_idx == 0:
                ax.set_ylabel(ct, fontsize=9, rotation=0, labelpad=30, va="center")
            #
            ax.set_xlabel("")
            ax.tick_params(labelsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    #
    fig.suptitle("Pseudobulk GP Activation (RA vs HD)", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig, dpi=200, bbox_inches='tight')
    plt.close(fig)

print(f"Single-page 6x12 PDF saved to: {output_pdf}")

#------------------------------------
# DEG 5 SUBTYPES
#------------------------------------
import os
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

# ── Parameters ──
data_dir = '/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/de_analysis/'
os.makedirs(data_dir, exist_ok=True)

condition_column = 'disease'
condition_1 = 'normal'
condition_2 = 'rheumatoid arthritis'

logfc_cutoff = 0.585
pval_cutoff = 0.05

# ── 1. Map symbols → Ensembl using map_series ──
# map_series: index = Ensembl, values = symbol
# We need symbol → Ensembl (invert it)
symbol_to_ensembl = pd.Series(map_series.index, index=map_series.values).to_dict()

# Get symbols from attention_matrix_df
gene_symbols = attention_matrix_df.columns.tolist()
print(f"Symbols from attention_matrix_df: {len(gene_symbols)}")

# Map to Ensembl
ensembl_list = []
unmapped = []
for sym in gene_symbols:
    ens = symbol_to_ensembl.get(sym)
    if ens:
        ensembl_list.append(ens)
    else:
        unmapped.append(sym)

print(f"Mapped {len(ensembl_list)} / {len(gene_symbols)} symbols to Ensembl IDs")
if unmapped:
    print(f"Unmapped symbols ({len(unmapped)}): {unmapped[:10]}{'...' if len(unmapped)>10 else ''}")

# Keep only Ensembl IDs that exist in adata.var_names
ensembl_in_adata = [ens for ens in ensembl_list if ens in adata.var_names]
print(f"Found {len(ensembl_in_adata)} matching Ensembl IDs in adata.var_names")
# ── 2. Subset adata_subset to GP-relevant genes ──
adata_subset_gpgenes = adata_subset[:, ensembl_in_adata].copy()
print(f"Subset to GP genes shape: {adata_subset_gpgenes.shape}")
# ── 3. Change var_names back to gene symbols ──
# Create a mapping from Ensembl → symbol (using map_series)
ensembl_to_symbol = map_series.to_dict()
# Apply to var_names
new_var_names = [ensembl_to_symbol.get(ens, ens) for ens in adata_subset_gpgenes.var_names]
adata_subset_gpgenes.var_names = new_var_names
print("Var names converted to symbols. First 10:")
print(adata_subset_gpgenes.var_names[:10])
# ── 4. Differential expression & volcano plots ──
unique_celltypes = sorted(adata_subset_gpgenes.obs['cell_type'].unique())
volcano_pdf = os.path.join(data_dir, "enhanced_volcano_all_clusters_gpgenes_fixed.pdf")
with PdfPages(volcano_pdf) as pdf:
    for cluster in unique_celltypes:
        print(f"\nProcessing cluster: {cluster}")
        adata_cluster = adata_subset_gpgenes[
            adata_subset_gpgenes.obs['cell_type'] == cluster
        ].copy()
        # Skip small clusters
        if adata_cluster.n_obs < 10:
            print(f"Skipping {cluster}: too few cells ({adata_cluster.n_obs})")
            continue
        #
        conds_present = adata_cluster.obs[condition_column].unique()
        if condition_1 not in conds_present or condition_2 not in conds_present:
            print(f"Skipping {cluster}: missing condition(s)")
            continue
        #
        # Run DE
        sc.tl.rank_genes_groups(
            adata_cluster,
            groupby=condition_column,
            groups=[condition_1],
            reference=condition_2,
            method='wilcoxon',
            layer='logcounts' if 'logcounts' in adata_cluster.layers else None,
            min_pct=0.15,
            use_raw=False
        )
        # Extract DE table
        deg_df = sc.get.rank_genes_groups_df(
            adata_cluster,
            group=condition_1,
            key='rank_genes_groups'
        )
        # Save full DE
        deg_csv = os.path.join(data_dir, f"DEG_{cluster}_{condition_1}_vs_{condition_2}.csv")
        deg_df.to_csv(deg_csv, index=False)
        # Add -log10(pvals) column (safe name, no hyphen)
        deg_df['neg_log10_pval'] = -np.log10(deg_df['pvals'].clip(lower=1e-300))
        # Filter UP/DOWN using raw pvals (or switch to 'pvals_adj' for FDR)
        deg_up = deg_df[
            (deg_df['logfoldchanges'] > logfc_cutoff) &
            (deg_df['pvals'] < pval_cutoff)
        ]
        deg_down = deg_df[
            (deg_df['logfoldchanges'] < -logfc_cutoff) &
            (deg_df['pvals'] < pval_cutoff)
        ]
        deg_up.to_csv(os.path.join(data_dir, f"DEG_UP_{cluster}.csv"), index=False)
        deg_down.to_csv(os.path.join(data_dir, f"DEG_DOWN_{cluster}.csv"), index=False)
        total_degs = len(deg_up) + len(deg_down)
        print(f"{cluster}: Total DEGs = {total_degs} | UP = {len(deg_up)} | DOWN = {len(deg_down)}")
        # ── Volcano plot ──
        fig, ax = plt.subplots(figsize=(7, 6))
        # Background (non-significant)
        sns.scatterplot(
            data=deg_df,
            x='logfoldchanges',
            y='neg_log10_pval',
            color='lightgray',
            s=25,
            alpha=0.6,
            ax=ax,
            edgecolor=None,
            legend=False
        )
        # UP / DOWN
        sns.scatterplot(
            data=deg_up,
            x='logfoldchanges',
            y='neg_log10_pval',
            color='#e74c3c',  # bright red
            s=50,
            label=f'UP (n={len(deg_up)})',
            ax=ax,
            edgecolor='black',
            linewidth=0.4
        )
        sns.scatterplot(
            data=deg_down,
            x='logfoldchanges',
            y='neg_log10_pval',
            color='#3498db',  # bright blue
            s=50,
            label=f'DOWN (n={len(deg_down)})',
            ax=ax,
            edgecolor='black',
            linewidth=0.4
        )
        # Cutoff lines
        ax.axvline(logfc_cutoff, ls='--', lw=1.3, color='black', alpha=0.8)
        ax.axvline(-logfc_cutoff, ls='--', lw=1.3, color='black', alpha=0.8)
        ax.axhline(-np.log10(pval_cutoff), ls='--', lw=1.3, color='black', alpha=0.8)
        # Label top 5 UP + top 5 DOWN
        top_up = deg_up.nlargest(5, 'logfoldchanges')
        top_down = deg_down.nsmallest(5, 'logfoldchanges')
        for df_top, color, offset_sign in [(top_up, '#e74c3c', 1), (top_down, '#3498db', -1)]:
            for _, row in df_top.iterrows():
                ax.text(
                    row['logfoldchanges'] + offset_sign * 0.05,
                    row['neg_log10_pval'] + 0.15,
                    row['names'],
                    fontsize=8.5,
                    color=color,
                    ha='left' if offset_sign > 0 else 'right',
                    va='bottom',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=0.2)
                )
        #
        # Styling
        ax.set_title(f"{cluster}: {condition_1} vs {condition_2}", fontsize=12, pad=10)
        ax.set_xlabel("log₂ Fold Change", fontsize=10)
        ax.set_ylabel("-log₁₀(p-value)", fontsize=10)
        ax.legend(frameon=True, fontsize=9, loc='upper left', bbox_to_anchor=(1.02, 1))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)
        pdf.savefig(fig, dpi=180, bbox_inches='tight')
        plt.close(fig)










#-----------------------------------------------------
#
# SUPER SUPER SUPER EXTRA: CENTRALITY ANALYSIS
#
#-----------------------------------------------------

import pandas as pd
import numpy as np
import networkx as nx

# ── Main loop: per patient, compute mean centrality per cell type ──
centrality_results = []
for patient in patients:
    print(f"\n=== Processing patient: {patient} ===")
    pdir = os.path.join(inter_dir, patient)
    # Load edge attention
    df2 = pd.read_csv(os.path.join(att_dir, f"{patient}_edge_att.csv"), delimiter=",").iloc[:, 1:]
    df2.columns = ["src", "tgt", "att"]
    # Patient cells
    patient_mask = cell_names["0"].str.startswith(patient + "__")
    patient_cell_names = cell_names.loc[patient_mask, "0"].reset_index(drop=True)
    cell_df = pd.DataFrame({
        "local_cell_index": np.arange(len(patient_cell_names)),
        "barcode_full": patient_cell_names
    })
    cell_df["barcode"] = cell_df["barcode_full"]
    cell_df["patient"] = patient
    cell_df["node_index"] = cell_df["local_cell_index"] + n_genes
    cell_df = cell_df.merge(
        meta[["barcode", "celltype", "patient"]],
        on=["barcode", "patient"],
        how="left"
    )
    # Filter to top_celltypes
    target_cells_mask = cell_df["celltype"].isin(top_celltypes)
    target_cells = cell_df.loc[target_cells_mask, "node_index"].values
    # ── Case 1: No cells in top_celltypes → add 0 for all 5 types ──
    if len(target_cells) == 0:
        print(f"  No cells in top cell types → adding 0 centrality for all types")
        for ct in top_celltypes:
            centrality_results.append({
                'patient': patient,
                'celltype': ct,
                'centrality': 0.0
            })
        continue
    #
    # ── Case 2: Build subgraph ──
    mask = (df2["src"].isin(target_cells)) | (df2["tgt"].isin(target_cells))
    subgraph_edges = df2[mask].copy()
    G = nx.DiGraph()
    # Add gene nodes
    for i in range(n_genes):
        G.add_node(i, node_type='gene')
    #
    # Add cell nodes (only from top_celltypes)
    for idx, row in cell_df[target_cells_mask].iterrows():
        G.add_node(row['node_index'], node_type='cell', celltype=row['celltype'])
    #
    # Add edges
    for _, row in subgraph_edges.iterrows():
        G.add_edge(row['src'], row['tgt'], weight=row['att'])
    #
    print(f"  Subgraph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    # ── Compute weighted degree centrality for cell nodes ──
    cell_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'cell']
    if len(cell_nodes) == 0:
        print(f"  No cell nodes after subgraph → adding 0 centrality")
        for ct in top_celltypes:
            centrality_results.append({
                'patient': patient,
                'celltype': ct,
                'centrality': 0.0
            })
        continue
    #
    weighted_centrality = {}
    for node in cell_nodes:
        in_edges = G.in_edges(node, data=True)
        out_edges = G.out_edges(node, data=True)
        total_weight = sum(d['weight'] for _, _, d in in_edges) + sum(d['weight'] for _, _, d in out_edges)
        weighted_centrality[node] = total_weight
    #
    # Merge with celltypes
    cell_centrality_df = pd.DataFrame({
        'node_index': list(weighted_centrality.keys()),
        'centrality': list(weighted_centrality.values())
    })
    cell_centrality_df = cell_centrality_df.merge(
        cell_df[['node_index', 'celltype']],
        on='node_index',
        how='left'
    )
    # Mean per cell type
    mean_centrality = cell_centrality_df.groupby('celltype')['centrality'].mean().reset_index()
    mean_centrality['patient'] = patient
    # Add missing cell types with 0
    missing_cts = set(top_celltypes) - set(mean_centrality['celltype'])
    for ct in missing_cts:
        mean_centrality = pd.concat([
            mean_centrality,
            pd.DataFrame({'patient': [patient], 'celltype': [ct], 'centrality': [0.0]})
        ], ignore_index=True)
    #
    centrality_results.append(mean_centrality)

# ── 5. Combine all patients into final table ──
final_table = pd.concat(centrality_results, ignore_index=True)

# Pivot: patients × cell types
final_table_pivot = final_table.pivot(
    index='patient',
    columns='celltype',
    values='centrality'
).fillna(0)  # already 0, but safe
# Reorder columns to match top_celltypes order
final_table_pivot = final_table_pivot[top_celltypes]
print("\nFinal table (mean centrality per cell type per patient):")
print(final_table_pivot)

# Save
output_csv = "/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/plots/mean_centrality_top5_celltypes.csv"
final_table_pivot.to_csv(output_csv)
print(f"Saved to: {output_csv}")

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import mannwhitneyu

# Load the table from CSV (adjust path if needed)
final_table = pd.read_csv(
    "/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/plots/mean_centrality_top5_celltypes_per_patient.csv")


# Add condition based on patient name
def get_condition(p):
    if 'RA' in p:
        return 'RA'
    else:
        return 'HD'


final_table['condition'] = final_table['patient'].apply(get_condition)
# Get the 5 cell types
cell_types = final_table['celltype'].unique()
cell_types = sorted(cell_types)
# PDF output
output_pdf = "/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/plots/centrality_boxplots_1x5.pdf"
with PdfPages(output_pdf) as pdf:
    fig, axs = plt.subplots(1, 5, figsize=(18, 5), sharey=True)
    for i, ct in enumerate(cell_types):
        ax = axs[i]
        # Data for this cell type
        data_ct = final_table[final_table['celltype'] == ct]
        # RA and HD values
        ra_vals = data_ct[data_ct['condition'] == 'RA']['centrality']
        hd_vals = data_ct[data_ct['condition'] == 'HD']['centrality']
        # Wilcoxon test
        if len(ra_vals) > 0 and len(hd_vals) > 0:
            stat, pval = mannwhitneyu(ra_vals, hd_vals, alternative='two-sided')
            if pval < 0.001:
                sig = '***'
            elif pval < 0.01:
                sig = '**'
            elif pval < 0.05:
                sig = '*'
            else:
                sig = 'ns'
            p_text = f'p = {pval:.2e}\n({sig})'
        else:
            p_text = 'N/A'
        #
        # Boxplot
        sns.boxplot(
            data=data_ct,
            x='condition',
            y='centrality',
            palette={'RA': '#d95f02', 'HD': '#1b9e77'},
            width=0.45,
            ax=ax
        )
        # Scatter
        sns.stripplot(
            data=data_ct,
            x='condition',
            y='centrality',
            hue='condition',
            palette={'RA': '#d95f02', 'HD': '#1b9e77'},
            size=6,
            jitter=0.18,
            alpha=0.85,
            ax=ax,
            legend=False
        )
        # p-value annotation
        x1, x2 = 0, 1
        y_max = data_ct['centrality'].max() * 1.15
        ax.plot([x1, x1, x2, x2], [y_max * 0.92, y_max, y_max, y_max * 0.92], lw=1.2, c='k')
        ax.text((x1 + x2) / 2, y_max * 1.02, p_text, ha='center', va='bottom', fontsize=9)
        ax.set_title(ct, fontsize=11, pad=6)
        ax.set_xlabel("")
        ax.set_ylabel("Mean Centrality" if i == 0 else "")
        ax.tick_params(labelsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    #
    fig.suptitle("Mean Centrality per Cell Type (RA vs HD)", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig, dpi=200, bbox_inches='tight')
    plt.close(fig)

print(f"PDF saved to: {output_pdf}")










#-----------------------------------------------------
#
# NEW UMAP PANEL
#
#-----------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.backends.backend_pdf import PdfPages
import umap
# 1. Setup Publication Aesthetics
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.sans-serif'] = 'Arial'
# Define your palette (matching your previous plots)
palette = {"HD": "blue", "RA": "red"}
point_colors = [palette.get(lbl, "gray") for lbl in labels]
# 2. Define the EXACT order requested
# Dictionary keys in Python 3.7+ maintain insertion order
plot_order = [
    ("emb_104", emb_104),
    ("l_128", l_128),
    ("g_128", g_128),
    ("l_64", l_64),
    ("g_64", g_64),
    ("l_32", l_32),
    ("g_32", g_32),
    ("ret_class", ret_class),
    ("softmax_ret_class", softmax_ret_class)
]
# 3. Initialize UMAP model
umap_model = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    metric="euclidean",
    random_state=42
)
# 4. Generate the PDF Panel
output_path = os.path.join(plots_dir, "FloREN_Nature_Methods_UMAP_Panel_Increase_Size.pdf")
with PdfPages(output_path) as pdf:
    # 1 row, 9 columns. Wide aspect ratio (36x4) is standard for this many subplots
    fig, axes = plt.subplots(1, 9, figsize=(32, 4), constrained_layout=True)
    for ax, (name, data) in zip(axes, plot_order):
        # Determine if we need UMAP or if it's already 2D (the last two)
        if data.shape[1] <= 2:
            embedding = data
        else:
            embedding = umap_model.fit_transform(data)
        #
        # Plotting with the categorical colors
        ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=point_colors,
            s=50,  # Slightly smaller points for a cleaner look
            alpha=0.8,
            edgecolors='white',
            linewidth=0.3
        )
        # Formatting for Nature style
        ax.set_title(name, fontsize=12, fontweight='bold', pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        # Remove all borders (spines) for the 'clean' look
        for spine in ax.spines.values():
            spine.set_visible(False)
    #
    # Add a single legend at the end or below if needed
    # (Optional: handles logic if you want an explicit legend on the last plot)
    pdf.savefig(fig, bbox_inches="tight")
    print(f"Successfully saved horizontal panel to {output_path}")
    plt.close(fig)










#---------------------------------------------------------------
#
#                 NEW RANKED CELL TYPE PLOT
#
#---------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

plt.rcParams["figure.figsize"] = (14, 6)
plt.rcParams["font.size"] = 12

##############################################################
# 1. Build list of patients from your earlier processing
##############################################################

patients = [d for d in os.listdir(inter_dir) if os.path.isdir(os.path.join(inter_dir, d))]
patients.sort()

print("Patients detected:", patients)

##############################################################
# 2. Cell metadata processing helper
##############################################################

meta = pd.read_csv("/Users/Inigo/Desktop/FloREN/Binvignat/cells_metadata.csv")
# Example: "classical monocyte__RA09_1_C_M__GGGACCTGTGCGAA-1"
meta_split = meta["x"].str.split("__", expand=True)
meta["celltype"] = meta_split[0]
meta["barcode"] = meta_split[1] + "__" + meta_split[2]  # e.g. "GGGACCTGTGCGAA-1"
meta["patient"] = meta_split[1]  # e.g. "RA09_1_C_M"

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
q75_celltype_saliency = all_cells.groupby("celltype")["saliency"].quantile(0.75).sort_values(ascending=False)


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Prepare data
df_plot = mean_celltype_saliency.sort_values(ascending=True).reset_index()
# Publication settings
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.sans-serif'] = 'Arial'
# Increase height slightly for 31 categories, but keep width compact
fig, ax = plt.subplots(figsize=(8, 8))
# 1. Smaller lollipops (linewidth and markersize)
ax.hlines(y=df_plot['celltype'], xmin=200, xmax=df_plot['saliency'],
          color='#bcbcbc', linestyle='-', linewidth=0.8, zorder=1)
# Smaller dots (s=30) with a thinner edge
ax.scatter(df_plot['saliency'], df_plot['celltype'],
           color='#2c3e50', s=30, edgecolors='white', linewidth=0.4, zorder=2)
# 2. Refine Labels
ax.set_xlabel("75th Percentile Saliency Score", fontsize=1, fontweight='bold')
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
    plots_dir,"Saliency_Ranking_NatureMethods_mean_mean.pdf"), dpi=300, bbox_inches='tight')










#---------------------------------------------------------------
#
#               NEW TOP 5 CELL TYPES SIGNATURE
#
#---------------------------------------------------------------

import mygene
mg = mygene.MyGeneInfo()
gene_map = mg.querymany(
    all_genes.gene.unique().tolist(),
    scopes="ensembl.gene",
    fields="symbol",
    species="human",
    as_dataframe=True
)
map_series = gene_map["symbol"].fillna(gene_map.index.to_series())


TOP_N = 5
top_celltypes = mean_celltype_saliency.head(TOP_N).index.tolist()
all_patients_gene_df = []
for target_celltype in top_celltypes:
    print(f"Processing cell type: {target_celltype}")
    for patient in patients:
        print(f"  Patient: {patient}")
        pdir = os.path.join(inter_dir, patient)
        df2 = pd.read_csv(
            os.path.join(att_dir, f"{patient}_edge_att.csv"),
            delimiter=","
        ).iloc[:, 1:]
        patient_mask = cell_names["0"].str.startswith(patient + "__")
        patient_cell_names = cell_names.loc[patient_mask, "0"].reset_index(drop=True)
        cell_df = pd.DataFrame({
            "local_cell_index": np.arange(len(patient_cell_names)),
            "barcode_full": patient_cell_names
        })
        cell_df["barcode"] = cell_df["barcode_full"]
        cell_df["patient"] = patient
        cell_df["node_index"] = cell_df["local_cell_index"] + n_genes
        cell_df = cell_df.merge(
            meta[["barcode", "celltype", "patient"]],
            on=["barcode", "patient"],
            how="left"
        )
        target_cells = cell_df.loc[
            cell_df["celltype"] == target_celltype,
            "node_index"
        ].values
        if len(target_cells) == 0:
            continue
        #
        df2.columns = ["src", "tgt", "att"]
        mask = (
            ((df2["src"].isin(target_cells)) & (df2["tgt"] < n_genes)) |
            ((df2["tgt"].isin(target_cells)) & (df2["src"] < n_genes))
        )
        df_celltype_edges = df2[mask]
        gene_attention = np.zeros(n_genes, dtype=np.float64)
        # src
        gene_mask = df_celltype_edges["src"] < n_genes
        np.add.at(
            gene_attention,
            df_celltype_edges.loc[gene_mask, "src"].values,
            df_celltype_edges.loc[gene_mask, "att"].values
        )
        # tgt
        gene_mask = df_celltype_edges["tgt"] < n_genes
        np.add.at(
            gene_attention,
            df_celltype_edges.loc[gene_mask, "tgt"].values,
            df_celltype_edges.loc[gene_mask, "att"].values
        )
        n_target_cells = len(target_cells)
        if n_target_cells > 0:
            gene_attention /= n_target_cells
        #
        gene_df = pd.DataFrame({
            "gene": gene_names,
            "saliency": gene_attention,
            "patient": patient,
            "celltype": target_celltype
        })
        all_patients_gene_df.append(gene_df)


all_genes = pd.concat(all_patients_gene_df, ignore_index=True)
all_genes["gene"] = all_genes["gene"].map(lambda g: map_series.get(g, g))
plot_df = all_genes.copy()
ranked_genes = (
    plot_df
    .groupby("gene")["saliency"]
    .mean()
    .sort_values(ascending=False)
)
top_ngenes = 50
top_genes = ranked_genes.head(top_ngenes)
top_genes = top_genes .reset_index()
top_genes.columns = ["gene", "saliency"]

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Prepare data
df_plot = top_genes
df_plot_sorted = df_plot.sort_values('saliency', ascending=True)
# Publication settings
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.sans-serif'] = 'Arial'
# Increase height slightly for 31 categories, but keep width compact
fig, ax = plt.subplots(figsize=(5, 8))
# 1. Smaller lollipops (linewidth and markersize)
ax.hlines(y=df_plot_sorted['gene'], xmin=0.2, xmax=df_plot_sorted['saliency'],
          color='#bcbcbc', linestyle='-', linewidth=0.8, zorder=1)
# Smaller dots (s=30) with a thinner edge
ax.scatter(df_plot_sorted['saliency'], df_plot_sorted['gene'],
           color='#2c3e50', s=30, edgecolors='white', linewidth=0.4, zorder=2)
# 2. Refine Labels
ax.set_xlabel("75th Percentile Saliency Score", fontsize=11, fontweight='bold')
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
    plots_dir,"Saliency_Ranking_NatureMethods_top5_signature.pdf"), dpi=300, bbox_inches='tight')










#---------------------------------------------------------------
#
#             DIFFERENTIAL GENE EXPRESSION IN FLOREN
#
#---------------------------------------------------------------

import mygene
mg = mygene.MyGeneInfo()
gene_map = mg.querymany(
    gene_names.tolist(),
    scopes="ensembl.gene",
    fields="symbol",
    species="human",
    as_dataframe=True
)
map_series = gene_map["symbol"].fillna(gene_map.index.to_series())

import os
import glob
import pandas as pd

# Directory
emb_dir = r"C:\Users\Inigo\Desktop\FloREN3.0\binvignat_outputs\floren_gene_embeddings"
# Dictionary to store embeddings
gene_embs = {}
gene_names_symbol = pd.Series(gene_names).map(lambda g: map_series.get(g, g))
# Iterate over csv files
for f in glob.glob(os.path.join(emb_dir, "*_gene_embs.csv")):
    # Extract patient_id from filename
    patient_id = os.path.basename(f).replace("_gene_embs.csv", "")
    # Load csv (no header)
    df = pd.read_csv(f, header=None)
    # Assign gene names as index
    df.index = gene_names_symbol
    # Store
    gene_embs[patient_id] = df

RA_dfs = []
HD_dfs = []
for patient_id, df in gene_embs.items():
    if "RA" in patient_id:
        RA_dfs.append(df)
    else:
        HD_dfs.append(df)

print(f"RA patients: {len(RA_dfs)}")
print(f"HD patients: {len(HD_dfs)}")
RA_mean_df = pd.concat(RA_dfs).groupby(level=0).mean()
HD_mean_df = pd.concat(HD_dfs).groupby(level=0).mean()

# Compute Euclidean distance per gene
distances = np.linalg.norm(RA_mean_df.values - HD_mean_df.values, axis=1)
# Convert to a pandas Series with gene names
gene_distances = pd.Series(distances, index=RA_mean_df.index, name="euclidean_distance")
print(gene_distances.head())
print(len(gene_distances))  # should be 2000

# 1. Setup Aesthetics
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.sans-serif'] = 'Arial'
sns.set_style("white")
fig, ax = plt.subplots(figsize=(7, 5))
# 2. Plot Distribution (Histogram + KDE)
sns.histplot(
    gene_distances,
    bins=60,
    kde=True,
    color="#2c3e50",
    edgecolor='white',
    line_kws={'linewidth': 2},
    ax=ax
)
# 3. Identify and Label Outliers (Top 10 genes)
top_n = 10
outliers = gene_distances.sort_values(ascending=False).head(top_n)
# We use a color gradient for the vertical lines to mark the top genes
colors = plt.cm.Reds(np.linspace(0.4, 0.8, top_n))
for i, (gene, dist) in enumerate(outliers.items()):
    # Add a subtle vertical line for each top gene
    ax.axvline(dist, color='red', alpha=0.2, linestyle='--', linewidth=0.8)
    # Annotate the gene name
    # We stagger the height of the text to prevent overlapping
    ax.text(
        dist,
        ax.get_ylim()[1] * (0.9 - (i * 0.07)),
        f" {gene}",
        color='red',
        fontsize=8,
        fontweight='bold',
        va='center'
    )

# 4. Refine Labels
ax.set_xlabel("Euclidean Distance (RA vs HD Mean Embeddings)", fontsize=10, fontweight='bold')
ax.set_ylabel("Number of Genes", fontsize=10, fontweight='bold')
ax.set_title("Shift in Gene Representation Space", loc='left', fontsize=12, pad=15)
# 5. Clean Spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', linestyle=':', alpha=0.6)
plt.tight_layout()
# Save
output_pdf = r"C:\Users\Inigo\Desktop\FloREN3.0\binvignat_outputs\plots\gene_embedding_distance_annotated.pdf"
plt.savefig(output_pdf, dpi=300, bbox_inches='tight')

dist = gene_distances.values
N = len(dist)
# Empirical p-value: probability of observing >= distance
pvals = np.array([(np.sum(dist >= d) + 1) / (N + 1) for d in dist])
gene_pvals = pd.Series(pvals, index=gene_distances.index, name="pval")
padj = multipletests(gene_pvals, method="fdr_bh")[1]
gene_results = pd.DataFrame({
    "distance": gene_distances,
    "pval": gene_pvals,
    "padj": padj
})
gene_results = gene_results.sort_values("distance", ascending=False)

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 1. Data Prep & Significance Calculation
gene_results["neglog10_pval"] = -np.log10(gene_results["pval"])
p_threshold = 0.05
# Use padj for biological rigor
sig_mask = (gene_results["pval"] < p_threshold)

# 2. Setup Aesthetics
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.sans-serif'] = 'Arial'
sns.set_style("white")

fig, ax = plt.subplots(figsize=(8, 7))

# 3. Plot Background (Non-significant)
ax.scatter(
    gene_results.loc[~sig_mask, "distance"],
    gene_results.loc[~sig_mask, "neglog10_pval"],
    color="#e0e0e0",
    alpha=0.5,
    s=15,
    zorder=2,
    edgecolors='none'
)

# 4. Plot Foreground (Significant)
ax.scatter(
    gene_results.loc[sig_mask, "distance"],
    gene_results.loc[sig_mask, "neglog10_pval"],
    color="#C23B3B",
    alpha=0.9,
    s=40,
    zorder=3,
    edgecolors='white',
    linewidth=0.3
)

# 5. Intelligent Manual Labeling (Avoid overlap without adjust_text)
# Rank by distance and take the top 15
top_hits = gene_results[sig_mask].sort_values("distance", ascending=False).head(10)

# We use a y_offset to stagger labels if they are too close
y_range = gene_results["neglog10_pval"].max() - gene_results["neglog10_pval"].min()
offset_step = y_range * 0.03

for i, (gene, row) in enumerate(top_hits.iterrows()):
    # Alternate the side or height slightly to create space
    # i%2 alternates between a slight boost and a slight drop
    stagger = offset_step if i % 2 == 0 else -offset_step
    ax.annotate(
        gene,
        xy=(row["distance"], row["neglog10_pval"]),
        xytext=(5, stagger),  # Offset text 5pts right and 'stagger' pts up/down
        textcoords="offset points",
        fontsize=4,
        fontweight='bold',
        color='black',
        va='center',
        # Adding a light box behind text makes it readable over points
        bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.7)
    )

# 6. Reference Lines
ax.axhline(-np.log10(p_threshold), color='black', linestyle='--', alpha=0.3, lw=1)
ax.text(ax.get_xlim()[0], -np.log10(p_threshold), f' P={p_threshold}',
        va='bottom', ha='left', fontsize=8, color='gray')

# 7. Final Polish
ax.set_xlabel("Euclidean Distance (RA vs HD)", fontsize=11, fontweight='bold')
ax.set_ylabel("$-log_{10}(p\text{-value})$", fontsize=11, fontweight='bold')
ax.set_title("Gene Embedding Shift Significance", loc='left', fontsize=14, pad=20)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
output_path = r"C:\Users\Inigo\Desktop\FloREN3.0\binvignat_outputs\plots\embedding_volcano_manual_clean.pdf"
plt.savefig(output_path, dpi=300)


GPgenes = [
"AGTRAP",
"RUNX3",
"CRYZ",
"DDX20",
"SCYL3",
"UBXN2A",
"CYP1B1",
"RHOB",
"PADI2",
"PADI4",
"CNR2",
"AK2",
"COQ8A",
"CDC42EP3",
"ID3",
"LMO4",
"AHCYL1",
"RBMXL1",
"CD2",
"PHTF1",
"FCRL5",
"S100A4",
"NIBAN1",
"NCF2",
"G0S2",
"UCN",
"TGFBR3",
"ENSG00000276216",
"GLUL",
"BCL11A",
"MXD1",
"PTAFR",
"BTBD8",
"PTGS2",
"FCGR3A",
"FCGR2A",
"OR2B11",
"ENSG00000237481",
"PGD",
"NFIA",
"GADD45A",
"GSTM3",
"TENT5C",
"CD58",
"FCER1A",
"LMNA",
"SLAMF7",
"FAM20B",
"C1orf21",
"ODR4",
"SLC30A1",
"FLVCR1",
"CHRM3-AS2",
"FAM228B",
"E2F6",
"SH3BGRL3",
"EVA1B",
"ENSG00000273338",
"CCDC88A",
"FGR",
"ZMYM1",
"SLC25A24",
"MNDA",
"ID2",
"DTNB",
"ENSG00000270605",
"NASP",
"LINC02591",
"VPS45",
"ENTREP3",
"RALGPS2",
"XCL1",
"TNFRSF18",
"TTF2",
"MCL1",
"FCGR1A",
"UROD",
"HOOK1",
"GBP5",
"SORT1",
"GBP1",
"GPATCH4",
"FCRL1",
"SLAMF1",
"PPOX",
"GDF7",
"WRAP73",
"PEA15",
"PFKFB2",
"PPP1R15B"
]
# distances of all genes
distances = gene_distances.values
N = len(distances)
# function to compute empirical pvalue
def empirical_pvalue(d):
    return (np.sum(distances >= d) + 1) / (N + 1)

# compute pvalues only for genes of interest
GPgenes_pvals = pd.Series({
    g: empirical_pvalue(gene_distances.loc[g])
    for g in GPgenes
    if g in gene_distances.index
})
GPgenes_pvals = GPgenes_pvals.sort_values()










#-----------------------------------------------------
#
# NEW GENE SIGNATURES COMPOSITION
#
#-----------------------------------------------------

# Assuming these are already defined in your environment:
# - patients
# - inter_dir
# - att_dir
# - cell_names
# - meta
# - n_genes
# - gene_names   ← very important! list/array of gene names in order (length = n_genes)

# List of all unique cell types (from your output)
cell_types = mean_celltype_saliency.index.tolist()

# We'll collect one aggregated vector per cell type
celltype_vectors = {}

for target_celltype in cell_types:
    print(f"\n=== Processing cell type: {target_celltype} ===")
    all_patients_gene_df = []
    for patient in patients:
        print(f"  Patient: {patient}")
        pdir = os.path.join(inter_dir, patient)
        # Load edge attention for this patient
        df2 = pd.read_csv(os.path.join(att_dir, f"{patient}_edge_att.csv"), delimiter=",").iloc[:, 1:]
        # Get cells for this patient
        patient_mask = cell_names["0"].str.startswith(patient + "__")
        patient_cell_names = cell_names.loc[patient_mask, "0"].reset_index(drop=True)
        cell_df = pd.DataFrame({
            "local_cell_index": np.arange(len(patient_cell_names)),
            "barcode_full": patient_cell_names
        })
        cell_df["barcode"] = cell_df["barcode_full"]
        cell_df["patient"] = patient
        # Map to global node index (cells start after genes)
        cell_df["node_index"] = cell_df["local_cell_index"] + n_genes
        # Merge with metadata to get celltype
        cell_df = cell_df.merge(
            meta[["barcode", "celltype", "patient"]],
            on=["barcode", "patient"],
            how="left"
        )
        # Select only target cell type cells
        target_cells = cell_df[
            (cell_df["celltype"] == target_celltype) &
            (cell_df["patient"] == patient)
            ]["node_index"].values
        # Filter edges connected to target cells and genes
        df2.columns = ["src", "tgt", "att"]
        mask = (
                ((df2["src"].isin(target_cells)) & (df2["tgt"] < n_genes)) |
                ((df2["tgt"].isin(target_cells)) & (df2["src"] < n_genes))
        )
        df_celltype_edges = df2[mask].copy()
        # Accumulate attention per gene
        gene_attention = np.zeros(n_genes, dtype=np.float64)
        # Gene as source
        gene_mask_src = df_celltype_edges["src"] < n_genes
        np.add.at(
            gene_attention,
            df_celltype_edges.loc[gene_mask_src, "src"].values,
            df_celltype_edges.loc[gene_mask_src, "att"].values
        )
        # Gene as target
        gene_mask_tgt = df_celltype_edges["tgt"] < n_genes
        np.add.at(
            gene_attention,
            df_celltype_edges.loc[gene_mask_tgt, "tgt"].values,
            df_celltype_edges.loc[gene_mask_tgt, "att"].values
        )
        # ✅ NEW: normalize by number of cells of this type in this patient
        n_target_cells = len(target_cells)
        if n_target_cells > 0:
            gene_attention /= n_target_cells
        #
        # Save per-patient file (optional but useful)
        out_dir = os.path.join(inter_dir, patient, "celltype_specific")
        os.makedirs(out_dir, exist_ok=True)
        #np.savetxt(
        #    os.path.join(out_dir, f"genes_att_{target_celltype.replace(' ', '_')}.csv"),
        #    gene_attention,
        #    delimiter=","
        #)
        # Create per-patient gene DataFrame
        gene_df = pd.DataFrame({
            "gene": gene_names,
            "saliency": gene_attention,
            "patient": patient,
            "celltype": target_celltype
        })
        all_patients_gene_df.append(gene_df)
    #
    # ── Aggregate across patients for this cell type ──
    if all_patients_gene_df:
        all_genes_celltype = pd.concat(all_patients_gene_df, ignore_index=True)
        # Compute mean saliency per gene across patients
        aggregated = all_genes_celltype.groupby('gene')['saliency'].mean().reset_index()
        aggregated.columns = ['gene', 'mean_attention']
        # Pivot to wide format (genes as columns)
        vector_wide = aggregated.set_index('gene')['mean_attention'].to_frame().T
        vector_wide.index = [target_celltype]
        celltype_vectors[target_celltype] = vector_wide
        print(f"  → Aggregated for {target_celltype}: {len(aggregated)} genes")
    else:
        print(f"  → No data for {target_celltype}")

# ── 5. Combine all cell types into one big matrix (like NMF W) ──
if celltype_vectors:
    celltype_attention_matrix = pd.concat(celltype_vectors.values(), axis=0)
    # Ensure all genes are present (fill missing with 0)
    celltype_attention_matrix = celltype_attention_matrix.reindex(columns=gene_names, fill_value=0.0)
    # Sort rows by cell type (optional)
    celltype_attention_matrix = celltype_attention_matrix.reindex(cell_types)
    print("\nFinal cell-type × gene attention matrix shape:", celltype_attention_matrix.shape)
    print(celltype_attention_matrix.iloc[:5, :8])  # preview
    # Save
    #output_path = "/Users/Inigo/Desktop/FloREN3.0/straeten_output/celltype_attention_matrix.csv"
    #celltype_attention_matrix.to_csv(output_path)
    #print(f"Saved to: {output_path}")
else:
    print("No cell types had valid data.")

celltype_attention_matrix.columns = celltype_attention_matrix.columns.map(lambda g: map_series.get(g, g))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------
# 1. Select top 50 genes by mean across celltypes
# -------------------------------------------------------
gene_means = celltype_attention_matrix.mean(axis=0)
top50_genes = (
    gene_means
    .sort_values(ascending=False)
    .head(50)
    .index
)

top_matrix = celltype_attention_matrix[top50_genes]
# Rank genes by mean
top_matrix = top_matrix.loc[:, gene_means[top50_genes].sort_values(ascending=False).index]
# -------------------------------------------------------
# 2. Convert to long format
# -------------------------------------------------------
plot_df = (
    top_matrix
    .reset_index()
    .melt(id_vars="index", var_name="gene", value_name="value")
    .rename(columns={"index": "celltype"})
)
# Gene rank order
plot_df["gene"] = pd.Categorical(
    plot_df["gene"],
    categories=top_matrix.columns,
    ordered=True
)
# -------------------------------------------------------
# 3. Plot
# -------------------------------------------------------
sns.set_style("white")
celltypes = top_matrix.index
palette = sns.color_palette("tab10", len(celltypes))
fig, ax = plt.subplots(figsize=(16,6))
for i, ct in enumerate(celltypes):
    ct_df = plot_df[plot_df["celltype"] == ct]
    ax.plot(
        ct_df["gene"],
        ct_df["value"],
        marker="o",
        markersize=4,
        linewidth=1.5,
        color=palette[i],
        label=ct
    )

ax.set_xlabel("Top 50 Genes (Ranked by Mean Attention)")
ax.set_ylabel("Attention Score")
ax.set_title("Celltype Attention Across Top Genes")
ax.set_xticklabels(ax.get_xticklabels(), rotation=90) #fontsize=5
ax.legend(
    title="Celltype",
    bbox_to_anchor=(1.02,1),
    loc="upper left",
    frameon=False
)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(
    plots_dir,"Saliency_gene_signatures_comparison_50.pdf"), dpi=300, bbox_inches='tight')










#-----------------------------------------------------
#
#  GENE CONNETIONS SIGNIFICANCE AND MODEL
#
#-----------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import phate

top50_genes = ['HES4', 'ISG15', 'TNFRSF18', 'TNFRSF4', 'WRAP73', 'PHF13', 'PGD',
       'RBP7', 'DFFA', 'AGTRAP', 'TNFRSF1B', 'EFHD2', 'CASP9', 'FBXO42',
       'ATP13A2', 'PADI2', 'PADI4', 'RCC2', 'PINK1', 'CDA', 'OTUD3', 'ECE1',
       'C1QA', 'LINC01355', 'ID3', 'RUNX3', 'PAQR7', 'ENSG00000233478', 'CNR2',
       'ENSG00000270605', 'IFI6', 'STMN1', 'ZNF683', 'RPA2', 'THEMIS2',
       'AHDC1', 'SH3BGRL3', 'FGR', 'PTAFR', 'AK2', 'AGO1', 'ZMYM1', 'SESN2',
       'GMEB1', 'CSF3R', 'SLC2A1', 'RPS8', 'MYCL', 'ZNF684', 'ATP6V0B']

# Load data
path = r"C:\Users\Inigo\Desktop\FloREN3.0\binvignat_outputs\plots\RA_mean_df.csv"
RA_mean_df = pd.read_csv(path, index_col=0)
X = RA_mean_df.values
# Dimensionality reduction
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

# Masks
gp_mask = RA_mean_df.index.isin(GPgenes)
top50_mask = RA_mean_df.index.isin(top50_genes)
# ensure hierarchy: top50 overrides GPgenes
gp_only_mask = gp_mask & ~top50_mask
top50_only_mask = top50_mask
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()
for ax, (method, emb) in zip(axes, embeddings.items()):
    # Background genes
    background_mask = ~(gp_only_mask | top50_only_mask)
    ax.scatter(
        emb[background_mask, 0],
        emb[background_mask, 1],
        s=10,
        alpha=0.25,
        color="gray"
    )
    # GPgenes
    ax.scatter(
        emb[gp_only_mask, 0],
        emb[gp_only_mask, 1],
        s=35,
        color="red",
        label="GPgenes"
    )
    # Top50 genes
    ax.scatter(
        emb[top50_only_mask, 0],
        emb[top50_only_mask, 1],
        s=45,
        color="blue",
        label="Top50 distance genes"
    )
    ax.set_title(method)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")

plt.tight_layout()
out_path = r"C:\Users\Inigo\Desktop\FloREN3.0\binvignat_outputs\plots\RA_embeddings_highlight_two_gene_sets.pdf"
plt.savefig(out_path)
plt.close()
print("Saved to:", out_path)










#-----------------------------------------------------
#
#  GENE PROGRAMS ON GENE EMBEDDINGS
#
#-----------------------------------------------------

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
import sklearn.metrics

matrices = {
    "Cosine similarity": X_cosine,
    "Euclidean distance": X_euclidean,
    "Pearson correlation": X_pearson
}
# Metadata annotation
meta_df = pd.DataFrame(index=RA_mean_df.index)
meta_df["GPgenes"] = meta_df.index.isin(GPgenes)
meta_df["Top50"] = meta_df.index.isin(top50_genes)
# Convert to colors
meta_colors = pd.DataFrame(index=RA_mean_df.index)
meta_colors["GPgenes"] = meta_df["GPgenes"].map({
    True: "red",
    False: "lightgray"
})
meta_colors["Top50"] = meta_df["Top50"].map({
    True: "blue",
    False: "lightgray"
})
# PDF output
pdf_path = r"C:\Users\Inigo\Desktop\FloREN3.0\binvignat_outputs\plots\gene_similarity_heatmaps.pdf"
with PdfPages(pdf_path) as pdf:
    for name, matrix in matrices.items():
        df = pd.DataFrame(matrix, index=RA_mean_df.index, columns=RA_mean_df.index)
        cg = sns.clustermap(
            df,
            cmap="viridis",
            row_cluster=True,
            col_cluster=True,
            row_colors=meta_colors,
            xticklabels=False,
            yticklabels=True,
            figsize=(12, 12)
        )
        cg.fig.suptitle(name, fontsize=16)
        pdf.savefig(cg.fig)
        plt.close(cg.fig)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from matplotlib.backends.backend_pdf import PdfPages
import igraph as ig
import leidenalg
#from dynamicTreeCut import cutreeHybrid

genes = RA_mean_df.index
dist_df = pd.DataFrame(X_euclidean, index=genes, columns=genes)

# KMeans clustering
kmeans = KMeans(n_clusters=9, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

# GMM clustering
gmm = GaussianMixture(n_components=9, random_state=42)
gmm_labels = gmm.fit_predict(X)

# Dynamic tree clustering
#link = linkage(squareform(X_euclidean), method="average")
#dynamic_clusters = cutreeHybrid(
#    link,
#    distM=X_euclidean
#)
#dynamic_labels = dynamic_clusters["labels"]

# Leiden clustering
nbrs = NearestNeighbors(n_neighbors=3).fit(X)
distances, indices = nbrs.kneighbors(X)
edges = []
for i in range(len(indices)):
    for j in indices[i]:
        if i != j:
            edges.append((i, j))

g = ig.Graph(edges=edges, directed=False)
partition = leidenalg.find_partition(
    g,
    leidenalg.RBConfigurationVertexPartition,
    resolution_parameter=1
)
leiden_labels = np.array(partition.membership)

# Metadata bars
meta_left = pd.DataFrame(index=genes)
meta_left["GPgenes"] = meta_left.index.isin(GPgenes)
meta_left["Top50"] = meta_left.index.isin(top50_genes)
meta_left_colors = pd.DataFrame(index=genes)
meta_left_colors["GPgenes"] = meta_left["GPgenes"].map(
    {True: "red", False: "lightgray"}
)
meta_left_colors["Top50"] = meta_left["Top50"].map(
    {True: "blue", False: "lightgray"}
)

# Clustering annotations (top)
meta_top = pd.DataFrame({
    "KMeans": kmeans_labels,
    "GMM": gmm_labels,
    #"DynamicTree": dynamic_labels,
    "Leiden": leiden_labels
}, index=genes)
palette = sns.color_palette("tab10", 10)
meta_top_colors = pd.DataFrame(index=genes)
for col in meta_top.columns:
    meta_top_colors[col] = meta_top[col].map(
        lambda x: palette[int(x) % len(palette)]
    )

# Heatmap
cg = sns.clustermap(
    dist_df,
    cmap="viridis",
    row_cluster=True,
    col_cluster=True,
    row_colors=meta_left_colors,
    col_colors=meta_top_colors,
    xticklabels=False,
    yticklabels=True,
    figsize=(40, 40)
)
# Move labels to right
cg.ax_heatmap.yaxis.tick_right()
# Decrease gene label size
cg.ax_heatmap.tick_params(axis='y', labelsize=1) # 2pt is usually the limit for PDF zoom legibility
# Save
pdf_path = r"C:\Users\Inigo\Desktop\FloREN3.0\binvignat_outputs\plots\euclidean_clustered_heatmap.pdf"
with PdfPages(pdf_path) as pdf:
    pdf.savefig(cg.fig)

plt.close()

print("Saved:", pdf_path)










#-----------------------------------------------------
#
# GENE PROGRAMS ON ATTENTION NETWORK
#
#-----------------------------------------------------

# Assuming your DataFrame is called celltype_attention_matrix
# 1. Compute mean value for each column
column_means = celltype_attention_matrix.mean(axis=0)
# 2. Sort columns by mean (descending)
sorted_means = column_means.sort_values(ascending=False)
# 3. Get top 50 column names
top_50_columns = sorted_means.head(50).index.tolist()
# Optional: also get their mean values
top_50_with_values = sorted_means.head(50)
top50_genes = top_50_columns
# Define threshold
pval_threshold = 0.05
# Subset significant genes
significant_genes = gene_results[gene_results['pval'] < pval_threshold]
# Get list of gene names (index)
sig_gene_list = significant_genes.index.tolist()
GPgenes = sig_gene_list



celltypes= list(mean_celltype_saliency.index)
all_patients_gene_df = []
for target_celltype in celltypes:
    print(f"Processing cell type: {target_celltype}")
    for patient in patients:
        print(f"  Patient: {patient}")
        pdir = os.path.join(inter_dir, patient)
        df2 = pd.read_csv(
            os.path.join(att_dir, f"{patient}_edge_att.csv"),
            delimiter=","
        ).iloc[:, 1:]
        patient_mask = cell_names["0"].str.startswith(patient + "__")
        patient_cell_names = cell_names.loc[patient_mask, "0"].reset_index(drop=True)
        cell_df = pd.DataFrame({
            "local_cell_index": np.arange(len(patient_cell_names)),
            "barcode_full": patient_cell_names
        })
        cell_df["barcode"] = cell_df["barcode_full"]
        cell_df["patient"] = patient
        cell_df["node_index"] = cell_df["local_cell_index"] + n_genes
        cell_df = cell_df.merge(
            meta[["barcode", "celltype", "patient"]],
            on=["barcode", "patient"],
            how="left"
        )
        target_cells = cell_df.loc[
            cell_df["celltype"] == target_celltype,
            "node_index"
        ].values
        if len(target_cells) == 0:
            continue
        #
        df2.columns = ["src", "tgt", "att"]
        mask = (
            ((df2["src"].isin(target_cells)) & (df2["tgt"] < n_genes)) |
            ((df2["tgt"].isin(target_cells)) & (df2["src"] < n_genes))
        )
        df_celltype_edges = df2[mask]
        gene_attention = np.zeros(n_genes, dtype=np.float64)
        # src
        gene_mask = df_celltype_edges["src"] < n_genes
        np.add.at(
            gene_attention,
            df_celltype_edges.loc[gene_mask, "src"].values,
            df_celltype_edges.loc[gene_mask, "att"].values
        )
        # tgt
        gene_mask = df_celltype_edges["tgt"] < n_genes
        np.add.at(
            gene_attention,
            df_celltype_edges.loc[gene_mask, "tgt"].values,
            df_celltype_edges.loc[gene_mask, "att"].values
        )
        gene_df = pd.DataFrame({
            "gene": gene_names,
            "saliency": gene_attention,
            "patient": patient,
            "celltype": target_celltype
        })
        all_patients_gene_df.append(gene_df)

all_genes = pd.concat(all_patients_gene_df, ignore_index=True)
all_genes["gene"] = all_genes["gene"].map(lambda g: map_series.get(g, g))

top_250_genes = list(set(list(top50_genes) + list(GPgenes)))
overlap_genes = list(set(top50_genes) & set(GPgenes))

ensembl_to_symbol = map_series.to_dict()
symbol_to_ensembl = {sym: ens for ens, sym in ensembl_to_symbol.items()}
# Handle cases where multiple Ensembl map to same symbol (rare, but possible)
# We'll take the first one for simplicity
symbol_to_ensembl = {}
for ens, sym in ensembl_to_symbol.items():
    if sym not in symbol_to_ensembl:
        symbol_to_ensembl[sym] = ens

# Now map your top 250 symbols to their original Ensembl IDs
top_250_ensembl = []
for sym in top_250_genes:
    ens = symbol_to_ensembl.get(sym)
    if ens:
        top_250_ensembl.append(ens)
    else:
        print(f"Warning: No Ensembl ID found for symbol '{sym}'")

print(f"Found {len(top_250_ensembl)} / {len(top_250_genes)} Ensembl IDs for top symbols")

# Map gene → index (for filtering edges)
gene_to_index = {gene: i for i, gene in enumerate(gene_names)}
#top_250_indices = [gene_to_index.get(g, -1) for g in top_250_genes]
top_250_indices = [gene_to_index.get(ens, -1) for ens in top_250_ensembl]
top_250_indices = [i for i in top_250_indices if i != -1]  # remove any missing

# Optional: add condition column for nodes
def patient_condition(p):
    if "RA" in p:
        return "RA"
    elif "Control" in p or "HD" in p or "HC" in p:
        return "Control"  # adjust labels as needed
    else:
        return "Other"

all_genes["condition"] = all_genes["patient"].apply(patient_condition)

# ===================== GRN: Limited to Top 250 Genes from Top Cell Types =====================

# ── 1. Aggregate gene saliency (nodes) — from celltype attention, limited to top 250 ──
gene_condition_mean = all_genes.groupby(["condition", "gene"], as_index=False)["saliency"].mean()

# Filter nodes to top 250 genes
ra_nodes = gene_condition_mean[
    (gene_condition_mean["condition"] == "RA") &
    (gene_condition_mean["gene"].isin(top_250_genes))
    ].copy()
ctrl_nodes = gene_condition_mean[
    (gene_condition_mean["condition"] == "Control") &
    (gene_condition_mean["gene"].isin(top_250_genes))
    ].copy()

# Optional: rename saliency column for consistency
#ra_nodes = ra_nodes.rename(columns={"saliency": "mean_attention"})
#ctrl_nodes = ctrl_nodes.rename(columns={"saliency": "mean_attention"})

print(f"RA nodes (top 250): {len(ra_nodes)}")
print(f"Control nodes (top 250): {len(ctrl_nodes)}")

# ── 2. Aggregate edges — gene-gene only, limited to pairs in top 250 ──
all_edges = []

for patient in patients:
    condition = patient_condition(patient)
    if condition not in ["RA", "Control"]:
        continue
    #
    df2 = pd.read_csv(os.path.join(att_dir, f"{patient}_edge_att.csv"), delimiter=",").iloc[:, 1:]
    df2.columns = ["src", "tgt", "att"]
    # Keep only gene-gene edges
    df2 = df2[(df2["src"] < n_genes) & (df2["tgt"] < n_genes)].copy()
    # Critical filter: only edges between top 250 genes
    df2 = df2[
        df2["src"].isin(top_250_indices) &
        df2["tgt"].isin(top_250_indices)
        ].copy()
    df2["patient"] = patient
    df2["condition"] = condition
    all_edges.append(df2)

all_edges = pd.concat(all_edges, ignore_index=True)

# Mean attention per edge per condition
edge_condition_mean = all_edges.groupby(["condition", "src", "tgt"], as_index=False)["att"].mean()

ra_edges = edge_condition_mean.query("condition == 'RA'")
ctrl_edges = edge_condition_mean.query("condition == 'Control'")

# Remove self-loops
ra_edges_noself = ra_edges[ra_edges["src"] != ra_edges["tgt"]].copy()
ctrl_edges_noself = ctrl_edges[ctrl_edges["src"] != ctrl_edges["tgt"]].copy()

print(f"RA edges (top 250 genes): {len(ra_edges_noself)}")
print(f"Control edges (top 250 genes): {len(ctrl_edges_noself)}")

# ── 3. Optional Cumulative Plot (adapted to top 250) ──
# Sum saliency across top 250 genes (from celltype attention)
#top250_saliency = all_genes[all_genes["gene"].isin(top_250_genes)].groupby("gene")["saliency"].sum()
#top250_sum_df = pd.DataFrame({
#    "gene": top250_saliency.index,
#    "saliency": top250_saliency.values
#}).sort_values("saliency", ascending=True).reset_index(drop=True)
#
#top250_sum_df["cum_sum"] = top250_sum_df["saliency"].cumsum()
#total_top250 = top250_sum_df["saliency"].sum()
#top250_sum_df["cum_frac"] = top250_sum_df["cum_sum"] / total_top250
#
# Plot cumulative for top 250
#plt.figure(figsize=(7, 5))
#plt.plot(top250_sum_df.index + 1, top250_sum_df["cum_frac"], linewidth=2)
#plt.axhline(0.95, linestyle="--", color="gray")
#plt.xlabel("Number of top genes (sorted by importance)")
#plt.ylabel("Cumulative fraction of total attention (top 250)")
#plt.title("Cumulative Attention in Top 250 Genes from Top 5 Cell Types")
#plt.tight_layout()
#cum_pdf = os.path.join(plots_dir, "FloREN_Top250_cumsum.pdf")
#plt.savefig(cum_pdf, format="pdf")
#plt.close()

# ── 4. Save files (limited to top 250) ──
ra_edges_noself.to_csv("/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/plots/ra_edges_noself_top125_new.csv")
ctrl_edges_noself.to_csv("/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/plots/ctrl_edges_noself_top125_new.csv")
ra_nodes.to_csv("/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/plots/ra_nodes_top125_new.csv")
ctrl_nodes.to_csv("/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/plots/ctrl_nodes_top125_new.csv")

print("GRN files saved (limited to top 250 genes from top 5 cell types)")

import pandas as pd
import igraph as ig
from collections import defaultdict
import mygene

# Load data (top 250 filtered)
ra_nodes = pd.read_csv("/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/plots/ra_nodes_top125_new.csv").iloc[:, 1:]
ctrl_nodes = pd.read_csv("/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/plots/ctrl_nodes_top125_new.csv").iloc[:, 1:]
ra_edges_noself = pd.read_csv("/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/plots/ra_edges_noself_top125_new.csv").iloc[
                  :, 1:]
ctrl_edges_noself = pd.read_csv(
    "/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/plots/ctrl_edges_noself_top125_new.csv").iloc[:, 1:]

# Rename saliency column if needed (from mean_attention)
if 'mean_attention' in ra_nodes.columns:
    ra_nodes = ra_nodes.rename(columns={'mean_attention': 'saliency'})
    ctrl_nodes = ctrl_nodes.rename(columns={'mean_attention': 'saliency'})

# Assume 'gene' column contains symbols (from previous mapping)
# If any are still Ensembl, convert them
ensembl_list = [g for g in set(ra_nodes['gene'].tolist() + ctrl_nodes['gene'].tolist()) if
                isinstance(g, str) and g.startswith('ENSG')]

ensembl_to_symbol = {}
if ensembl_list:
    mg = mygene.MyGeneInfo()
    results = mg.querymany(ensembl_list, scopes='ensembl.gene', fields='symbol', species='human')
    for r in results:
        if 'notfound' not in r:
            ens = r['query']
            sym = r.get('symbol', ens)
            ensembl_to_symbol[ens] = sym
    print(f"Converted {len(ensembl_to_symbol)} remaining Ensembl IDs to symbols")
else:
    print("All genes appear to be symbols already")

# Apply mapping to gene column (safe: symbols stay as is)
ra_nodes['gene'] = ra_nodes['gene'].map(lambda g: ensembl_to_symbol.get(g, g))
ctrl_nodes['gene'] = ctrl_nodes['gene'].map(lambda g: ensembl_to_symbol.get(g, g))

# Create local index maps: gene symbol → local index (0 to len-1)
ra_gene_to_local = {row['gene']: i for i, row in ra_nodes.iterrows()}
ctrl_gene_to_local = {row['gene']: i for i, row in ctrl_nodes.iterrows()}

# Remap edges: convert original src/tgt (0-1999) to local indices (0-249) based on gene symbols
def remap_edges(edges_df, nodes_df, gene_to_local):
    edges_remapped = []
    for _, row in edges_df.iterrows():
        # Get genes from original indices (using nodes_df)
        src_idx = int(row['src'])
        tgt_idx = int(row['tgt'])
        if src_idx >= len(nodes_df) or tgt_idx >= len(nodes_df):
            continue
        src_gene = nodes_df.iloc[src_idx]['gene']
        tgt_gene = nodes_df.iloc[tgt_idx]['gene']
        src_local = gene_to_local.get(src_gene, -1)
        tgt_local = gene_to_local.get(tgt_gene, -1)
        if src_local != -1 and tgt_local != -1:
            edges_remapped.append({
                'src': src_local,
                'tgt': tgt_local,
                'att': row['att']
            })
    return pd.DataFrame(edges_remapped)


ra_edges_local = remap_edges(ra_edges_noself, ra_nodes, ra_gene_to_local)
ctrl_edges_local = remap_edges(ctrl_edges_noself, ctrl_nodes, ctrl_gene_to_local)

print(f"RA remapped edges: {len(ra_edges_local)}")
print(f"Control remapped edges: {len(ctrl_edges_local)}")

# Gene map: local index → gene symbol
ra_gene_map = {i: gene for i, gene in enumerate(ra_nodes['gene'])}
ctrl_gene_map = {i: gene for i, gene in enumerate(ctrl_nodes['gene'])}

def build_and_detect_infomap(edges_df_local, nodes_df, gene_map, condition):
    n_vertices = len(nodes_df)
    if n_vertices == 0:
        print(f"No nodes for {condition}")
        return []
    #
    G = ig.Graph(directed=True)
    G.add_vertices(n_vertices)
    #
    G.vs['saliency'] = nodes_df['saliency'].tolist()
    G.vs['gene'] = nodes_df['gene'].tolist()
    #
    edges = list(zip(edges_df_local['src'], edges_df_local['tgt']))
    weights = edges_df_local['att'].tolist()
    G.add_edges(edges)
    G.es['weight'] = weights
    #
    communities = G.community_infomap(edge_weights='weight', trials=10)
    #
    modules = defaultdict(list)
    for node_idx, mod_id in enumerate(communities.membership):
        gene = gene_map.get(node_idx, f"Unknown_{node_idx}")
        sal = G.vs[node_idx]['saliency']
        modules[mod_id].append((gene, sal))
    #
    sorted_modules = sorted(modules.items(), key=lambda x: len(x[1]), reverse=True)
    #
    print(f"\nInfomap communities for {condition} (top 250 genes):")
    for mod_id, gene_list in sorted_modules:
        print(f"Module {mod_id} (size: {len(gene_list)}):")
        for gene, sal in sorted(gene_list, key=lambda x: x[1], reverse=True):
            print(f"  - {gene}: {sal:.4f}")
    #
    return sorted_modules

from collections import defaultdict
import igraph as ig
import leidenalg

def build_and_detect_leiden(edges_df_local, nodes_df, gene_map, condition, resolution=1):
    n_vertices = len(nodes_df)
    if n_vertices == 0:
        print(f"No nodes for {condition}")
        return []
    #
    # Build graph
    G = ig.Graph(directed=False)
    G.add_vertices(n_vertices)
    # Node attributes
    G.vs['saliency'] = nodes_df['saliency'].tolist()
    G.vs['gene'] = nodes_df['gene'].tolist()
    # Add edges
    edges = list(zip(edges_df_local['src'], edges_df_local['tgt']))
    weights = edges_df_local['att'].tolist()
    G.add_edges(edges)
    G.es['weight'] = weights
    # Leiden community detection
    partition = leidenalg.find_partition(
        G,
        leidenalg.RBConfigurationVertexPartition,
        weights='weight',
        resolution_parameter=resolution
    )
    # Collect modules
    modules = defaultdict(list)
    for node_idx, mod_id in enumerate(partition.membership):
        gene = gene_map.get(node_idx, f"Unknown_{node_idx}")
        sal = G.vs[node_idx]['saliency']
        modules[mod_id].append((gene, sal))
    #
    sorted_modules = sorted(modules.items(), key=lambda x: len(x[1]), reverse=True)
    print(f"\nLeiden communities for {condition} (resolution={resolution}):")
    for mod_id, gene_list in sorted_modules:
        print(f"Module {mod_id} (size: {len(gene_list)}):")
        for gene, sal in sorted(gene_list, key=lambda x: x[1], reverse=True):
            print(f"  - {gene}: {sal:.4f}")
    #
    return sorted_modules

# Run
ra_modules = build_and_detect_infomap(ra_edges_local, ra_nodes, ra_gene_map, 'RA')
ctrl_modules = build_and_detect_infomap(ctrl_edges_local, ctrl_nodes, ctrl_gene_map, 'Control')
ra_modules = build_and_detect_leiden(ra_edges_local, ra_nodes, ra_gene_map, 'RA', resolution=1)
ctrl_modules = build_and_detect_leiden(ctrl_edges_local, ctrl_nodes, ctrl_gene_map, 'Control', resolution=1)


# Jaccard similarity example
def module_jaccard(mod1, mod2):
    set1 = set([g for g, s in mod1])
    set2 = set([g for g, s in mod2])
    return len(set1 & set2) / len(set1 | set2) if len(set1 | set2) > 0 else 0

print("\nTop RA module vs all CTRL modules Jaccard:")
if ra_modules:
    top_ra = ra_modules[0][1]
    for i, (mod_id, ctrl_mod) in enumerate(ctrl_modules):
        sim = module_jaccard(top_ra, ctrl_mod)
        print(f"RA top vs CTRL {mod_id}: {sim:.4f}")
else:
    print("No RA modules detected")

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# ── Assuming you already ran the previous blocks and have: ──
# ra_nodes, ctrl_nodes, ra_edges_local, ctrl_edges_local
# ra_modules, ctrl_modules (lists of sorted modules from Infomap)

# ── 1. Create module mappings: symbol → module ID ──
ra_symbol_to_module = {}
for mod_id, gene_list in ra_modules:
    for gene, _ in gene_list:
        ra_symbol_to_module[gene] = mod_id

ctrl_symbol_to_module = {}
for mod_id, gene_list in ctrl_modules:
    for gene, _ in gene_list:
        ctrl_symbol_to_module[gene] = mod_id


# ── 2. Build full NetworkX graphs ──
def build_full_graph(nodes_df, edges_df, symbol_to_module):
    G = nx.DiGraph()
    # Add nodes with saliency and module
    for i, row in nodes_df.iterrows():
        symbol = row['gene']
        mod_id = symbol_to_module.get(symbol, -1)
        G.add_node(symbol, saliency=row['saliency'], module=mod_id)
    #
    # Add all edges
    for _, row in edges_df.iterrows():
        src_idx = int(row['src'])
        tgt_idx = int(row['tgt'])
        if src_idx < len(nodes_df) and tgt_idx < len(nodes_df):
            src_gene = nodes_df.iloc[src_idx]['gene']
            tgt_gene = nodes_df.iloc[tgt_idx]['gene']
            G.add_edge(src_gene, tgt_gene, weight=row['att'])
    #
    return G


G_ra_full = build_full_graph(ra_nodes, ra_edges_local, ra_symbol_to_module)
G_ctrl_full = build_full_graph(ctrl_nodes, ctrl_edges_local, ctrl_symbol_to_module)

print(f"RA full graph: {G_ra_full.number_of_nodes()} nodes, {G_ra_full.number_of_edges()} edges")
print(f"Control full graph: {G_ctrl_full.number_of_nodes()} nodes, {G_ctrl_full.number_of_edges()} edges")


# ── 3. Filter out small modules (≤ 3 nodes) ──
def filter_small_modules(G, min_size=4):
    if len(G.nodes()) == 0:
        return G
    #
    # Count size per module
    module_sizes = {}
    for node in G.nodes():
        mod = G.nodes[node]['module']
        if mod not in module_sizes:
            module_sizes[mod] = 0
        module_sizes[mod] += 1
    #
    # Keep only nodes in modules with size >= 4
    kept_nodes = set()
    for node in G.nodes():
        mod = G.nodes[node]['module']
        if module_sizes.get(mod, 0) >= min_size:
            kept_nodes.add(node)
    #
    return G.subgraph(kept_nodes).copy()

min_size=2
G_ra = filter_small_modules(G_ra_full, min_size=min_size)
G_ctrl = filter_small_modules(G_ctrl_full, min_size=min_size)

print(f"After filtering small modules:")
print(f"RA: {G_ra.number_of_nodes()} nodes, {G_ra.number_of_edges()} edges")
print(f"Control: {G_ctrl.number_of_nodes()} nodes, {G_ctrl.number_of_edges()} edges")

# ── 4. Plotting ──
# Prepare colors (normalize across all modules from both graphs)
all_modules = set(ra_symbol_to_module.values()) | set(ctrl_symbol_to_module.values())
if not all_modules:
    all_modules = {-1}  # fallback if no modules

#module_norm = mcolors.Normalize(vmin=min(all_modules), vmax=max(all_modules))
#cmap = cm.get_cmap('tab20')  # good for up to ~20 modules
# ── Dynamically compute modules after filtering ──
def get_filtered_modules(G_list):
    """Return a sorted list of all unique module IDs present in the filtered graphs."""
    all_mods = set()
    for G in G_list:
        all_mods.update([G.nodes[n]['module'] for n in G.nodes()])
    return sorted(all_mods)

# Get modules present after filtering
filtered_modules = get_filtered_modules([G_ra, G_ctrl])
if not filtered_modules:
    filtered_modules = [-1]  # fallback if no modules

# Normalize module IDs to colormap
module_norm = mcolors.Normalize(vmin=min(filtered_modules), vmax=max(filtered_modules))
# Adapt colormap to number of modules
num_modules = len(filtered_modules)
if num_modules <= 10:
    cmap = cm.get_cmap('tab10', num_modules)
elif num_modules <= 20:
    cmap = cm.get_cmap('tab20', num_modules)
else:
    # For many modules, use a continuous colormap
    cmap = cm.get_cmap('viridis', num_modules)

print(f"Number of modules after filtering: {num_modules}")
print(f"Filtered module IDs: {filtered_modules}")

#cmap = cm.get_cmap('tab20b')  # good for up to ~20 modules
#cmap = cm.get_cmap('Set1')
#cmap = cm.get_cmap('Paired')
#cmap = cm.get_cmap('Dark2')
#cmap = cm.get_cmap('Accent')

def plot_graph_clean(G, ax, title, layout_name='kamada_kawai'):
    if len(G.nodes()) == 0:
        ax.text(0.5, 0.5, 'No nodes after filtering', ha='center', va='center')
        ax.set_title(title)
        ax.axis('off')
        return
    #
    # Layout
    if layout_name == 'spring':
        pos = nx.spring_layout(G, seed=42, k=0.2, iterations=150)
    elif layout_name == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    elif layout_name == 'fruchterman_reingold':
        pos = nx.fruchterman_reingold_layout(G, seed=42, k=0.25)
    else:
        pos = nx.spring_layout(G, seed=42)
    #
    # Node sizes
    #node_sizes = [max(40, min(1200, G.nodes[n]['saliency'] * 180)) for n in G.nodes()]
    node_sizes = [max(40, min(900, G.nodes[n]['saliency'] * 140)) for n in G.nodes()]
    #
    # Node colors by module
    node_colors = [cmap(module_norm(G.nodes[n]['module'])) for n in G.nodes()]
    #
    # Edge widths (thinner for all edges)
    edge_widths = [max(0.4, G[u][v]['weight'] * 4.5) for u, v in G.edges()]
    #
    # Draw
    nx.draw_networkx_nodes(G, pos,
                           node_size=node_sizes,
                           node_color=node_colors,
                           ax=ax,
                           edgecolors='gray',
                           linewidths=0.4,
                           alpha=0.95)
    nx.draw_networkx_edges(G, pos,
                           width=edge_widths,
                           arrows=True,
                           ax=ax,
                           alpha=0.55,
                           arrowsize=8,
                           connectionstyle='arc3,rad=0.08')
    nx.draw_networkx_labels(G, pos,
                            labels={n: n for n in G.nodes()},
                            font_size=5.8,
                            font_weight='normal',
                            ax=ax,
                            font_family='Arial')
    ax.set_title(f"{title} – All edges – Modules ≥ 4 genes – Layout: {layout_name}", fontsize=14, pad=15)
    ax.axis('off')


# ── Choose layout ──
layout_choice = 'kamada_kawai'

# ── Save ──
output_path = f"/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/plots/GRN_top125_{layout_choice}.pdf"

with PdfPages(output_path) as pdf:
    fig, axs = plt.subplots(1, 2, figsize=(26, 13))
    plot_graph_clean(G_ra, axs[0], 'RA Graph (top 250 genes, all edges, modules ≥ 4)', layout_choice)
    plot_graph_clean(G_ctrl, axs[1], 'Control Graph (top 250 genes, all edges, modules ≥ 4)', layout_choice)
    plt.tight_layout()
    pdf.savefig(fig, dpi=180)
    plt.close(fig)

print(f"PDF saved: {output_path}")
print(f"RA after filtering: {G_ra.number_of_nodes()} nodes, {G_ra.number_of_edges()} edges")
print(f"Control after filtering: {G_ctrl.number_of_nodes()} nodes, {G_ctrl.number_of_edges()} edges")

# ── 1. Helper: Get module → list of (symbol, saliency) for each condition ──
def get_modules_from_graph(G, condition_prefix):
    module_dict = defaultdict(list)
    for node in G.nodes():
        mod_id = G.nodes[node]['module']
        saliency = G.nodes[node]['saliency']
        module_dict[mod_id].append((node, saliency))  # node = symbol here
    #
    # Sort each module by saliency descending
    sorted_modules = {}
    for mod_id, genes in module_dict.items():
        sorted_genes = sorted(genes, key=lambda x: x[1], reverse=True)
        sorted_modules[mod_id] = [gene[0] for gene in sorted_genes]  # only symbols
    #
    # Create nicely named columns
    module_data = {}
    for old_mod_id, genes in sorted(sorted_modules.items(), key=lambda x: x[0]):
        new_col_name = f"{condition_prefix}_GP_{old_mod_id}"
        module_data[new_col_name] = genes
    #
    return module_data

# ── 2. Extract modules from both graphs ──
ra_module_data = get_modules_from_graph(G_ra, "RA")
ctrl_module_data = get_modules_from_graph(G_ctrl, "CTRL")
# ── 3. Combine into one big dictionary ──
all_module_data = {**ra_module_data, **ctrl_module_data}
# ── 4. Create DataFrame ──
# Find the longest module to set the number of rows
max_length = max(len(genes) for genes in all_module_data.values()) if all_module_data else 0
# Create DataFrame with NaN padding
modules_df = pd.DataFrame(
    {col: pd.Series(genes) for col, genes in all_module_data.items()}
)
# Optional: sort columns by condition then by module number (RA first, then CTRL)
ra_cols = sorted([c for c in modules_df.columns if c.startswith("RA_")],
                 key=lambda x: int(x.split('_')[-1]))
ctrl_cols = sorted([c for c in modules_df.columns if c.startswith("CTRL_")],
                   key=lambda x: int(x.split('_')[-1]))
sorted_columns = ra_cols + ctrl_cols
modules_df = modules_df[sorted_columns]
# Optional: make it look nicer by filling NaN with empty string
modules_df = modules_df.fillna("")
print(f"Module DataFrame created with {len(sorted_columns)} columns and {len(modules_df)} rows")
print(modules_df.head(12))  # preview first 12 rows
# ── 5. Save to file (choose your preferred path) ──
output_csv = f"/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/plots/GRN_top125_composition.csv"
modules_df.to_csv(output_csv, index=False)
print(f"Saved to: {output_csv}")

import seaborn as sns
# ── 1. Separate RA and CTRL modules ──
ra_cols = [col for col in modules_df.columns if col.startswith('RA_')]
ctrl_cols = [col for col in modules_df.columns if col.startswith('CTRL_')]
# Get gene sets for each module (drop empty/NaN)
ra_module_sets = {col: set(modules_df[col].dropna()) - {''} for col in ra_cols}
ctrl_module_sets = {col: set(modules_df[col].dropna()) - {''} for col in ctrl_cols}
# ── 2. Compute Jaccard Index for every pair ──
def jaccard_index(set1, set2):
    if not set1 and not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0

# Create matrix: rows=RA modules, columns=CTRL modules
ra_labels = sorted(ra_cols, key=lambda x: int(x.split('_')[-1]))
ctrl_labels = sorted(ctrl_cols, key=lambda x: int(x.split('_')[-1]))
jaccard_matrix = np.zeros((len(ra_labels), len(ctrl_labels)))
for i, ra_mod in enumerate(ra_labels):
    for j, ctrl_mod in enumerate(ctrl_labels):
        jaccard_matrix[i, j] = jaccard_index(ra_module_sets[ra_mod], ctrl_module_sets[ctrl_mod])

# ── 3. Create DataFrame for heatmap ──
jaccard_df = pd.DataFrame(jaccard_matrix, index=ra_labels, columns=ctrl_labels)
# ── 4. Plot Heatmap ──
output_pdf = f"/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/plots/GRN_top125_jaccard_heatmap.pdf"
with PdfPages(output_pdf) as pdf:
    fig, ax = plt.subplots(figsize=(max(8, len(ctrl_labels) * 0.5), max(6, len(ra_labels) * 0.5)))
    sns.heatmap(jaccard_df,
                annot=True,  # Show values in cells
                fmt=".2f",  # Format to 2 decimals
                vmin= 0,
                vmax= 1,
                cmap="YlGnBu",  # Color map (yellow low → blue high)
                cbar_kws={'label': 'Jaccard Index'},
                linewidths=0.5,  # Grid lines
                ax=ax)
    ax.set_title("Jaccard Similarity Between RA and Control Gene Modules", fontsize=14)
    ax.set_xlabel("Control Modules")
    ax.set_ylabel("RA Modules")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)

print(f"Jaccard heatmap saved to: {output_pdf}")










#-----------------------------------------------------
#
# GENE PROGRAMS ANNOTATION
#
#-----------------------------------------------------

import pandas as pd
import numpy as np
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from scipy.stats import pearsonr
import requests
import io

# Step 2: For each gene program (column in gene_table), get top genes
gene_table = pd.read_csv(f"/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/plots/GRN_leiden_top125_composition.csv")
programs = gene_table.columns.tolist()
program_top_genes = {}
for prog in programs:
    top_genes = gene_table[prog].dropna().tolist()
    program_top_genes[prog] = set(top_genes)

# Universe: all genes in the data (from adata.var_names or scores_df.columns)
universe_genes = set(all_genes["gene"]) # or set(scores_df.columns)



#-----------------------------------------------------
#                TRANSCRIPTION FACTORS
#-----------------------------------------------------

def download_and_parse_msigdb():
    url = 'https://data.broadinstitute.org/gsea-msigdb/msigdb/release/7.1/c3.tft.v7.1.symbols.gmt'
    response = requests.get(url)
    tf_targets = {}
    for line in response.text.splitlines():
        parts = line.strip().split('\t')
        tf = parts[0]
        targets = parts[2:]  # skip description
        tf_targets[tf] = targets
    return tf_targets

msigdb_tfs = download_and_parse_msigdb()
combined_tf_targets = {}
for d in [msigdb_tfs]:
    for tf, targets in d.items():
        if tf not in combined_tf_targets:
            combined_tf_targets[tf] = set()
        combined_tf_targets[tf].update(targets)

# Step 3: Enrichment analysis per program
results = []
for prog, prog_genes in program_top_genes.items():
    prog_genes = prog_genes & universe_genes  # intersect with universe
    if len(prog_genes) == 0:
        continue
    #
    for tf, tf_targets in combined_tf_targets.items():
        tf_targets = tf_targets & universe_genes
        if len(tf_targets) == 0:
            continue
        #
        overlap = prog_genes & tf_targets
        overlap_size = len(overlap)
        if overlap_size < 2:
            continue
        #
        # Fisher's exact test
        a = overlap_size
        b = len(tf_targets) - overlap_size
        c = len(prog_genes) - overlap_size
        d = len(universe_genes) - len(prog_genes) - len(tf_targets) + overlap_size
        oddsratio, pvalue = fisher_exact([[a, b], [c, d]], alternative='greater')
        results.append({
            'Program': prog,
            'TF': tf,
            'Overlap_size': overlap_size,
            'Overlap_genes': ','.join(sorted(overlap)),
            'pvalue': pvalue
        })

enrich_df = pd.DataFrame(results)
# FDR correction
if not enrich_df.empty:
    enrich_df_tf['FDR'] = multipletests(enrich_df['pvalue'], method='fdr_bh')[1]
    enrich_df = enrich_df[enrich_df['FDR'] < 0.1]
else:
    print("No enrichments found.")
    enrich_df = pd.DataFrame()

enrich_df_tf = enrich_df.copy()



#-----------------------------------------------------
#                     PATHWAYS
#-----------------------------------------------------

def download_and_parse_pathways():
    """
    Downloads and parses the canonical pathways (C2:CP) collection from MSigDB.
    This includes KEGG, Reactome, WikiPathways, etc. — very useful for general pathway annotation.

    Returns:
        dict: {pathway_name: [list of gene symbols]}
    """
    url = 'https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2023.2.Hs/c2.cp.v2023.2.Hs.symbols.gmt'
    response = requests.get(url)
    tf_targets = {}
    for line in response.text.splitlines():
        parts = line.strip().split('\t')
        tf = parts[0]
        targets = parts[2:]  # skip description
        tf_targets[tf] = targets
    return tf_targets

msigdb_tfs = download_and_parse_pathways()
combined_tf_targets = {}
for d in [msigdb_tfs]:
    for tf, targets in d.items():
        if tf not in combined_tf_targets:
            combined_tf_targets[tf] = set()
        combined_tf_targets[tf].update(targets)

# Step 3: Enrichment analysis per program
results = []
for prog, prog_genes in program_top_genes.items():
    prog_genes = prog_genes & universe_genes  # intersect with universe
    if len(prog_genes) == 0:
        continue
    #
    for tf, tf_targets in combined_tf_targets.items():
        tf_targets = tf_targets & universe_genes
        if len(tf_targets) == 0:
            continue
        #
        overlap = prog_genes & tf_targets
        overlap_size = len(overlap)
        if overlap_size < 2:
            continue
        #
        # Fisher's exact test
        a = overlap_size
        b = len(tf_targets) - overlap_size
        c = len(prog_genes) - overlap_size
        d = len(universe_genes) - len(prog_genes) - len(tf_targets) + overlap_size
        oddsratio, pvalue = fisher_exact([[a, b], [c, d]], alternative='greater')
        results.append({
            'Program': prog,
            'TF': tf,
            'Overlap_size': overlap_size,
            'Overlap_genes': ','.join(sorted(overlap)),
            'pvalue': pvalue
        })

enrich_df = pd.DataFrame(results)
# FDR correction
if not enrich_df.empty:
    enrich_df['FDR'] = multipletests(enrich_df['pvalue'], method='fdr_bh')[1]
    enrich_df = enrich_df[enrich_df['FDR'] < 0.1]
else:
    print("No enrichments found.")
    enrich_df = pd.DataFrame()

enrich_df_pathways = enrich_df.copy()



#-----------------------------------------------------
#                     HALLMARKS
#-----------------------------------------------------

def download_and_parse_immune_hallmarks():
    """
    Downloads and parses the immunologic signatures (C7) collection from MSigDB.
    These are perturbation-induced gene sets from immune cells, cytokines, infections, etc.
    Extremely useful for immune/autoimmune contexts.

    Returns:
        dict: {signature_name: [list of gene symbols]}
    """
    url = 'https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2023.2.Hs/c7.all.v2023.2.Hs.symbols.gmt'
    response = requests.get(url)
    tf_targets = {}
    for line in response.text.splitlines():
        parts = line.strip().split('\t')
        tf = parts[0]
        targets = parts[2:]  # skip description
        tf_targets[tf] = targets
    return tf_targets


msigdb_tfs = download_and_parse_immune_hallmarks()
combined_tf_targets = {}
for d in [msigdb_tfs]:
    for tf, targets in d.items():
        if tf not in combined_tf_targets:
            combined_tf_targets[tf] = set()
        combined_tf_targets[tf].update(targets)

# Step 3: Enrichment analysis per program
results = []
for prog, prog_genes in program_top_genes.items():
    prog_genes = prog_genes & universe_genes  # intersect with universe
    if len(prog_genes) == 0:
        continue
    #
    for tf, tf_targets in combined_tf_targets.items():
        tf_targets = tf_targets & universe_genes
        if len(tf_targets) == 0:
            continue
        #
        overlap = prog_genes & tf_targets
        overlap_size = len(overlap)
        if overlap_size < 2:
            continue
        #
        # Fisher's exact test
        a = overlap_size
        b = len(tf_targets) - overlap_size
        c = len(prog_genes) - overlap_size
        d = len(universe_genes) - len(prog_genes) - len(tf_targets) + overlap_size
        oddsratio, pvalue = fisher_exact([[a, b], [c, d]], alternative='greater')
        results.append({
            'Program': prog,
            'TF': tf,
            'Overlap_size': overlap_size,
            'Overlap_genes': ','.join(sorted(overlap)),
            'pvalue': pvalue
        })

enrich_df = pd.DataFrame(results)
# FDR correction
if not enrich_df.empty:
    enrich_df['FDR'] = multipletests(enrich_df['pvalue'], method='fdr_bh')[1]
    enrich_df = enrich_df[enrich_df['FDR'] < 0.1]
else:
    print("No enrichments found.")
    enrich_df = pd.DataFrame()

enrich_df_hallmarks = enrich_df.copy()





import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#enrich_df = enrich_df_tf.loc[enrich_df_tf.Program.str.contains("RA"),:]
#enrich_df = enrich_df_pathways.loc[enrich_df_pathways.Program.str.contains("RA"),:]
enrich_df = enrich_df_hallmarks.loc[enrich_df_hallmarks.Program.str.contains("RA"),:]
# 1. Data Processing
# Calculate -log10(FDR) for the color scale
enrich_df['-log10FDR'] = -np.log10(enrich_df['FDR'])
# Sort by Program and then by Significance to keep the plot organized
enrich_df = enrich_df.sort_values(['Program', '-log10FDR'], ascending=[True, True])
# 2. Setup Figure Aesthetics
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.sans-serif'] = 'Arial'
sns.set_style("whitegrid")
# Dynamic height based on number of TFs
fig_height = max(6, len(enrich_df['TF'].unique()) * 0.3)
fig, ax = plt.subplots(figsize=(8, fig_height))
# 3. Create the Bubble Plot
scatter = ax.scatter(
    x=enrich_df['Program'],
    y=enrich_df['TF'],
    s=enrich_df['Overlap_size'] * 40, # Scale size for visibility
    c=enrich_df['-log10FDR'],
    cmap='Reds', # 'Reds' or 'magma' are great for significance
    edgecolors='black',
    linewidth=0.5,
    alpha=0.9
)
# 4. Add Legends
# Colorbar for Significance
cbar = plt.colorbar(scatter, ax=ax, shrink=0.5)
cbar.set_label('$-log_{10}(FDR)$', fontsize=10, fontweight='bold')
# Size Legend (Manual)
# We create dummy points for the legend to show what Overlap_size means
handles, labels = [], []
for sz in [2, 5, 10]: # Adjust these based on your actual data range
    handles.append(plt.scatter([], [], s=sz*40, color='gray', edgecolors='black', alpha=0.6))
    labels.append(str(sz))

legend_sizes = [2, 5, 10]
ax.legend(
    handles,
    [str(sz) for sz in legend_sizes],
    title="Overlap Size",
    title_fontsize=10,
    loc='upper left',
    bbox_to_anchor=(1.02, 1), # Move legend outside to the right
    frameon=False,
    labelspacing=1.5,   # <-- Increase this if they still overlap
    handletextpad=1.2,  # <-- Space between circle and number
    borderpad=1.0       # <-- Internal padding
)
# 5. Final Formatting
ax.set_xlabel("Gene Program", fontsize=11, fontweight='bold')
ax.set_ylabel("Transcription Factor / Target Set", fontsize=11, fontweight='bold')
ax.set_title("TF Enrichment by Gene Program", loc='left', fontsize=14, pad=20)
# Rotate X-labels for better fit
plt.xticks(rotation=45, ha='right')
# Clean spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
# Save
#output_pdf = r"C:\Users\Inigo\Desktop\FloREN3.0\binvignat_outputs\plots\Enrichment_DotPlot_TF.pdf"
#output_pdf = r"C:\Users\Inigo\Desktop\FloREN3.0\binvignat_outputs\plots\Enrichment_DotPlot_Pathwyas.pdf"
output_pdf = r"C:\Users\Inigo\Desktop\FloREN3.0\binvignat_outputs\plots\Enrichment_DotPlot_Hallmarks.pdf"
plt.savefig(output_pdf, bbox_inches='tight')










#-----------------------------------------------------
#
# CELLS ATTENTION UMAP
#
#-----------------------------------------------------
import scanpy as sc
import os
import numpy as np
import pandas as pd

result_dir = '/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs'  # Sets parent directory for output
inter_dir = result_dir + '/interpretability/'  # Sets directory for loss output
patients = [d for d in os.listdir(inter_dir) if os.path.isdir(os.path.join(inter_dir, d))]
cell_names = pd.read_csv(os.path.join(result_dir, "All_AUC_Cell_names.csv"))

data_path = "C:/Users/Inigo/Desktop/FloREN/Binvignat/output.h5ad"
adata = sc.read_h5ad(data_path)
meta = adata.obs[['cell_type','donor_id']]
meta["barcode"]  = meta.donor_id.astype(str)+"__"+meta.index
meta.columns = ["celltype", "patient","barcode"]

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
meta["full_id"] = meta.barcode
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

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(
    adata,
    n_top_genes=2000,
    #min_mean=0.0125,
    #max_mean=3,
    #min_disp=0.5,
    flavor="seurat_v3",
)
adata.layers["scaled"] = adata.X.toarray()
#sc.pp.scale(adata, max_value=10, layer="scaled")
sc.pp.pca(adata, layer="scaled", svd_solver="arpack")
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30)
#sc.tl.paga(adata)
#sc.pl.paga(adata, plot=False)  # remove `plot=False` if you want to see the coarse-grained graph
#sc.tl.umap(adata)
sc.tl.umap(adata)
adata.obsm['X_umap'] = adata.obsm['UMAP']
adata.obsm["X_umap"] = adata.obsm["UMAP"].values

# Create clean export dataframe
df_attention = pd.DataFrame({
    "barcode": adata.obs_names,
    "attention_z": adata.obs["attention_z"].values
})
# Safety check
assert df_attention["barcode"].is_unique
assert len(df_attention) == adata.n_obs

out_path = "C:/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/floren_cells_attention_scores.csv"
df_attention.to_csv(out_path, index=False)
#
# import scanpy as sc
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages
# # Make plots PDF-friendly (vectorized text)
# sc.set_figure_params(dpi=100, dpi_save=300, format='pdf', vector_friendly=True)
# # Output path
# pdf_path = "C:/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/plots/umap_attention_summary.pdf"
# with PdfPages(pdf_path) as pdf:
#     fig, axes = plt.subplots(1, 3, figsize=(18, 6))
#     # ── LEFT: SLE_status ──
#     sc.pl.umap(
#         adata,
#         color="disease",
#         ax=axes[0],
#         show=False,
#         frameon=False,
#         title="Disease Status",
#         size=10
#     )
#     # ── MIDDLE: cell type ──
#     sc.pl.umap(
#         adata,
#         color="cell_type",
#         ax=axes[1],
#         show=False,
#         frameon=False,
#         title="Main Cell Type",
#         size=10
#     )
#     # ── RIGHT: attention ──
#     sc.pl.umap(
#         adata,
#         #color="attention",
#         #color="attention_log",
#         #color="attention_sqrt",
#         color="attention_z",
#         #color="attention_rank",
#         ax=axes[2],
#         show=False,
#         frameon=False,
#         title="Attention",
#         cmap="viridis",
#         size=10
#     )
#     #sc.pl.umap(
#     #    adata,
#     #    color="attention",
#     #    cmap="viridis",
#     #    vmin=vmin,
#     #    vmax=vmax,
#     #    sort_order=True,
#     #    ax=axes[2],
#     #    show=False,
#     #    frameon=False,
#     #    title="Attention (clipped)"
#     #)
#     plt.tight_layout()
#     pdf.savefig(fig, bbox_inches="tight")
#     plt.close(fig)










#-----------------------------------------------------
#
# FINAL POWERFUL VISUALIZATION
#
#-----------------------------------------------------

# celltype_attention_matrix
# celltype_attention_matrix.to_csv("C:/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/viz_celltype_attention_matrix.csv")
# gene_results
# gene_results.to_csv("C:/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/viz_gene_results.csv")
# mean_celltype_saliency
# mean_celltype_saliency.to_csv("C:/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/viz_mean_celltype_saliency.csv")
# ra_edges_local
# ra_edges_local.to_csv("C:/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/viz_ra_edges_local.csv")
# ra_gene_map
# ra_gene_map.to_csv("C:/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/viz_ra_gene_map.csv")

import pandas as pd
import numpy as np
import os

all_celltype_edges = []
for patient in patients:
    print(f"Processing patient: {patient}")
    condition = patient_condition(patient)
    if condition not in ["RA", "Control"]:
        continue
    #
    # ── Load edges ──
    df2 = pd.read_csv(
        os.path.join(att_dir, f"{patient}_edge_att.csv"),
        delimiter=","
    ).iloc[:, 1:]
    df2.columns = ["src", "tgt", "att"]
    # ── Build cell index → celltype mapping ──
    patient_mask = cell_names["0"].str.startswith(patient + "__")
    patient_cell_names = cell_names.loc[patient_mask, "0"].reset_index(drop=True)
    cell_df = pd.DataFrame({
        "local_cell_index": np.arange(len(patient_cell_names)),
        "barcode_full": patient_cell_names
    })
    cell_df["barcode"] = cell_df["barcode_full"]
    cell_df["patient"] = patient
    # Node index = offset by n_genes
    cell_df["node_index"] = cell_df["local_cell_index"] + n_genes
    # Add celltype
    cell_df = cell_df.merge(
        meta[["barcode", "celltype", "patient"]],
        on=["barcode", "patient"],
        how="left"
    )
    # Create mapping: node_index → celltype
    node_to_celltype = dict(zip(cell_df["node_index"], cell_df["celltype"]))
    # ── Keep ONLY cell–cell edges ──
    df_cells = df2[
        (df2["src"] >= n_genes) &
        (df2["tgt"] >= n_genes)
        ].copy()
    # ── Map to celltypes ──
    df_cells["celltype_src"] = df_cells["src"].map(node_to_celltype)
    df_cells["celltype_tgt"] = df_cells["tgt"].map(node_to_celltype)
    # Remove missing mappings
    df_cells = df_cells.dropna(subset=["celltype_src", "celltype_tgt"])
    # ── Aggregate within patient ──
    agg = df_cells.groupby(
        ["celltype_src", "celltype_tgt"],
        as_index=False
    )["att"].sum()
    agg["patient"] = patient
    agg["condition"] = condition
    all_celltype_edges.append(agg)

# ── Combine all patients ──
all_celltype_edges = pd.concat(all_celltype_edges, ignore_index=True)

print("Total aggregated edges:", len(all_celltype_edges))

celltype_edge_mean = all_celltype_edges.groupby(
    ["condition", "celltype_src", "celltype_tgt"],
    as_index=False
)["att"].mean()

# Split conditions
ra_cell_edges = celltype_edge_mean.query("condition == 'RA'").copy()
ctrl_cell_edges = celltype_edge_mean.query("condition == 'Control'").copy()

# Remove self-loops (optional)
ra_cell_edges = ra_cell_edges[ra_cell_edges["celltype_src"] != ra_cell_edges["celltype_tgt"]]
ctrl_cell_edges = ctrl_cell_edges[ctrl_cell_edges["celltype_src"] != ctrl_cell_edges["celltype_tgt"]]

print("RA celltype edges:", len(ra_cell_edges))
print("Control celltype edges:", len(ctrl_cell_edges))

celltypes = sorted(mean_celltype_saliency.index)
def build_matrix(edges, celltypes):
    mat = pd.DataFrame(
        0,
        index=celltypes,
        columns=celltypes,
        dtype=float
    )
    for _, row in edges.iterrows():
        mat.loc[row["celltype_src"], row["celltype_tgt"]] = row["att"]
    #
    return mat

ra_matrix = build_matrix(ra_cell_edges, celltypes)
ctrl_matrix = build_matrix(ctrl_cell_edges, celltypes)

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
pdf_path = "C:/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/plots/celltype_attention_heatmaps.pdf"
# ── Define shared scale ──
vmin = 0
vmax = max(ctrl_matrix.values.max(), ra_matrix.values.max())
with PdfPages(pdf_path) as pdf:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # ── LEFT: CONTROL ──
    sns.heatmap(
        ctrl_matrix,
        ax=axes[0],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        cbar=False,
        xticklabels=False,
        yticklabels=True
    )
    axes[0].set_title("Control")
    axes[0].tick_params(axis='y', labelsize=3)
    # ── RIGHT: RA ──
    sns.heatmap(
        ra_matrix,
        ax=axes[1],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        cbar=True,
        xticklabels=False,
        yticklabels=False
    )
    axes[1].set_title("RA")
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


import numpy as np

# ── Gene scores (column_means assumed to exist) ──
gene_scores = column_means.copy()
# Keep only genes in RA network
gene_scores = gene_scores[gene_scores.index.isin(ra_nodes["gene"])]
# Normalize 0–1
gene_scores_norm = (gene_scores - gene_scores.min()) / (gene_scores.max() - gene_scores.min())
# ── Celltype scores ──
cell_scores = mean_celltype_saliency.copy()
cell_scores_norm = (cell_scores - cell_scores.min()) / (cell_scores.max() - cell_scores.min())

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# --- Copy your inputs ---
gene_scores = gene_scores_norm.copy()
cell_scores = cell_scores_norm.copy()
edges_df = ct_gene_mat.copy()

# --- Ensure no zero sizes (add 0.01) ---
#gene_scores = gene_scores[gene_scores>0]
#cell_scores = cell_scores[cell_scores>0]

# --- Normalize node sizes (0–1) ---
#gene_scores = (gene_scores - gene_scores.min()) / (gene_scores.max() - gene_scores.min())
#cell_scores = (cell_scores - cell_scores.min()) / (cell_scores.max() - cell_scores.min())

# --- Normalize edge weights (0–1) ---
edges_norm = (edges_df - edges_df.min().min()) / (edges_df.max().max() - edges_df.min().min())
#edges_norm[edges_norm<0.5] = 0

# --- Build graph ---
G = nx.Graph()

# Add cell nodes
for cell in edges_norm.index:
    G.add_node(cell, bipartite='cell', size=cell_scores[cell])

# Add gene nodes
for gene in edges_norm.columns:
    G.add_node(gene, bipartite='gene', size=gene_scores[gene])

# Add edges (only if weight > 0)
for cell in edges_norm.index:
    for gene in edges_norm.columns:
        weight = edges_norm.loc[cell, gene]
        if weight > 0:
            G.add_edge(cell, gene, weight=weight)

edge_threshold = np.percentile(edges_norm.values.flatten(), 95)
edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < edge_threshold]
G.remove_edges_from(edges_to_remove)
# --- Normalize ra_matrix to 0-1 ---
ra_norm = (ra_matrix - ra_matrix.min().min()) / (ra_matrix.max().max() - ra_matrix.min().min())
# --- Add cell-cell edges to G ---
edge_threshold = 0.5  # optional threshold to filter weak connections
for cell1 in ra_norm.index:
    for cell2 in ra_norm.columns:
        if cell1 == cell2:
            continue  # skip self-loops
        weight = ra_norm.loc[cell1, cell2]
        if weight > edge_threshold:
            # Add or update edge (if already exists, we can sum or take max)
            if G.has_edge(cell1, cell2):
                G[cell1][cell2]['weight'] = max(G[cell1][cell2]['weight'], weight)
            else:
                G.add_edge(cell1, cell2, weight=weight)

# Keep strongest edges only (IMPORTANT)
threshold = ra_edges_local["att"].quantile(0.5)
for _, row in ra_edges_local.iterrows():
    if row["att"] < threshold:
        continue
    #
    g1 = ra_gene_map.get(row["src"])
    g2 = ra_gene_map.get(row["tgt"])
    if g1 and g2:
        G.add_edge(
            g1,
            g2,
            weight=row["att"],
            type="gene-gene"
        )

# --- Remove nodes with size == 0 (after edges removed) ---
nodes_to_remove = [n for n, d in G.nodes(data=True) if d['size'] == 0]
G.remove_nodes_from(nodes_to_remove)
# --- Also remove any new isolates created by edge removal ---
G.remove_nodes_from(list(nx.isolates(G)))
non_orphan_nodes = [node for node, degree in dict(G.degree()).items() if degree > 0]
G_sub = G.subgraph(non_orphan_nodes).copy()


import networkx as nx

# Total nodes and edges
print("Total nodes in G:", G_sub.number_of_nodes())
print("Total edges in G:", G_sub.number_of_edges())


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

def normalize_pos(pos_dict, xmin=0.05, xmax=0.95, ymin=0.05, ymax=0.95):
    """Rescale any layout to a fixed [0,1] box — prevents clipping."""
    xs = np.array([v[0] for v in pos_dict.values()])
    ys = np.array([v[1] for v in pos_dict.values()])
    # Guard against degenerate (all-same) positions
    xr = xs.max() - xs.min()
    yr = ys.max() - ys.min()
    if xr < 1e-6: xr = 1.0
    if yr < 1e-6: yr = 1.0
    nodes = list(pos_dict.keys())
    xs_n = xmin + (xs - xs.min()) / xr * (xmax - xmin)
    ys_n = ymin + (ys - ys.min()) / yr * (ymax - ymin)
    return {n: (xs_n[i], ys_n[i]) for i, n in enumerate(nodes)}


def build_layout(G, cell_nodes, gene_nodes):
    """
    Deterministic two-ring layout.
    Cells on outer ring, genes on inner ring.
    Sorts by degree so high-degree nodes face each other.
    """
    cell_sorted = sorted(cell_nodes,  key=lambda n: -G.degree(n))
    gene_sorted = sorted(gene_nodes,  key=lambda n: -G.degree(n))
    pos = {}
    nc = len(cell_sorted)
    for i, n in enumerate(cell_sorted):
        a = 2 * np.pi * i / nc - np.pi / 2
        pos[n] = (np.cos(a) * 2.8, np.sin(a) * 2.8)
    ng = len(gene_sorted)
    for i, n in enumerate(gene_sorted):
        a = 2 * np.pi * i / ng - np.pi / 2 + np.pi / ng
        pos[n] = (np.cos(a) * 1.3, np.sin(a) * 1.3)
    return pos

def draw_attention_network(G, cell_nodes, gene_nodes,
                           layout='two_ring',
                           figsize=(16, 16),
                           dark=True,
                           save_path='attention_network.png'):
    BG    = '#0F1117' if dark else '#FAFAFA'
    TXTC  = 'white'   if dark else '#111111'
    # ── Layout ──────────────────────────────────────────────────────
    if layout == 'two_ring':
        raw_pos = build_layout(G, cell_nodes, gene_nodes)
    elif layout == 'spring':
        seed = build_layout(G, cell_nodes, gene_nodes)
        raw_pos = nx.spring_layout(G, pos=seed, fixed=None,
                                   k=2.0, iterations=150,
                                   seed=42, weight=None)  # weight=None is key!
    elif layout == 'kamada':
        raw_pos = nx.kamada_kawai_layout(G, weight=None)
    else:
        raw_pos = nx.spectral_layout(G)
    #
    pos = normalize_pos(raw_pos)  # always normalize!
    # ── Node attributes ─────────────────────────────────────────────
    CCOL, CEDGE = '#5B8EE6', '#2C5BB5'
    GCOL, GEDGE = '#E6A23C', '#B07010'
    node_list  = list(G.nodes())
    sizes, colors, edgecs = [], [], []
    for n in node_list:
        d = G.nodes[n]
        s = float(d.get('size', 0.1))
        if n in cell_nodes:
            sizes.append(800  + 1400 * s)
            colors.append(CCOL); edgecs.append(CEDGE)
        else:
            sizes.append(200  + 500  * s)
            colors.append(GCOL); edgecs.append(GEDGE)
    #
    # ── Edge attributes ─────────────────────────────────────────────
    e_col, e_wid, e_alp, e_sty = [], [], [], []
    for u, v, d in G.edges(data=True):
        w = float(d.get('weight', 0.3))
        t = d.get('type', '')
        if t == 'gene-gene':
            e_col.append('#A87CDB'); e_wid.append(0.8 + w*2)
            e_alp.append(0.6);       e_sty.append('solid')
        elif u in cell_nodes and v in cell_nodes:
            e_col.append('#5B9EE6'); e_wid.append(1.0 + w*2.5)
            e_alp.append(0.5);       e_sty.append('solid')
        else:  # cell–gene
            e_col.append('#888888'); e_wid.append(0.4 + w*1.2)
            e_alp.append(0.3);       e_sty.append('solid')
    #
    # ── Draw ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize, facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(-0.05, 1.05)   # explicit limits — critical!
    ax.set_ylim(-0.05, 1.05)
    ax.axis('off')
    # Edges — draw by type so gene-gene renders on top
    for etype, z in [('ct_gene', 1), ('cell_cell', 2), ('gene_gene', 3)]:
        el = []
        for (u, v, d), c, w, a in zip(G.edges(data=True), e_col, e_wid, e_alp):
            t = d.get('type', '')
            is_cc = u in cell_nodes and v in cell_nodes
            if etype == 'gene_gene' and t == 'gene-gene': el.append((u,v,c,w,a))
            elif etype == 'cell_cell' and is_cc and t != 'gene-gene': el.append((u,v,c,w,a))
            elif etype == 'ct_gene' and not is_cc and t != 'gene-gene': el.append((u,v,c,w,a))
        for u, v, c, w, a in el:
            x0,y0 = pos[u]; x1,y1 = pos[v]
            ax.plot([x0,x1],[y0,y1], color=c, lw=w, alpha=a, zorder=z)
    #
    # Nodes
    sc = ax.scatter(
        [pos[n][0] for n in node_list],
        [pos[n][1] for n in node_list],
        s=sizes, c=colors,
        edgecolors=edgecs, linewidths=1.0,
        zorder=5
    )
    #
    # ── Labels ────────────
    cx = np.mean([pos[n][0] for n in node_list])
    cy = np.mean([pos[n][1] for n in node_list])
    #
    for n in node_list:
        x, y = pos[n]
        is_cell = n in cell_nodes
        dx, dy = x - cx, y - cy
        dist = max(np.hypot(dx, dy), 1e-3)
        off  = 0.06 if is_cell else 0.045
        lx = np.clip(x + dx/dist*off, 0.02, 0.98)
        ly = np.clip(y + dy/dist*off, 0.02, 0.98)
        ha = 'left' if dx >= 0 else 'right'
        va = 'bottom' if dy >= 0 else 'top'
        ax.annotate(
            n, xy=(x, y), xytext=(lx, ly),
            fontsize=7.5 if is_cell else 6.0,
            fontweight='semibold' if is_cell else 'normal',
            color=TXTC if is_cell else '#FFD580',
            ha=ha, va=va,
            bbox=dict(boxstyle='round,pad=0.12', fc=BG, ec='none', alpha=0.75),
            zorder=6
        )
    #
    # Degree badge on cell nodes
    #for n in cell_nodes:
    #    ax.text(*pos[n], str(G.degree(n)), fontsize=5.5,
    #            color='white', ha='center', va='center',
    #            fontweight='bold', zorder=7)
    #
    # ── Legend ───────────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(fc=CCOL, ec=CEDGE, lw=1.2,
                       label='Cell type  (size = saliency score)'),
        mpatches.Patch(fc=GCOL, ec=GEDGE, lw=1.2,
                       label='Gene  (size = attention score)'),
        Line2D([0],[0], color='#888888', lw=1.2, alpha=0.6,
               label='Cell–gene attention'),
        Line2D([0],[0], color='#5B9EE6', lw=1.5, alpha=0.6,
               label='Cell–cell co-attention'),
        Line2D([0],[0], color='#A87CDB', lw=1.5, alpha=0.8,
               label='Gene–gene interaction'),
    ]
    ax.legend(handles=legend_elements, loc='lower left',
              framealpha=0.7, facecolor='#1A1D27',
              edgecolor='#333', fontsize=8.5,
              labelcolor=TXTC, title='Node / Edge types',
              title_fontsize=8.5)
    ax.set_title(
        f'GNN Attention Network — Single Cell Transcriptomics\n'
        f'({G.number_of_nodes()} nodes · {G.number_of_edges()} edges · '
        f'{len(cell_nodes)} cell types · {len(gene_nodes)} genes)',
        color=TXTC, fontsize=13, pad=14
    )
    #plt.tight_layout(pad=1.5)
    #plt.savefig(save_path, format="pdf",dpi=300, bbox_inches='tight', facecolor=BG)
    #plt.show()
    fig.patch.set_facecolor('#FFFFFF')  # force figure patch to pure white
    ax.set_facecolor('#FFFFFF')  # force axes background to pure white
    plt.tight_layout(pad=1.5)
    plt.savefig(save_path, format="pdf", dpi=300, bbox_inches='tight',
                facecolor='#FFFFFF',  # explicit hex, not BG variable
                edgecolor='none')  # remove any figure border
    print(f"Saved → {save_path}")


# ── Run it ────────────────────────────────────────────────────────
draw_attention_network(
    G_sub, cell_nodes, gene_nodes,
    layout='kamada',   # start here, guaranteed to work
    figsize=(16, 16),
    dark=False,
    save_path='attention_network_fixed_kamada.pdf'
)










#-----------------------------------------------------
#
# THE MOST POWERFUL VISUALIZATION
#
#-----------------------------------------------------


import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase

def draw_attention_network_heatmap(G, cell_nodes, gene_nodes,
                                   layout='kamada',
                                   figsize=(16, 16),
                                   save_path='attention_network_heatmap.pdf'):
    """
    Same layout as draw_attention_network, but:
    - No node labels
    - Node color  = score mapped to blue→red (RdBu_r)
    - Node size   = score (unchanged)
    - Edge color  = mean score of its two endpoints, same colormap
    - Edge width  = weight (unchanged)
    - Separate colorbars for cell-type scores and gene scores
    """
    CMAP = cm.get_cmap('RdBu_r')   # blue=low, red=high
    # ── Layout ──────────────────────────────────────────────────────
    if layout == 'two_ring':
        raw_pos = build_layout(G, cell_nodes, gene_nodes)
    elif layout == 'spring':
        seed = build_layout(G, cell_nodes, gene_nodes)
        raw_pos = nx.spring_layout(G, pos=seed, fixed=None,
                                   k=2.0, iterations=150,
                                   seed=42, weight=None)
    elif layout == 'kamada':
        raw_pos = nx.kamada_kawai_layout(G, weight=None)
    else:
        raw_pos = nx.spectral_layout(G)
    #
    pos = normalize_pos(raw_pos)
    # ── Score lookup (already normalized 0-1) ───────────────────────
    def node_score(n):
        return float(G.nodes[n].get('size', 0.0))
    #
    # ── Node attributes ─────────────
    node_list = list(G.nodes())
    sizes, colors, edgecs = [], [], []
    for n in node_list:
        s = node_score(n)
        rgba = CMAP(s)
        # Darken the edge color slightly for contrast
        dark = tuple(max(0, c * 0.65) for c in rgba[:3]) + (1.0,)
        if n in cell_nodes:
            sizes.append(800  + 1400 * s)
        else:
            sizes.append(200  + 500  * s)
        colors.append(rgba)
        edgecs.append(dark)
    #
    # ── Edge attributes ──────────────────────────────────────────────
    e_col, e_wid, e_alp = [], [], []
    for u, v, d in G.edges(data=True):
        w    = float(d.get('weight', 0.3))
        t    = d.get('type', '')
        smean = (node_score(u) + node_score(v)) / 2.0
        rgba  = CMAP(smean)
        if t == 'gene-gene':
            e_col.append(rgba); e_wid.append(0.8 + w * 2);   e_alp.append(0.70)
        elif u in cell_nodes and v in cell_nodes:
            e_col.append(rgba); e_wid.append(1.0 + w * 2.5); e_alp.append(0.60)
        else:  # cell–gene
            e_col.append(rgba); e_wid.append(0.4 + w * 1.2); e_alp.append(0.40)
    #
    # ── Figure with room for colorbars on the right ──────────────────
    fig = plt.figure(figsize=figsize, facecolor='#FFFFFF')
    # Main axes takes 80% of width; colorbars share the remaining 20%
    ax      = fig.add_axes([0.02, 0.05, 0.78, 0.88])
    ax_cb1  = fig.add_axes([0.83, 0.55, 0.025, 0.35])  # cell colorbar
    ax_cb2  = fig.add_axes([0.83, 0.12, 0.025, 0.35])  # gene colorbar
    ax.set_facecolor('#FFFFFF')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.axis('off')
    # ── Draw edges ───────────────────────────────────────────────────
    for etype, z in [('ct_gene', 1), ('cell_cell', 2), ('gene_gene', 3)]:
        el = []
        for (u, v, d), c, w, a in zip(G.edges(data=True), e_col, e_wid, e_alp):
            t     = d.get('type', '')
            is_cc = u in cell_nodes and v in cell_nodes
            if   etype == 'gene_gene' and t == 'gene-gene':               el.append((u,v,c,w,a))
            elif etype == 'cell_cell' and is_cc and t != 'gene-gene':     el.append((u,v,c,w,a))
            elif etype == 'ct_gene'   and not is_cc and t != 'gene-gene': el.append((u,v,c,w,a))
        for u, v, c, w, a in el:
            x0, y0 = pos[u]; x1, y1 = pos[v]
            ax.plot([x0,x1],[y0,y1], color=c, lw=w, alpha=a, zorder=z)
    #
    # ── Draw nodes ───────────────────────────────────────────────────
    ax.scatter(
        [pos[n][0] for n in node_list],
        [pos[n][1] for n in node_list],
        s=sizes, c=colors,
        edgecolors=edgecs, linewidths=1.1,
        zorder=5
    )
    # ── Colorbars ────────────────────────────────────────────────────
    norm = mcolors.Normalize(vmin=0, vmax=1)
    sm = cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    cb1 = ColorbarBase(ax_cb1, cmap=CMAP, norm=norm, orientation='vertical')
    cb1.set_label('Cell type saliency score', fontsize=9, color='#111111')
    cb1.ax.yaxis.set_tick_params(color='#111111', labelcolor='#111111', labelsize=8)
    # Draw cell-type marker shapes on colorbar
    ax_cb1.set_title('● Cell type', fontsize=8, color='#111111', pad=6)
    cb2 = ColorbarBase(ax_cb2, cmap=CMAP, norm=norm, orientation='vertical')
    cb2.set_label('Gene attention score', fontsize=9, color='#111111')
    cb2.ax.yaxis.set_tick_params(color='#111111', labelcolor='#111111', labelsize=8)
    ax_cb2.set_title('◆ Gene', fontsize=8, color='#111111', pad=6)
    # ── Edge-width legend ────────────────────────────────────────────
    legend_elements = [
        Line2D([0], [0], color='#888888', lw=0.6, alpha=0.6,
               label='Cell–gene attention  (thin=low w)'),
        Line2D([0], [0], color='#888888', lw=2.5, alpha=0.6,
               label='Cell–gene attention  (thick=high w)'),
        Line2D([0], [0], color='#444444', lw=1.5, alpha=0.7,
               linestyle='solid', label='Cell–cell co-attention'),
        Line2D([0], [0], color='#444444', lw=1.5, alpha=0.7,
               linestyle='solid', label='Gene–gene interaction'),
        mpatches.Patch(fc='none', ec='none', label=''),
        mpatches.Patch(fc='none', ec='none',
                       label='Color: blue=low · red=high score'),
        mpatches.Patch(fc='none', ec='none',
                       label='Size:  small=low · large=high score'),
    ]
    ax.legend(handles=legend_elements, loc='lower left',
              framealpha=0.9, facecolor='#F8F8F8',
              edgecolor='#CCCCCC', fontsize=8,
              labelcolor='#111111', title='Edge width = attention weight',
              title_fontsize=8)
    ax.set_title(
        f'GNN Attention Network — Single Cell Transcriptomics\n'
        f'({G.number_of_nodes()} nodes · {G.number_of_edges()} edges · '
        f'{len(cell_nodes)} cell types · {len(gene_nodes)} genes)',
        color='#111111', fontsize=13, pad=14
    )
    fig.patch.set_facecolor('#FFFFFF')
    plt.savefig(save_path, format="pdf", dpi=300, bbox_inches='tight',
                facecolor='#FFFFFF', edgecolor='none')
    #plt.show()
    print(f"Saved → {save_path}")


# ── Run ───────────────────────────────────────────────────────────
draw_attention_network_heatmap(
    G_sub, cell_nodes, gene_nodes,
    layout='kamada',
    figsize=(16, 16),
    save_path='attention_network_fixed_heatmap.pdf'
)










#-----------------------------------------------------
#
# ATTENTION - SIGNIFICANCE VOLCANO
#
#-----------------------------------------------------

import matplotlib.pyplot as plt

# Make sure the indices match
common_genes = column_means.index.intersection(gene_results.index)
x = gene_results.loc[common_genes, 'distance']
y = column_means.loc[common_genes]

plt.figure(figsize=(8,6))
plt.scatter(x, y, alpha=0.6, edgecolor='k', linewidth=0.3)
plt.xlabel("Gene Distance")
plt.ylabel("Mean Gene Attention")
plt.title("Gene Attention vs Distance")
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig("C:/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/plots/gene_volcano_overall.pdf", format="pdf",dpi=300)





