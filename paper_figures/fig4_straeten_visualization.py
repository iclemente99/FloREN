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
output_path = "C:/Users/Inigo/Desktop/FloREN3.0/straeten_output"
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

data_path = "C:/Users/Inigo/Desktop/FloREN/Straeten/genes"
#count_matrices_path = os.path.join(data_path, "count_matrices")
#count_matrices_path = "/home/inigo/Desktop/FloREN3.0/Binvignat/genes/"
print(f"Looking for CSV files in: {data_path}")
csv_files = glob.glob(os.path.join(data_path, "*.csv"))
print(f"Found {len(csv_files)} CSV files: {csv_files}")

# Load reference for gene names and concatenate all matrices
reference = pd.read_csv(csv_files[0])
reference = reference.set_index("Unnamed: 0")
reference = reference.T
reference = reference.reset_index().rename(columns={"index": "gene"})
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
    transformed_matrix = transformed_matrix.set_index("Unnamed: 0")
    transformed_matrix = transformed_matrix.T
    transformed_matrix = transformed_matrix.reset_index().rename(columns={"index": "gene"})
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
    # cells_f = pd.read_csv(
    #    f"/Users/Inigo/Desktop/FloREN/Binvignat/cell_connections/"
    #    + str.split(load_file, '/genes')[1]
    # )
    # cells_f.set_index(cells_f.columns[0], inplace=True)
    cells_f = pd.DataFrame(np.zeros([gene_cell.shape[1], gene_cell.shape[1]]))
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

################################################################
#
# DEEPMAPS PREPARATION FOR HGT IMPLEMENTATION
#
################################################################

args.epoch = 100  # Hyperparameter specification
args.n_hid = 128  # Hyperparameter specification
args.n_heads = 8  # Hyperparameter specification
# args.n_heads = 13 # Hyperparameter specification
# args.lr = 0.1 # Hyperparameter specification
args.lr = 0.0005  # Hyperparameter specification
args.lr.txt = '01'
# args.n_batch = 32 # Hyperparameter specification
# sample_name = re.split('TFs_',tfs_files[file],2)[-1][:re.split('TFs_',tfs_files[file],2)[-1].index(".")]
sample_name = "FloREN"
file0 = f'sample_{sample_name}_epoch_{args.epoch}_n_hid_{args.n_hid}_nheads_{args.n_heads}_lr_01_n_batch{args.n_batch}'  # Text saving hypermarametrization
# print(f'\n{file0}') #Print hyperparametrization
args.result_dir = '/Users/Inigo/Desktop/FloREN3.0/straeten_output'  # Sets parent directory for output
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

################################################################
#
# SUBGRAPHS SAMPLING FOR TRAINING
#
################################################################

metadata = pd.read_csv("/Users/Inigo/Desktop/FloREN/Straeten/samples_metadata.csv")
metadata.columns = ["Unnamed", "patient_id", "group", "age", "sex", "batch"]
metadata["group"] = metadata["group"].astype("category").cat.codes


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

################################################################
#
# TRAIN HGT IN SUB-SAMPLING
#
################################################################


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
        transformed_matrix = transformed_matrix.set_index("Unnamed: 0")
        transformed_matrix = transformed_matrix.T
        transformed_matrix = transformed_matrix.reset_index().rename(columns={"index": "gene"})
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
            #
            node_num = 0
            types = graph.get_types()
            for t in types:
                node_dict[t] = [node_num, len(node_dict)]
                node_num += len(feature[t])
            #
            for t in types:
                t_i = node_dict[t][1]
                node_feature.append(torch.tensor(feature[t], dtype=torch.float32).to(device))
                node_time += list(time_info[t])
                node_type += [node_dict[t][1] for _ in range(len(feature[t]))]
            #
            edge_dict = {e[2]: i for i, e in enumerate(graph.get_meta_graph())}
            edge_dict['self'] = len(edge_dict)
            #
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
            node_feature = torch.cat(tuple(node_feature), 0).to(device)
            node_type = torch.LongTensor(node_type).to(device)
            edge_time = torch.LongTensor(edge_time).to(device)
            edge_index = torch.LongTensor(edge_index).t().to(device)
            edge_type = torch.LongTensor(edge_type).to(device)
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
gene_dir = "/Users/Inigo/Desktop/FloREN3.0/straeten_output/floren_gene_embeddings/"
cell_dir = "/Users/Inigo/Desktop/FloREN3.0/straeten_output/floren_cell_embeddings/"
att_dir = "/Users/Inigo/Desktop/FloREN3.0/straeten_output/floren_attention_scores/"
patient_emb_dir = "/Users/Inigo/Desktop/FloREN3.0/straeten_output/floren_patient_embeddings/"
patient_emb_dir_split = "/Users/Inigo/Desktop/FloREN3.0/straeten_output/floren_patient_embeddings/split/"
os.makedirs(gene_dir, exist_ok=True)
os.makedirs(cell_dir, exist_ok=True)
os.makedirs(att_dir, exist_ok=True)
os.makedirs(patient_emb_dir, exist_ok=True)
os.makedirs(patient_emb_dir_split, exist_ok=True)
plots_dir = "/Users/Inigo/Desktop/FloREN3.0/straeten_output/plots"

# Load metadata
#metadata = pd.read_csv("/home/inigo/Desktop/FloREN3.0/Binvignat/metadata_updated.csv")

# Device setup
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print("Using device:", device)

# Load trained model
#model_name = f'Binvignat_epoch_{args.epoch}_n_hid_{args.n_hid}_nheads_{args.n_heads}_lr_01_n_batch{args.n_batch}'
#model_path = os.path.join(model_dir, model_name)
model0=f'Straeten_epoch_{args.epoch}_n_hid_{args.n_hid}_nheads_{args.n_heads}_lr_01_n_batch{args.n_batch}'
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










#---------------------------------------------------------------
#
#                         UMAPS
#
#---------------------------------------------------------------

import matplotlib.pyplot as plt
import umap
import smtplib
from email.message import EmailMessage
import mimetypes
import ssl

# === UMAP SAMPLES PLOT ===
vector_path = "/Users/Inigo/Desktop/FloREN3.0/straeten_output/floren_patient_embeddings/"  # Path to saved patient embeddings
vector_files = [f for f in os.listdir(vector_path) if f.endswith(".csv")]
vector_files = [f for f in vector_files if "n_hid_128_nheads_8" in f]
all_vectors = []
labels = []
for file in vector_files:
    df = pd.read_csv(os.path.join(vector_path, file), header=None)  # No index column
    all_vectors.append(df.values.squeeze())  # Shape: [552]
    label = "MS" if "MS" in file else "HC"
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
    colors = {"MS": "red", "HC": "blue"}
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










#---------------------------------------------------------------
#
#                     CELL TYPES
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

import scanpy as sc
adata = sc.read_h5ad("C:/Users/Inigo/Desktop/FloREN/Straeten/Muenster_Human_CSF_toscanpy_Apr23_2021.h5ad")
meta = adata.obs[['Celltype.Lev1.manuscript','Sample']]
meta["barcode"]  = meta.Sample.astype(str)+"__"+meta.Sample.astype(str)+"_"+meta['Celltype.Lev1.manuscript'].astype(str)+"_"+meta.index
meta.columns = ["celltype", "patient","barcode"]
meta["celltype"] = adata.obs[['Celltype.Lev1.manuscript']]

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



#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------



from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

# Output
plots_dir = "/Users/Inigo/Desktop/FloREN3.0/straeten_output/plots"
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


#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------



# ============================
# USER CONTROLS
# ============================
TOP_N = 4   # 👈 choose number of top cell types
output_pdf = os.path.join(
    plots_dir,
    f"FloREN_Celltype_Saliency_Top{TOP_N}_mean.pdf"
)

# ============================
# PREPARE DATA
# ============================
# Select top-N cell types by mean saliency
top_celltypes = mean_celltype_saliency.head(TOP_N).index.tolist()

plot_df = all_cells[all_cells["celltype"].isin(top_celltypes)].copy()
plot_df = plot_df.reset_index(drop=True)
plot_df["celltype"] = pd.Categorical(
    plot_df["celltype"],
    categories=top_celltypes,
    ordered=True
)

# ============================
# STYLE (Nature Methods–like)
# ============================
sns.set_theme(style="white")  # no grid
plt.rcParams.update({
    "font.size": 12,
    "axes.linewidth": 1.2,
    "pdf.fonttype": 42,   # editable text in Illustrator
    "ps.fonttype": 42
})

# Muted, high-contrast palette
palette = sns.color_palette("Set2", n_colors=TOP_N)

# ============================
# PLOT
# ============================
with PdfPages(output_pdf) as pdf:
    fig, ax = plt.subplots(figsize=(5 + TOP_N * 1.2, 5))
    # Violin
    sns.violinplot(
        data=plot_df,
        x="celltype",
        y="saliency",
        palette=palette,
        inner=None,
        cut=0,
        width=0.8,
        linewidth=1.2,
        saturation=0.9,
        ax=ax
    )
    # Jittered points
    sns.stripplot(
        data=plot_df,
        x="celltype",
        y="saliency",
        color="black",
        size=4,
        jitter=0.25,
        alpha=0.7,
        ax=ax
    )
    # Median markers
    medians = plot_df.groupby("celltype")["saliency"].median()
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
    ax.tick_params(axis="x", rotation=30)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)



#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------



# ============================
# USER CONTROLS
# ============================
TOP_N = 4
output_pdf = os.path.join(
    plots_dir,
    f"FloREN_Celltype_Saliency_Top{TOP_N}_NatureMethods_MS_HD_mean.pdf"
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
    lambda p: "MS" if "MS" in p else "HD"
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
    "MS": "#B22222",   # firebrick (muted red)
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
    medians = plot_df.groupby("celltype")["saliency"].median()
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
    ax.tick_params(axis="x", rotation=30)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Legend cleanup
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[:2],
        labels[:2],
        title="Condition",
        frameon=False,
        loc="upper right"
    )
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)










#-----------------------------------------------------
#
#              MONO GENE SIGNATURE
#
#-----------------------------------------------------

meta["celltype"] = adata.obs[['Celltype.Lev1.manuscript']]

target_celltype = "Mono"
#target_celltype = "CD4 T"

patients = [d for d in os.listdir(inter_dir) if os.path.isdir(os.path.join(inter_dir, d))]
patients.sort()

for patient in patients:
    print(f"Processing patient:", patient)
    pdir = os.path.join(inter_dir, patient)
    #
    # Load gradients
    df2 = pd.read_csv(os.path.join(att_dir, f"{patient}_edge_att.csv"), delimiter=",").iloc[:,1:]
    patient_mask = cell_names["0"].str.startswith(patient + "__")
    patient_cell_names = cell_names.loc[patient_mask, "0"].reset_index(drop=True)
    cell_df = pd.DataFrame({
        "local_cell_index": np.arange(len(patient_cell_names)),
        "barcode_full": patient_cell_names
    })
    # Extract barcode
    split = cell_df["barcode_full"].str.split("__", expand=True)
    cell_df["barcode"] = cell_df["barcode_full"]
    cell_df["patient"] = patient
    # Convert to GLOBAL graph node index
    cell_df["node_index"] = cell_df["local_cell_index"] + n_genes
    cell_df = cell_df.merge(
        meta[["barcode", "celltype", "patient"]],
        on=["barcode", "patient"],
        how="left"
    )
    #target_celltype = "classical monocyte"
    target_cells = cell_df[
        (cell_df["celltype"] == target_celltype) &
        (cell_df["patient"] == patient)
        ]["node_index"].values
    df2.columns = ["src", "tgt", "att"]
    mask = (
            ((df2["src"].isin(target_cells)) & (df2["tgt"] < n_genes)) |
            ((df2["tgt"].isin(target_cells)) & (df2["src"] < n_genes))
    )
    df_celltype_edges = df2[mask].copy()
    gene_attention = np.zeros(n_genes, dtype=np.float64)
    # If gene is source
    gene_mask_src = df_celltype_edges["src"] < n_genes
    np.add.at(
        gene_attention,
        df_celltype_edges.loc[gene_mask_src, "src"].values,
        df_celltype_edges.loc[gene_mask_src, "att"].values
    )
    # If gene is target
    gene_mask_tgt = df_celltype_edges["tgt"] < n_genes
    np.add.at(
        gene_attention,
        df_celltype_edges.loc[gene_mask_tgt, "tgt"].values,
        df_celltype_edges.loc[gene_mask_tgt, "att"].values
    )
    n_target_cells = len(target_cells)
    if n_target_cells > 0:
        gene_attention /= n_target_cells
    #
    out_dir = os.path.join(inter_dir, patient, "celltype_specific")
    os.makedirs(out_dir, exist_ok=True)
    np.savetxt(
        os.path.join(out_dir, f"genes_att_{target_celltype.replace(' ', '_')}_mean.csv"),
        gene_attention,
        delimiter=","
    )

##############################################################
# 3. Iterate patients and compute gene & cell saliency
##############################################################

all_patients_gene_df = []

for patient in patients:
    print(f"Processing patient:", patient)
    pdir = os.path.join(inter_dir, patient, "celltype_specific")
    #
    # Load gradients
    genes_grad = np.loadtxt(os.path.join(pdir, f"genes_att_{target_celltype.replace(' ', '_')}_mean.csv"), delimiter=",")
    #
    # Reduce: take row mean = saliency per gene/cell
    gene_scores = genes_grad
    #
    # -------------  GENES: no grouping needed -------------------
    #
    gene_df = pd.DataFrame({
        "gene": gene_names,
        "saliency": gene_scores,
        "patient": patient
    })
    #
    all_patients_gene_df.append(gene_df)

# ===================== 4. Combine Across Patients ====================
all_genes = pd.concat(all_patients_gene_df)

# Compute mean across patients
mean_gene_saliency = all_genes.groupby("gene")["saliency"].mean().sort_values(ascending=False)
# ===================== 5.  PLOTS  ===================================

# ============================
# USER PARAMETERS
# ============================
TOP_N_GENES = 100
output_pdf = os.path.join(
    plots_dir,
    f"FloREN_{target_celltype.replace(' ', '_')}_GeneSignature_Top{TOP_N_GENES}_mean_mean.pdf"
)

# ============================
# PREPARE DATA
# ============================
top_genes = mean_gene_saliency.head(TOP_N_GENES).index.tolist()

plot_df = all_genes[all_genes["gene"].isin(top_genes)].copy()
plot_df = plot_df.reset_index(drop=True)
plot_df["gene"] = pd.Categorical(
    plot_df["gene"],
    categories=top_genes,
    ordered=True
)
plot_df["condition"] = plot_df["patient"].apply(
    lambda p: "MS" if "MS" in p else "HD"
)

# ============================
# STYLE (Nature Methods)
# ============================
sns.set_theme(style="white")

plt.rcParams.update({
    "font.size": 11,
    "axes.linewidth": 1.2,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

# 🎨 Colors
violin_color = "#F6C28B"   # soft, clean orange
condition_palette = {
    "MS": "#B22222",       # muted red
    "HD": "#1F4E79"        # muted blue
}

# ============================
# PLOT
# ============================
with PdfPages(output_pdf) as pdf:
    fig, ax = plt.subplots(
        figsize=(8.5, 4.8)   # 🔥 compressed width, journal-friendly
    )
    # Violin plots
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
    # Jittered points
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
    # Median markers
    medians = plot_df.groupby("gene")["saliency"].mean()
    ax.scatter(
        np.arange(len(medians)),
        medians.values,
        color="black",
        s=40,
        zorder=5
    )
    # Log scale
    ax.set_yscale("log")
    # Labels & title
    ax.set_title(
        f"{target_celltype} Gene Attention Signature (Top {TOP_N_GENES})",
        fontsize=13,
        pad=8
    )
    ax.set_xlabel("")
    ax.set_ylabel("Attention saliency (log scale)")
    # Axis aesthetics
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Legend (clean)
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
# EXTRA: CELL TYPE GENE SIGNATURES COMPARISONS
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
cell_types = ['CD4 T', 'CD8 T', 'Mono', 'Un_assigned', 'NK', 'other T', 'other', 'B', 'DC']

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
    output_path = "/Users/Inigo/Desktop/FloREN3.0/straeten_output/celltype_attention_matrix.csv"
    #celltype_attention_matrix.to_csv(output_path)
    print(f"Saved to: {output_path}")
else:
    print("No cell types had valid data.")

import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
# ── Assuming celltype_attention_matrix is already in memory ──
# (rows = cell types, columns = genes, values = mean attention)
# 1. Prepare data
cell_types = celltype_attention_matrix.index.tolist()
# ── A. Cosine Similarity Heatmap ──
# Cosine similarity between cell-type attention profiles
cos_sim = cosine_similarity(celltype_attention_matrix)
cos_sim_df = pd.DataFrame(cos_sim, index=cell_types, columns=cell_types)
# ── B. Jaccard Index on top 50 genes per cell type ──
# Get top 50 genes per cell type (by attention value)
top_n = 20
top_genes_per_ct = {}
for ct in cell_types:
    row = celltype_attention_matrix.loc[ct]
    # Genes with highest attention (exclude zeros if you prefer strict non-zero)
    top_genes = row.nlargest(top_n).index.tolist()
    top_genes_per_ct[ct] = set(top_genes)

# Compute Jaccard matrix
jaccard_matrix = np.zeros((len(cell_types), len(cell_types)))
for i, ct1 in enumerate(cell_types):
    for j, ct2 in enumerate(cell_types):
        set1 = top_genes_per_ct[ct1]
        set2 = top_genes_per_ct[ct2]
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        jaccard_matrix[i, j] = intersection / union if union > 0 else 0.0

jaccard_df = pd.DataFrame(jaccard_matrix, index=cell_types, columns=cell_types)

# ── Plotting ──
output_pdf = f"/Users/Inigo/Desktop/FloREN3.0/straeten_output/plots/celltype_comparison_heatmaps_top{top_n}_mean.pdf"
with PdfPages(output_pdf) as pdf:
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), constrained_layout=True)
    # Left: Cosine Similarity
    sns.heatmap(cos_sim_df,
                annot=True,
                fmt=".2f",
                cmap="YlGnBu",
                vmin=0, vmax=1,
                cbar_kws={'label': 'Cosine Similarity'},
                linewidths=0.5,
                ax=axes[0])
    axes[0].set_title("Cosine Similarity\nbetween Cell-type Attention Profiles")
    axes[0].set_xlabel("Cell Type")
    axes[0].set_ylabel("Cell Type")
    # Right: Jaccard on top 50 genes
    sns.heatmap(jaccard_df,
                annot=True,
                fmt=".2f",
                cmap="OrRd",
                vmin=0, vmax=1,
                cbar_kws={'label': 'Jaccard Index'},
                linewidths=0.5,
                ax=axes[1])
    axes[1].set_title(f"Jaccard Index\n(top {top_n} genes per cell type)")
    axes[1].set_xlabel("Cell Type")
    axes[1].set_ylabel("Cell Type")
    # Rotate labels if needed
    for ax in axes:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    #
    plt.suptitle("Cell-type Attention Profile Comparison", fontsize=16, y=1.02)
    pdf.savefig(fig, dpi=180)
    plt.close(fig)

print(f"Both heatmaps saved to: {output_pdf}")

# Assuming celltype_attention_matrix is your DataFrame:
# rows = cell types, columns = genes, values = mean attention
W = celltype_attention_matrix.copy()
# Step 1: Make non-negative and add small pseudocount (like +1 in R)
W = np.maximum(W, 0) + 1e-6  # small epsilon instead of +1 to avoid inflating small values too much
# Alternative (closer to R): W = W.clip(lower=0) + 1
# Step 2: Compute specificity / uniqueness score
n_celltypes, n_genes = W.shape
specificity = np.zeros_like(W.values)
for g in range(n_genes):
    col = W.iloc[:, g].values  # attention values for this gene across all cell types
    for j in range(n_celltypes):
        own_val = col[j]
        others = np.delete(col, j)  # all other cell types
        if len(others) == 0:
            max_other = 1e-6
        else:
            max_other = np.max(others)
        #
        # Compute log ratio (uniqueness term)
        ratio = np.log(own_val / max_other) if max_other > 0 else 0
        ratio = max(ratio, 0)  # clamp negative → no penalty if not unique
        # Final weighted specificity score
        specificity[j, g] = own_val * ratio  # or own_val * (ratio + 1) if you want to keep more signal

# Convert to DataFrame with same index/columns as original
specificity_df = pd.DataFrame(specificity,
                              index=W.index,
                              columns=W.columns)
print("Specificity-weighted scores shape:", specificity_df.shape)

# Cosine similarity between cell-type attention profiles
cos_sim = cosine_similarity(specificity_df)
cos_sim_df = pd.DataFrame(cos_sim, index=cell_types, columns=cell_types)
# ── B. Jaccard Index on top 50 genes per cell type ──
# Get top 50 genes per cell type (by attention value)
top_n = 20
top_genes_per_ct = {}
for ct in cell_types:
    row = specificity_df.loc[ct]
    # Genes with highest attention (exclude zeros if you prefer strict non-zero)
    top_genes = row.nlargest(top_n).index.tolist()
    top_genes_per_ct[ct] = set(top_genes)

# Compute Jaccard matrix
jaccard_matrix = np.zeros((len(cell_types), len(cell_types)))
for i, ct1 in enumerate(cell_types):
    for j, ct2 in enumerate(cell_types):
        set1 = top_genes_per_ct[ct1]
        set2 = top_genes_per_ct[ct2]
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        jaccard_matrix[i, j] = intersection / union if union > 0 else 0.0

jaccard_df = pd.DataFrame(jaccard_matrix, index=cell_types, columns=cell_types)

# ── Plotting ──
output_pdf = f"/Users/Inigo/Desktop/FloREN3.0/straeten_output/plots/celltype_nir_comparison_heatmaps_top{top_n}_mean.pdf"
with PdfPages(output_pdf) as pdf:
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), constrained_layout=True)
    # Left: Cosine Similarity
    sns.heatmap(cos_sim_df,
                annot=True,
                fmt=".2f",
                cmap="YlGnBu",
                vmin=0, vmax=1,
                cbar_kws={'label': 'Cosine Similarity'},
                linewidths=0.5,
                ax=axes[0])
    axes[0].set_title("Cosine Similarity\nbetween Cell-type Attention Profiles")
    axes[0].set_xlabel("Cell Type")
    axes[0].set_ylabel("Cell Type")
    # Right: Jaccard on top 50 genes
    sns.heatmap(jaccard_df,
                annot=True,
                fmt=".2f",
                cmap="OrRd",
                vmin=0, vmax=1,
                cbar_kws={'label': 'Jaccard Index'},
                linewidths=0.5,
                ax=axes[1])
    axes[1].set_title(f"Jaccard Index\n(top {top_n} genes per cell type)")
    axes[1].set_xlabel("Cell Type")
    axes[1].set_ylabel("Cell Type")
    # Rotate labels if needed
    for ax in axes:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    #
    plt.suptitle("Cell-type Attention Profile Comparison", fontsize=16, y=1.02)
    pdf.savefig(fig, dpi=180)
    plt.close(fig)

print(f"Both heatmaps saved to: {output_pdf}")

top_n = 100
top_genes_per_ct = {}
for celltype in specificity_df.index:
    scores = specificity_df.loc[celltype]
    top_genes = scores.nlargest(top_n).index.tolist()
    top_scores = scores[top_genes].values
    top_genes_per_ct[celltype] = {
        'genes': top_genes,
        'scores': top_scores,
        #'symbolized': [get_symbol(g) for g in top_genes]  # for plotting
    }



#target_celltype = "CD8 T"
#target_celltype = "CD4 T"
target_celltype = "Mono"

patients = [d for d in os.listdir(inter_dir) if os.path.isdir(os.path.join(inter_dir, d))]
patients.sort()

for patient in patients:
    print(f"Processing patient:", patient)
    pdir = os.path.join(inter_dir, patient)
    #
    # Load gradients
    df2 = pd.read_csv(os.path.join(att_dir, f"{patient}_edge_att.csv"), delimiter=",").iloc[:,1:]
    patient_mask = cell_names["0"].str.startswith(patient + "__")
    patient_cell_names = cell_names.loc[patient_mask, "0"].reset_index(drop=True)
    cell_df = pd.DataFrame({
        "local_cell_index": np.arange(len(patient_cell_names)),
        "barcode_full": patient_cell_names
    })
    # Extract barcode
    split = cell_df["barcode_full"].str.split("__", expand=True)
    cell_df["barcode"] = cell_df["barcode_full"]
    cell_df["patient"] = patient
    # Convert to GLOBAL graph node index
    cell_df["node_index"] = cell_df["local_cell_index"] + n_genes
    cell_df = cell_df.merge(
        meta[["barcode", "celltype", "patient"]],
        on=["barcode", "patient"],
        how="left"
    )
    #target_celltype = "classical monocyte"
    target_cells = cell_df[
        (cell_df["celltype"] == target_celltype) &
        (cell_df["patient"] == patient)
        ]["node_index"].values
    df2.columns = ["src", "tgt", "att"]
    mask = (
            ((df2["src"].isin(target_cells)) & (df2["tgt"] < n_genes)) |
            ((df2["tgt"].isin(target_cells)) & (df2["src"] < n_genes))
    )
    df_celltype_edges = df2[mask].copy()
    gene_attention = np.zeros(n_genes, dtype=np.float64)
    # If gene is source
    gene_mask_src = df_celltype_edges["src"] < n_genes
    np.add.at(
        gene_attention,
        df_celltype_edges.loc[gene_mask_src, "src"].values,
        df_celltype_edges.loc[gene_mask_src, "att"].values
    )
    # If gene is target
    gene_mask_tgt = df_celltype_edges["tgt"] < n_genes
    np.add.at(
        gene_attention,
        df_celltype_edges.loc[gene_mask_tgt, "tgt"].values,
        df_celltype_edges.loc[gene_mask_tgt, "att"].values
    )
    out_dir = os.path.join(inter_dir, patient, "celltype_specific")
    os.makedirs(out_dir, exist_ok=True)
    np.savetxt(
        os.path.join(out_dir, f"genes_att_{target_celltype.replace(' ', '_')}_mean.csv"),
        gene_attention,
        delimiter=","
    )

##############################################################
# 3. Iterate patients and compute gene & cell saliency
##############################################################

all_patients_gene_df = []

for patient in patients:
    print(f"Processing patient:", patient)
    pdir = os.path.join(inter_dir, patient, "celltype_specific")
    #
    # Load gradients
    genes_grad = np.loadtxt(os.path.join(pdir, f"genes_att_{target_celltype.replace(' ', '_')}_mean.csv"), delimiter=",")
    #
    # Reduce: take row mean = saliency per gene/cell
    gene_scores = genes_grad
    #
    # -------------  GENES: no grouping needed -------------------
    #
    gene_df = pd.DataFrame({
        "gene": gene_names,
        "saliency": gene_scores,
        "patient": patient
    })
    #
    all_patients_gene_df.append(gene_df)

# ===================== 4. Combine Across Patients ====================
all_genes = pd.concat(all_patients_gene_df)

# Compute mean across patients
mean_gene_saliency = all_genes.groupby("gene")["saliency"].mean().sort_values(ascending=False)

# ============================
# USER PARAMETERS
# ============================
TOP_N_GENES = 20
target_celltype = "Mono"

output_pdf = os.path.join(
    plots_dir,
    f"FloREN_{target_celltype}_GeneSignature_Top{TOP_N_GENES}_NIR_mean_mean_notlog.pdf"
)

# ============================
# SELECT TOP UNIQUE GENES (NIR)
# ============================
top_genes = top_genes_per_ct['Mono']['genes'][1:20]

# Subset data
plot_df = all_genes[all_genes["gene"].isin(top_genes)].copy()
plot_df = plot_df.reset_index(drop=True)
# Preserve ranking order
plot_df["gene"] = pd.Categorical(
    plot_df["gene"],
    categories=top_genes,
    ordered=True
)

# Condition label
plot_df["condition"] = plot_df["patient"].apply(
    lambda p: "MS" if "MS" in p else "HD"
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
    "MS": "#8B1E3F",       # deep muted red (distinct from violin)
    "HD": "#264653"        # muted navy blue
}

# ============================
# PLOT
# ============================
from matplotlib.backends.backend_pdf import PdfPages

with PdfPages(output_pdf) as pdf:
    fig, ax = plt.subplots(
        figsize=(8.2, 4.6)  # compact & journal-friendly
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
        .loc[top_genes]
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
    #ax.set_yscale("log")
    # ── Labels & title ───────────────────────────
    ax.set_title(
        f"{target_celltype} — Top {TOP_N_GENES} Unique Genes\n"
        "Ranked by specificity-weighted attention",
        fontsize=13,
        pad=8
    )
    ax.set_xlabel("")
    ax.set_ylabel("Attention saliency (log scale)")
    # ── Axis aesthetics ──────────────────────────
    ax.tick_params(axis="x", rotation=45, labelsize=8)
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

print(f"Saved: {output_pdf}")










#-----------------------------------------------------
#
# EXTRA: CELL TYPE MEAN VISUALIZATION
#
#-----------------------------------------------------

meta["celltype"] = adata.obs[['Celltype.Lev1.manuscript']]

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
    #celltype_df.to_csv(os.path.join(pdir, "Celltype_LVL1_att_saliency.csv"), index=False)
    #
    all_patients_cell_df.append(celltype_df)

# ===================== 4. Combine Across Patients ====================
all_genes = pd.concat(all_patients_gene_df)
all_cells = pd.concat(all_patients_cell_df)

# Compute mean across patients
mean_gene_saliency = all_genes.groupby("gene")["saliency"].mean().sort_values(ascending=False)
mean_celltype_saliency = all_cells.groupby("celltype")["saliency"].mean().sort_values(ascending=False)

# ===================== 5.  PLOTS  ===================================
plots_dir = "/Users/Inigo/Desktop/FloREN3.0/straeten_output/plots"
output_pdf = os.path.join(plots_dir, "FloREN_att_lvl2_visualization_mean.pdf")


# Rank by group max
def compute_group_max(df):
    vals_MS = df[df["patient"].str.contains("MS")]["saliency"]
    vals_HC = df[~df["patient"].str.contains("MS")]["saliency"]
    max_MS = vals_MS.max() if len(vals_MS) > 0 else 0
    max_HC = vals_HC.max() if len(vals_HC) > 0 else 0
    return max(max_MS, max_HC)


gene_group_max = (
    all_genes.groupby("gene")
    .apply(compute_group_max)
    .sort_values(ascending=False)
)
ordered_genes = gene_group_max.head(100).index.tolist()
cell_group_max = (
    all_cells.groupby("celltype")
    .apply(compute_group_max)
    .sort_values(ascending=False)
)
ordered_cells = cell_group_max.index.tolist()
# ---------------------------------------------------------
# 1) Make labels smaller
# 2) Convert ENSG IDs → Gene symbols
# 3) Make bars transparent
# 4) Plot per-patient points
# ---------------------------------------------------------

import mygene
mg = mygene.MyGeneInfo()

# -----------------------------
# Convert ENSG IDs → Gene Symbols
# -----------------------------
#gene_map = mg.querymany(
#    mean_gene_saliency.index.tolist(),
#    scopes="ensembl.gene",
#    fields="symbol",
#    species="human",
#    as_dataframe=True
#)

# Clean mapping
#map_series = gene_map["symbol"].fillna(gene_map.index.to_series())
#mean_gene_saliency.index = mean_gene_saliency.index.map(map_series)
#all_genes["gene"] = all_genes["gene"].map(lambda g: map_series.get(g, g))
#gene_group_max_indices = pd.Series(gene_group_max.index).map(lambda g: map_series.get(g, g))
#gene_group_max.index = gene_group_max_indices


# ---------------------------------------------------------
# Colors by group
# ---------------------------------------------------------
def patient_color(p):
    return "red" if "MS" in p else "blue"


from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

with PdfPages(output_pdf) as pdf:
    fig, axes = plt.subplots(2, 1, figsize=(20, 14))
    # =========================================================
    # ======================= GENES ===========================
    # =========================================================
    ax = axes[0]
    # --- Order top 100 genes by mean saliency ---
    top_genes = mean_gene_saliency.head(100)
    ordered_genes = top_genes.index.tolist()
    positions = np.arange(len(ordered_genes))
    pos_blue = positions - 0.15
    pos_red = positions + 0.15
    # Background barplot (mean values)
    # ax.bar(
    #    positions,
    #    top_genes.values,
    #    color="gray",
    #    alpha=0.3,
    #    edgecolor="black",
    #    zorder=1
    # )
    ax.bar(
        positions,
        gene_group_max.loc[ordered_genes].values,
        color="gray",
        alpha=0.3,
        edgecolor="black",
        zorder=1
    )
    # Prepare split data
    blue_data = []
    red_data = []
    for gene in ordered_genes:
        df = all_genes[all_genes["gene"] == gene]
        blue_vals = df[df["patient"].apply(lambda p: "MS" not in p)]["saliency"]
        red_vals = df[df["patient"].apply(lambda p: "MS" in p)]["saliency"]
        blue_data.append(blue_vals)
        red_data.append(red_vals)
        #
    # --- Boxplots without outliers ---
    ax.boxplot(
        blue_data,
        positions=pos_blue,
        widths=0.25,
        patch_artist=True,
        showfliers=False,  # 🔥 no outliers
        boxprops=dict(facecolor="blue", alpha=0.3),
        medianprops=dict(color="black"),
        zorder=2
    )
    ax.boxplot(
        red_data,
        positions=pos_red,
        widths=0.25,
        patch_artist=True,
        showfliers=False,  # 🔥 no outliers
        boxprops=dict(facecolor="red", alpha=0.3),
        medianprops=dict(color="black"),
        zorder=2
    )
    # --- Scatter points with minimal size ---
    for i, gene in enumerate(ordered_genes):
        df = all_genes[all_genes["gene"] == gene]
        df_blue = df[df["patient"].apply(lambda p: "MS" not in p)]
        df_red = df[df["patient"].apply(lambda p: "MS" in p)]
        ax.scatter(
            np.repeat(pos_blue[i], len(df_blue)),
            df_blue["saliency"],
            color="blue",
            s=6,  # 🔥 minimum size
            alpha=0.7,
            zorder=3
        )
        ax.scatter(
            np.repeat(pos_red[i], len(df_red)),
            df_red["saliency"],
            color="red",
            s=6,  # 🔥 minimum size
            alpha=0.7,
            zorder=3
        )
        #
    ax.set_xticks(positions)
    ax.set_xticklabels(ordered_genes, rotation=90, fontsize=4)
    ax.set_title("Top 100 Genes — Split Boxplots with Mean Background")
    ax.set_ylabel("Saliency Score")
    ax.grid(axis="y", alpha=0.3)
    # =========================================================
    # ======================= CELLS ===========================
    # =========================================================
    ax = axes[1]
    ordered_cells = mean_celltype_saliency.index.tolist()
    positions = np.arange(len(ordered_cells))
    pos_blue = positions - 0.15
    pos_red = positions + 0.15
    # Background barplot
    # ax.bar(
    #    positions,
    #    mean_celltype_saliency.values,
    #    color="gray",
    #    alpha=0.3,
    #    edgecolor="black",
    #    zorder=1
    # )
    ax.bar(
        positions,
        cell_group_max.loc[ordered_cells].values,
        color="gray",
        alpha=0.3,
        edgecolor="black",
        zorder=1
    )
    blue_data = []
    red_data = []
    for ct in ordered_cells:
        df = all_cells[all_cells["celltype"] == ct]
        blue_vals = df[df["patient"].apply(lambda p: "MS" not in p)]["saliency"]
        red_vals = df[df["patient"].apply(lambda p: "MS" in p)]["saliency"]
        blue_data.append(blue_vals)
        red_data.append(red_vals)
        #
    # Boxplots
    ax.boxplot(
        blue_data,
        positions=pos_blue,
        widths=0.25,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor="blue", alpha=0.3),
        medianprops=dict(color="black"),
        zorder=2
    )
    ax.boxplot(
        red_data,
        positions=pos_red,
        widths=0.25,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor="red", alpha=0.3),
        medianprops=dict(color="black"),
        zorder=2
    )
    # Scatter points
    for i, ct in enumerate(ordered_cells):
        df = all_cells[all_cells["celltype"] == ct]
        df_blue = df[df["patient"].apply(lambda p: "MS" not in p)]
        df_red = df[df["patient"].apply(lambda p: "MS" in p)]
        ax.scatter(
            np.repeat(pos_blue[i], len(df_blue)),
            df_blue["saliency"],
            color="blue",
            s=6,
            alpha=0.7,
            zorder=3
        )
        ax.scatter(
            np.repeat(pos_red[i], len(df_red)),
            df_red["saliency"],
            color="red",
            s=6,
            alpha=0.7,
            zorder=3
        )
        #
    ax.set_xticks(positions)
    ax.set_xticklabels(ordered_cells, rotation=90, fontsize=6)
    ax.set_title("Cell Types — Split Boxplots with Mean Background")
    ax.set_ylabel("Saliency Score")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    pdf.savefig(fig)




#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

# Output
plots_dir = "/Users/Inigo/Desktop/FloREN3.0/straeten_output/plots"
output_pdf = os.path.join(
    plots_dir,
    "FloREN_Celltype_Saliency_Violinplots_mean.pdf"
)

# Order cell types by mean saliency (already computed)
ordered_cells = mean_celltype_saliency.index.tolist()

# Keep only ordered cells
plot_df = all_cells.copy()
plot_df["celltype"] = pd.Categorical(
    plot_df["celltype"],
    categories=ordered_cells,
    ordered=True
)

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

# ============================
# USER CONTROLS
# ============================
TOP_N = 4   # 👈 choose number of top cell types
output_pdf = os.path.join(
    plots_dir,
    f"FloREN_Celltype_att_lvl1_Top{TOP_N}_mean.pdf"
)

# ============================
# PREPARE DATA
# ============================
# Select top-N cell types by mean saliency
top_celltypes = mean_celltype_saliency.head(TOP_N).index.tolist()

plot_df = all_cells[all_cells["celltype"].isin(top_celltypes)].copy()
plot_df["celltype"] = pd.Categorical(
    plot_df["celltype"],
    categories=top_celltypes,
    ordered=True
)

# ============================
# STYLE (Nature Methods–like)
# ============================
sns.set_theme(style="white")  # no grid
plt.rcParams.update({
    "font.size": 12,
    "axes.linewidth": 1.2,
    "pdf.fonttype": 42,   # editable text in Illustrator
    "ps.fonttype": 42
})

# Muted, high-contrast palette
palette = sns.color_palette("Set2", n_colors=TOP_N)

# ============================
# PLOT
# ============================
with PdfPages(output_pdf) as pdf:
    fig, ax = plt.subplots(figsize=(5 + TOP_N * 1.2, 5))
    # Violin
    sns.violinplot(
        data=plot_df,
        x="celltype",
        y="saliency",
        palette=palette,
        inner=None,
        cut=0,
        width=0.8,
        linewidth=1.2,
        saturation=0.9,
        ax=ax
    )
    # Jittered points
    sns.stripplot(
        data=plot_df,
        x="celltype",
        y="saliency",
        color="black",
        size=4,
        jitter=0.25,
        alpha=0.7,
        ax=ax
    )
    # Median markers
    medians = plot_df.groupby("celltype")["saliency"].median()
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
    ax.tick_params(axis="x", rotation=30)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)



#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------



# ============================
# USER CONTROLS
# ============================
TOP_N = 4
output_pdf = os.path.join(
    plots_dir,
    f"FloREN_Celltype_att_lvl1_Top{TOP_N}_ms_hd_mean.pdf"
)

# ============================
# PREPARE DATA
# ============================
top_celltypes = mean_celltype_saliency.head(TOP_N).index.tolist()

plot_df = all_cells[all_cells["celltype"].isin(top_celltypes)].copy()
plot_df["celltype"] = pd.Categorical(
    plot_df["celltype"],
    categories=top_celltypes,
    ordered=True
)

# Define condition
plot_df["condition"] = plot_df["patient"].apply(
    lambda p: "MS" if "MS" in p else "HD"
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
    "MS": "#B22222",   # firebrick (muted red)
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
    medians = plot_df.groupby("celltype")["saliency"].median()
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
    ax.tick_params(axis="x", rotation=30)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Legend cleanup
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[:2],
        labels[:2],
        title="Condition",
        frameon=False,
        loc="upper right"
    )
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)





#---------------------------------------------------------------
#
#                 DIFFERENTIAL ABUNDANCE ANALYSIS
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

import scanpy as sc
adata = sc.read_h5ad("C:/Users/Inigo/Desktop/FloREN/Straeten/Muenster_Human_CSF_toscanpy_Apr23_2021.h5ad")
meta = adata.obs[['Celltype.Lev1.manuscript','Sample']]
meta["barcode"]  = meta.Sample.astype(str)+"__"+meta.Sample.astype(str)+"_"+meta['Celltype.Lev1.manuscript'].astype(str)+"_"+meta.index
meta.columns = ["celltype", "patient","barcode"]
meta["celltype"] = adata.obs[['Celltype.Lev1.manuscript']]

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
    #grouped = merged.groupby("celltype")["saliency"].mean().sort_values(ascending=False)
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

all_cells["group"] = all_cells["patient"].str.contains("MS", case=False, na=False)
all_cells["group"] = all_cells["group"].map({True: "MS", False: "HC"})

import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

results = []
for celltype, df_ct in all_cells.groupby("celltype"):
    hc_vals = df_ct.loc[df_ct["group"] == "HC", "saliency"]
    ms_vals = df_ct.loc[df_ct["group"] == "MS", "saliency"]
    # Only test if both groups have values
    if len(hc_vals) > 0 and len(ms_vals) > 0:
        stat, pval = mannwhitneyu(
            hc_vals,
            ms_vals,
            alternative="two-sided"
        )
        results.append({
            "celltype": celltype,
            "n_HC": len(hc_vals),
            "n_MS": len(ms_vals),
            "U_statistic": stat,
            "p_value": pval,
            "HC_median": hc_vals.median(),
            "MS_median": ms_vals.median()
        })

results_df = pd.DataFrame(results)

# Optional: FDR correction (strongly recommended for publication)
results_df["p_adj"] = multipletests(
    results_df["p_value"],
    method="fdr_bh"
)[1]

from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import os

# Output
plots_dir = "/Users/Inigo/Desktop/FloREN3.0/straeten_output/plots"
output_pdf = os.path.join(
    plots_dir,
    "FloREN_Celltype_Saliency_Boxplots_HC_vs_MS.pdf"
)
ordered_cells = mean_celltype_saliency.index.tolist()
plot_df = all_cells.copy().reset_index(drop=True)
plot_df["celltype"] = pd.Categorical(
    plot_df["celltype"],
    categories=ordered_cells,
    ordered=True
)
# Clean Nature-style colors
palette = {
    "HC": "#3A7FB0",   # clean blue
    "MS": "#C23B3B"    # clean red
}
sns.set_style("white")
with PdfPages(output_pdf) as pdf:
    for scale in ["linear", "log"]:
        fig, ax = plt.subplots(figsize=(22, 8))
        # BOX PLOT (transparent)
        box = sns.boxplot(
            data=plot_df,
            x="celltype",
            y="saliency",
            hue="group",
            palette=palette,
            dodge=True,
            width=0.6,
            showcaps=True,
            showfliers=False,
            boxprops=dict(facecolor='none', linewidth=1.5),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            medianprops=dict(linewidth=2),
            ax=ax
        )
        # Recolor box edges manually (important trick)
        #for artist, group in zip(ax.artists,
        #                         [g for _ in ordered_cells for g in ["HC","MS"]]):
        #    artist.set_edgecolor(palette[group])
        #    artist.set_facecolor('none')
        #
        # Recolor lines (whiskers, caps, medians)
        #for line, group in zip(ax.lines,
        #                       [g for _ in ordered_cells for g in ["HC","MS"] for _ in range(6)]):
        #    line.set_color(palette[group])
        #
        # STRIP PLOT
        sns.stripplot(
            data=plot_df,
            x="celltype",
            y="saliency",
            hue="group",
            palette=palette,
            dodge=True,
            size=4,
            alpha=0.7,
            linewidth=0,
            ax=ax
        )
        # Remove duplicate legends
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:2], labels[:2], frameon=False)
        # Log scale if needed
        if scale == "log":
            ax.set_yscale("log")
            ax.set_title("Cell Type Saliency Distribution (HC vs MS, Log Scale)", fontsize=16)
            ax.set_ylabel("Log Saliency Score")
        else:
            ax.set_title("Cell Type Saliency Distribution (HC vs MS)", fontsize=16)
            ax.set_ylabel("Saliency Score")
        #
        ax.set_xlabel("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)
        # Nature-style cleanup
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

print("Saved to:", output_pdf)

from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import matplotlib.pyplot as plt
import os

plots_dir = "/Users/Inigo/Desktop/FloREN3.0/straeten_output/plots"
output_pdf = os.path.join(
    plots_dir,
    "FloREN_Celltype_Saliency_Boxplots_HC_vs_MS_test.pdf"
)
ordered_cells = mean_celltype_saliency.index.tolist()
plot_df = all_cells.copy().reset_index(drop=True)
plot_df["celltype"] = pd.Categorical(
    plot_df["celltype"],
    categories=ordered_cells,
    ordered=True
)
# Clean publication palette
palette = {
    "HC": "#3A7FB0",   # blue
    "MS": "#C23B3B"    # red
}
sns.set_style("white")

with PdfPages(output_pdf) as pdf:
    for scale in ["linear", "log"]:
        fig, ax = plt.subplots(figsize=(22, 8))
        # 1. Create the boxplot
        sns.boxplot(
            data=plot_df,
            x="celltype",
            y="saliency",
            hue="group",
            palette=palette,
            dodge=True,
            width=0.6,
            showfliers=False,
            ax=ax
        )
        # 2. Iterate through the artists to color the lines and remove fill
        # Each box in a grouped boxplot has 6 lines (whiskers, caps, median) + 1 patch
        #lines_per_box = 6
        for i, patch in enumerate(ax.patches):
            # Get the color assigned by the palette based on index
            # This handles the HC/MS alternation
            col = patch.get_facecolor()
            # Make the box itself transparent but set the edge color
            patch.set_edgecolor(col)
            patch.set_facecolor('none')
            # Color the corresponding lines (median, whiskers, caps)
            box_x = patch.get_path().get_extents().get_points()[:, 0].mean()
            #for j in range(i * lines_per_box, (i + 1) * lines_per_box):
            #    ax.lines[j].set_color(col)
            for line in ax.lines:
                line_x = line.get_xdata()
                if np.all(np.isclose(line_x, box_x, atol=0.2)):  # atol=0.2 captures whiskers/caps
                    line.set_color(col)
        #
        sns.stripplot(
            data=plot_df,
            x="celltype",
            y="saliency",
            hue="group",
            palette=palette,
            dodge=True,
            size=4,
            alpha=0.7,
            linewidth=0,
            ax=ax
        )
        # Remove duplicated legends
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:2], labels[:2], frameon=False)
        if scale == "log":
            ax.set_yscale("log")
            ax.set_title("Cell Type Saliency Distribution (HC vs MS, Log Scale)", fontsize=16)
            ax.set_ylabel("Log Saliency Score")
        else:
            ax.set_title("Cell Type Saliency Distribution (HC vs MS)", fontsize=16)
            ax.set_ylabel("Saliency Score")
        #
        ax.set_xlabel("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

print("Saved to:", output_pdf)










#---------------------------------------------------------------
#
#                     CELL TYPE RANKING
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

import scanpy as sc
adata = sc.read_h5ad("C:/Users/Inigo/Desktop/FloREN/Straeten/Muenster_Human_CSF_toscanpy_Apr23_2021.h5ad")
meta = adata.obs[['Celltype.Lev1.manuscript','Sample']]
meta["barcode"]  = meta.Sample.astype(str)+"__"+meta.Sample.astype(str)+"_"+meta['Celltype.Lev1.manuscript'].astype(str)+"_"+meta.index
meta.columns = ["celltype", "patient","barcode"]
meta["celltype"] = adata.obs[['Celltype.Lev1.manuscript']]

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
# Ensure data is sorted (highest to lowest)
#df_plot = q75_celltype_saliency.sort_values(ascending=True).reset_index()
df_plot = mean_celltype_saliency.sort_values(ascending=True).reset_index()
# Setup the figure for Nature Methods standards
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
plt.rcParams['pdf.fonttype'] = 42  # Essential for editable PDF text
fig, ax = plt.subplots(figsize=(5, 4)) # Compact size for a single-column figure
# 1. Add subtle horizontal lines (the "lollipops")
ax.hlines(y=df_plot['celltype'], xmin=(df_plot['saliency'].min()-5), xmax=df_plot['saliency'],
          color='#d1d1d1', linestyle='-', linewidth=1, zorder=1)
# 2. Plot the data points (the dots)
ax.scatter(df_plot['saliency'], df_plot['celltype'],
           color='#2c3e50', s=60, edgecolors='white', linewidth=0.25, zorder=2)
# 3. Refine the Axes
ax.set_xlabel("75th Percentile Saliency Score", fontsize=10, fontweight='bold')
ax.set_ylabel("") # Category labels on Y-axis are self-explanatory
ax.set_title("Ranked Cell Type Saliency", loc='left', fontsize=12, pad=15)
# 4. Professional "Nature" styling
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False) # Remove Y-axis spine for cleaner look
ax.tick_params(axis='both', labelsize=9)
ax.xaxis.grid(True, linestyle='--', alpha=0.4, zorder=0) # Subtle vertical guide
# Set a tighter X-axis range to emphasize differences
# Adjust based on your data spread (min is ~190, max ~204)
#ax.set_xlim(188, 206)
plt.tight_layout()
# To save for publication:
fig.savefig(os.path.join(
    plots_dir,"Saliency_Ranking_NatureMethods_LVL1_mean_q75.pdf"), dpi=300, bbox_inches='tight')


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Prepare data
df_plot = q75_celltype_saliency.sort_values(ascending=True).reset_index()
# Publication settings
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.sans-serif'] = 'Arial'
# Increase height slightly for 31 categories, but keep width compact
fig, ax = plt.subplots(figsize=(5, 8))
# 1. Smaller lollipops (linewidth and markersize)
ax.hlines(y=df_plot['celltype'], xmin=(df_plot['saliency'].min()-5), xmax=df_plot['saliency'],
          color='#bcbcbc', linestyle='-', linewidth=0.8, zorder=1)
# Smaller dots (s=30) with a thinner edge
ax.scatter(df_plot['saliency'], df_plot['celltype'],
           color='#2c3e50', s=30, edgecolors='white', linewidth=0.4, zorder=2)
# 2. Refine Labels
ax.set_xlabel("75th Percentile Saliency Score", fontsize=9, fontweight='bold')
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
    plots_dir,"Saliency_Ranking_NatureMethods_LVL2_mean_q75.pdf"), dpi=300, bbox_inches='tight')











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
cell_types = ['CD4 T', 'CD8 T', 'Mono', 'Un_assigned', 'NK', 'other T', 'other', 'B', 'DC']

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
    output_path = "/Users/Inigo/Desktop/FloREN3.0/straeten_output/celltype_attention_matrix.csv"
    #celltype_attention_matrix.to_csv(output_path)
    print(f"Saved to: {output_path}")
else:
    print("No cell types had valid data.")

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

# Assuming celltype_attention_matrix is your DataFrame:
# rows = cell types, columns = genes, values = mean attention
W = celltype_attention_matrix.copy()
# Step 1: Make non-negative and add small pseudocount (like +1 in R)
W = np.maximum(W, 0) + 1e-6  # small epsilon instead of +1 to avoid inflating small values too much
# Alternative (closer to R): W = W.clip(lower=0) + 1
# Step 2: Compute specificity / uniqueness score
n_celltypes, n_genes = W.shape
specificity = np.zeros_like(W.values)
for g in range(n_genes):
    col = W.iloc[:, g].values  # attention values for this gene across all cell types
    for j in range(n_celltypes):
        own_val = col[j]
        others = np.delete(col, j)  # all other cell types
        if len(others) == 0:
            max_other = 1e-6
        else:
            max_other = np.max(others)
        #
        # Compute log ratio (uniqueness term)
        ratio = np.log(own_val / max_other) if max_other > 0 else 0
        ratio = max(ratio, 0)  # clamp negative → no penalty if not unique
        # Final weighted specificity score
        specificity[j, g] = own_val * ratio  # or own_val * (ratio + 1) if you want to keep more signal

# Convert to DataFrame with same index/columns as original
specificity_df = pd.DataFrame(specificity,
                              index=W.index,
                              columns=W.columns)

# 1. Select top 50 genes by mean across celltypes
gene_means = specificity_df.iloc[2,:]
top50_genes = (
    gene_means
    .sort_values(ascending=False)
    .head(50)
    .index
)

top_matrix = celltype_attention_matrix[top50_genes]
# Rank genes by mean
top_matrix = top_matrix.loc[:, gene_means[top50_genes].sort_values(ascending=False).index]
# 2. Convert to long format
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
# 3. Plot
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
    plots_dir,"Saliency_MONO_gene_signatures_comparison_50.pdf"), dpi=300, bbox_inches='tight')



# Prepare data
gene_means = specificity_df.iloc[2,:]
top50_genes = (
    gene_means
    .sort_values(ascending=False)
    .head(50)
    .index
)
top50_genes = top50_genes[~top50_genes.str.startswith("AC")]
top50_genes = top50_genes[~top50_genes.str.startswith("AL")]
#top_matrix = specificity_df[top50_genes]
df_plot = gene_means[top50_genes].reset_index()
df_plot.columns = ["gene", "saliency"]
df_plot_sorted = df_plot.sort_values('saliency', ascending=True)
#plot_df = df.set_index("gene")
#df_plot = q75_celltype_saliency.sort_values(ascending=True).reset_index()
# Publication settings
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.sans-serif'] = 'Arial'
# Increase height slightly for 31 categories, but keep width compact
fig, ax = plt.subplots(figsize=(5, 8))
# 1. Smaller lollipops (linewidth and markersize)
ax.hlines(y=df_plot_sorted['gene'], xmin=0.005, xmax=df_plot_sorted['saliency'],
          color='#bcbcbc', linestyle='-', linewidth=0.8, zorder=1)
# Smaller dots (s=30) with a thinner edge
ax.scatter(df_plot_sorted['saliency'], df_plot_sorted['gene'],
           color='#2c3e50', s=30, edgecolors='white', linewidth=0.4, zorder=2)
# 2. Refine Labels
ax.set_xlabel("Mean Percentile Saliency Score", fontsize=9, fontweight='bold')
ax.set_title("Ranked Gene Saliency", loc='left', fontsize=11, pad=10)
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
ax.set_xlim(0.005, 0.035)
plt.tight_layout()
# To save for publication:
fig.savefig(os.path.join(
    plots_dir,"Saliency_MONO_signature_50_noAC.pdf"), dpi=300, bbox_inches='tight')










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
palette = {"HC": "blue", "MS": "red"}
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










#-----------------------------------------------------
#
# CELLS ATTENTION UMAP
#
#-----------------------------------------------------
import scanpy as sc
import os
import numpy as np
import pandas as pd

result_dir = '/Users/Inigo/Desktop/FloREN3.0/straeten_output'  # Sets parent directory for output
inter_dir = result_dir + '/interpretability/'  # Sets directory for loss output
patients = [d for d in os.listdir(inter_dir) if os.path.isdir(os.path.join(inter_dir, d))]
cell_names = pd.read_csv(os.path.join(result_dir, "All_AUC_Cell_names.csv"))

adata = sc.read_h5ad("C:/Users/Inigo/Desktop/FloREN/Straeten/Muenster_Human_CSF_toscanpy_Apr23_2021.h5ad")
meta = adata.obs[['Celltype.Lev1.manuscript','Sample']]
meta["barcode"]  = meta.Sample.astype(str)+"__"+meta.Sample.astype(str)+"_"+meta['Celltype.Lev1.manuscript'].astype(str)+"_"+meta.index
meta.columns = ["celltype", "patient","barcode"]
meta["celltype"] = adata.obs[['Celltype.Lev1.manuscript']]


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


import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# Make plots PDF-friendly (vectorized text)
sc.set_figure_params(dpi=100, dpi_save=300, format='pdf', vector_friendly=True)
# Output path
pdf_path = "C:/Users/Inigo/Desktop/FloREN3.0/straeten_output/plots/umap_attention_summary_2.pdf"
with PdfPages(pdf_path) as pdf:
    fig, axes = plt.subplots(1, 4, figsize=(60, 12))
    # ── LEFT: SLE_status ──
    sc.pl.umap(
        adata,
        color="Disease_Status",
        ax=axes[0],
        show=False,
        frameon=False,
        title="Disease Status",
        size=10
    )
    # ── MIDDLE: cell type ──
    sc.pl.umap(
        adata,
        color="Celltype.Lev1.manuscript",
        ax=axes[1],
        show=False,
        frameon=False,
        title="Main Cell Type",
        size=10
    )
    # ── MIDDLE: cell type ──
    sc.pl.umap(
        adata,
        color="Celltype.Lev2.manuscript",
        ax=axes[2],
        show=False,
        frameon=False,
        title="Fine Cell Type",
        size=10
    )
    # ── RIGHT: attention ──
    sc.pl.umap(
        adata,
        #color="attention",
        #color="attention_log",
        #color="attention_sqrt",
        color="attention_z",
        #color="attention_rank",
        ax=axes[3],
        show=False,
        frameon=False,
        title="Attention",
        cmap="viridis",
        size=10
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










#-----------------------------------------------------
#
# VOLCANO PLOT CELLS
#
#-----------------------------------------------------
# 1. Merge the dataframes
df = pd.merge(results_df, df_plot, on='celltype')

# 2. Calculate Log2 Fold Change (MS median / HC median)
# We use log2 to represent doubling/halving symmetrically around 0
df['log_fc'] = np.log2(df['MS_median'] / df['HC_median'])

# 3. Define coloring logic
# Significant (p < 0.05) & LFC > 0 -> Red
# Significant (p < 0.05) & LFC < 0 -> Blue
# Others -> Gray
def assign_color(row):
    if row['p_value'] < 0.05:
        return 'red' if row['log_fc'] > 0 else 'blue'
    return 'gray'

df['color'] = df.apply(assign_color, axis=1)

# 4. Create the plot
plt.figure(figsize=(10, 7))

# Plot each group separately for the legend
for color, label in zip(['red', 'blue', 'gray'], ['Significant Increase', 'Significant Decrease', 'Not Significant']):
    subset = df[df['color'] == color]
    plt.scatter(subset['log_fc'], subset['saliency'],
                c=color, label=label, edgecolors='black', alpha=0.8, s=120)

# 5. Add cell type labels to each point
for i, row in df.iterrows():
    plt.text(row['log_fc'] + 0.05, row['saliency'] + 0.1, row['celltype'], fontsize=9)

# Formatting
plt.axvline(x=0, color='black', linestyle='--', linewidth=1) # Baseline
plt.xlabel('$\log_2$ Fold Change (MS median / HC median)')
plt.ylabel('Attention Score')
plt.title('Volcano Plot: Cell Type Attention vs. Median Fold Change')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig("C:/Users/Inigo/Desktop/FloREN3.0/straeten_output/plots/LFC_vs_Saliency_Panel.pdf", dpi=300)




