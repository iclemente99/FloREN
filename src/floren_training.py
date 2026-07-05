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

import scanpy as sc



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
parser.add_argument('--cell_rate', type=float, default=0.1)
parser.add_argument('--gene_rate', type=float, default=0.1)

# Result
parser.add_argument('--data_name', type=str, default='FloREN', help='The name for dataset')
parser.add_argument('--reduction', type=str, default='AE', help='the method for feature extraction, pca, raw, AE')
parser.add_argument('--in_dim', type=int, default=256, help='Number of hidden dimension (AE)')
parser.add_argument('--floren_grn', type=str, default='True', help='Decide if to run the floren published gene-gene connections inference')

# HGTGSSL
parser.add_argument('--knn', type=float, default=30, help='Number of Nearest Neighbors to keep in KNN cell-gene calculation when --tfs = True ')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--n_hid', type=int, default=128, help='Number of hidden dimension')
parser.add_argument('--n_heads', type=int, default=8 ,help='Number of attention head')
parser.add_argument('--n_layers', type=int, default=2, help='Number of GNN layers')
parser.add_argument('--dropout', type=float, default=0, help='Dropout ratio')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--batch_size', type=int, help='Number of output nodes for training')
parser.add_argument('--layer_type', type=str, default='hgt', help='the layer type for GAE')
parser.add_argument('--loss', type=str, default='kl', help='the loss for GAE')
parser.add_argument('--factor', type=float, default='0.5', help='the attenuation factor')
#parser.add_argument('--patience', type=int, default=5, help='patience')
parser.add_argument('--rf', type=float, default='0.0', help='the weights of regularization')
parser.add_argument('--cuda', type=int, default=1, help='cuda 0 use GPU0 else cpu')
parser.add_argument('--rep', type=str, default='T', help='precision truncation')
parser.add_argument('--AEtype', type=int, default=1, help='AEtype:1 embedding node autoencoder 2:HGT node autoencode')
parser.add_argument('--optimizer', type=str, default='adamw', help='optimizer')
parser.add_argument('--test_size', type=int, default=0.2, help='Size of test during training')
parser.add_argument('--val_size', type=float, default=0.2, help='Size of validation during training')
parser.add_argument('--patience', type=float, default=30, help='patience for early stopping')

# Data directories
parser.add_argument('--data_path', default='./data/', type=str, help='Path to FloREN data')
parser.add_argument('--output_path', default='./floren_output', type=str, help='Path to folder with graph construction outputs')
parser.add_argument('--cell_comm_path', default=None, type=str, help='Path to folder with cell-cell communication adjacency matrix')
parser.add_argument('--tfs', default=False, type=str, help='Option to work at tfs level')
parser.add_argument('--result_dir', default=None, type=str, help='Path to folder with FloREN model outputs')
#parser.add_argument('--metadata_path', default=None, type=str, help='Path to metadata file')
parser.add_argument('--patient_id', default='patient_id', type=str,
                    help='adata.obs column with the patient identifier')
parser.add_argument('--metadata_group', default='disease', type=str,
                    help='adata.obs column with the metadata group to differentiate')
parser.add_argument('--count_layer', default='logcounts', type=str,
                    help='adata.layer name with the log normalized counts')
parser.add_argument('--min_count', default=0, type=int,
                    help='minimum count presence to have a cell-gene association')
parser.add_argument('--adata_path', default='./data/binvignat_example.h5ad', type=str, help='Path to adata object')


args = parser.parse_args()
#args.data_path = "C:/Users/Inigo/Desktop/FloREN/Schafflick/genes"
#args.output_path = "C:/Users/Inigo/Desktop/FloREN3.0/schafflick_output"
#args.metadata_path = "C:/Users/Inigo/Desktop/FloREN/Schafflick/samples_metadata.csv"

#args.epoch = 100
#args.n_hid = 128
#args.n_heads = 13
#args.lr = 0.1
#args.n_batch = 32
#args.epoch = 100  # Hyperparameter specification
#args.n_hid = 128  # Hyperparameter specification
#args.n_heads = 8  # Hyperparameter specification
# args.n_heads = 13 # Hyperparameter specification
# args.lr = 0.1 # Hyperparameter specification
#args.lr = 0.0005  # Hyperparameter specification
sample_name = args.data_name
patient_id = args.patient_id
count_layer = args.count_layer
metadata_group = args.metadata_group
min_count = args.min_count

# Set up result directories
#output_path = os.path.abspath(os.path.expanduser(args.output_path))
output_path = args.output_path
#output_path = "C:/Users/Inigo/Desktop/FloREN3.0/perez_output"
embeddings_path = os.path.join(output_path, "embeddings")
gene_embeddings_path = os.path.join(embeddings_path, "gene_embeddings")
cell_embeddings_path = os.path.join(embeddings_path, "cell_embeddings")
connections_path = os.path.join(output_path, "connections")
gene_embeddings_norm_path = os.path.join(embeddings_path, "gene_embeddings_norm")
cell_embeddings_norm_path = os.path.join(embeddings_path, "cell_embeddings_norm")
#os.makedirs(gene_embeddings_path, exist_ok=True)
#os.makedirs(cell_embeddings_path, exist_ok=True)
#os.makedirs(connections_path, exist_ok=True)

print("Using device:")
#args.cuda = 1
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

print("Loading adata object")
data_path = args.data_path
adata = sc.read_h5ad(args.adata_path)
inds = np.unique(adata.obs[patient_id].values.astype(str))
#count_matrices_path = os.path.join(data_path, "count_matrices")
#count_matrices_path = "/home/inigo/Desktop/FloREN3.0/Binvignat/genes/"
#print(f"Looking for CSV files in: {data_path}")
#csv_files = glob.glob(os.path.join(data_path, "*.csv"))
#print(f"Found {len(csv_files)} CSV files: {csv_files}")
# Load reference for gene names and concatenate all matrices
#reference = pd.read_csv(csv_files[0])
#gene_names = reference[reference.columns[0]].values
#n_genes = reference.shape[0]
gene_names = adata.var_names
n_genes = len(adata.var_names)

if n_genes < args.in_dim:
    h_n = n_genes
else:
    h_n = args.in_dim

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
files = [f[1:] if f.startswith("\\") else f for f in files]


patient_graphs = []

from sklearn.metrics import pairwise_distances
from scipy.sparse import lil_matrix

print("\nGENERATING PATIENT GRAPHS")
# for file in range(len(files)):
for file_idx, patient_name in enumerate(files):
    # patient_name = files[file]
    print(f"    Processing patient: {patient_name}")
    #
    # -------------------------
    # Gene-cell adjacency
    # -------------------------
    if args.tfs == False:
        #load_file = [f for f in csv_files if patient_name in f][0]
        #transformed_matrix = pd.read_csv(load_file)
        #transformed_matrix = transformed_matrix.iloc[:, 1:].values
        adata_subset = adata[adata.obs[patient_id].isin([patient_name])]
        transformed_matrix = adata_subset.layers[count_layer].A.T
        gene_cell = transformed_matrix
        gene_cell[gene_cell >= min_count] = 1
        # gene_cell[gene_cell <= 5] = 0
    else:
        cells_files = pd.read_csv(os.path.join(cell_embeddings_norm_path, f"{patient_name}_AE_Emb_Cells.csv"))
        cells_files.drop(cells_files.columns[0], axis=1, inplace=True)
        encoded2 = torch.tensor(cells_files.values, dtype=torch.float32).to(device)
        genes_files = pd.read_csv(os.path.join(gene_embeddings_norm_path, f"{patient_name}_AE_Emb_Genes.csv"))
        genes_files.drop(genes_files.columns[0], axis=1, inplace=True)
        encoded = torch.tensor(genes_files.values, dtype=torch.float32).to(device)
        dist = pairwise_distances(encoded, encoded2)  # genes × cells
        K = args.knn
        transformed_matrix = lil_matrix((encoded.shape[0], encoded2.shape[0]))
        for i in range(dist.shape[0]):
            top_idx = np.argpartition(dist[i], K)[:K]
            transformed_matrix[i, top_idx] = 1
        #
        gene_cell = transformed_matrix
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
    if args.cell_comm_path == None:
        cells_f = pd.DataFrame(np.zeros([gene_cell.shape[1], gene_cell.shape[1]]))
    else:
        cells_f = pd.read_csv(os.path.join(args.cell_comm_path, f"{patient_name}.csv"))
        cells_f.set_index(cells_f.columns[0], inplace=True)
    #
    # -------------------------
    # Gene embeddings (per patient)
    # -------------------------
    genes_files = pd.read_csv(os.path.join(gene_embeddings_norm_path, f"{patient_name}_AE_Emb_Genes.csv"))
    genes_files.drop(genes_files.columns[0], axis=1, inplace=True)
    encoded = torch.tensor(genes_files.values, dtype=torch.float32).to(device)
    print(encoded.shape)
    #
    # -------------------------
    # Cell embeddings (per patient)
    # -------------------------
    cells_files = pd.read_csv(os.path.join(cell_embeddings_norm_path, f"{patient_name}_AE_Emb_Cells.csv"))
    cells_files.drop(cells_files.columns[0], axis=1, inplace=True)
    encoded2 = torch.tensor(cells_files.values, dtype=torch.float32).to(device)
    print(encoded2.shape)
    #
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

#args.epoch = 100  # Hyperparameter specification
#args.n_hid = 128  # Hyperparameter specification
#args.n_heads = 8  # Hyperparameter specification
# args.n_heads = 13 # Hyperparameter specification
# args.lr = 0.1 # Hyperparameter specification
#args.lr = 0.0005  # Hyperparameter specification
# args.lr_txt = '01'
# args.n_batch = 32 # Hyperparameter specification
# sample_name = re.split('TFs_',tfs_files[file],2)[-1][:re.split('TFs_',tfs_files[file],2)[-1].index(".")]
#sample_name = "Perez"
file0 = f'sample_{sample_name}_epoch_{args.epochs}_n_hid_{args.n_hid}_nheads_{args.n_heads}_lr_01_n_batch{args.n_batch}'  # Text saving hypermarametrization
# print(f'\n{file0}') #Print hyperparametrization
#args.result_dir = '/Users/Inigo/Desktop/FloREN3.0/perez_output'  # Sets parent directory for output
args.result_dir = args.output_path
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

print(device)  # Print device. IMPORTANT for incompatibilities.

################################################################
#
# SUBGRAPHS SAMPLING FOR TRAINING
#
################################################################

#metadata = pd.read_csv(args.metadata_path)
#metadata = pd.read_csv("/Users/Inigo/Desktop/FloREN/Perez/samples_metadata_unique.csv")
#metadata.columns = ["Unnamed: 0", "0", "patient_id", "group", "Age", "Sex", "Ethnicity", "Pool", "Batch", "Age_Group"]
#metadata.columns = ["patient_id", "patient_num", "patient_ID", "group", "Age", "Sex", "Tissue"]
#metadata["group"] = metadata["group"].astype("category").cat.codes
#metadata["patient_id"] = "schafflick_" + metadata["patient_id"].astype(str)
print("Splitting in training and test")
metadata = adata.obs[[patient_id,metadata_group]]
metadata = metadata.groupby(patient_id).first().reset_index()
metadata["group"] = metadata[metadata_group].astype("category").cat.codes

def stratified_split_graphs(patient_graphs, metadata_df, test_size=0.15, val_size=0.15, random_state=42):
    # Make lookup from metadata
    meta_lookup = dict(zip(metadata_df[patient_id], metadata_df["group"]))
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
    patient_graphs, metadata, test_size=args.test_size, val_size=args.val_size
)

print("Sampling end!")
debuginfoStr('Cell Graph constructed and pruned')

gnn = GNN(conv_name=args.layer_type, in_dim=h_n,
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
nb_epochs = args.epochs
weight_contrastive = 0.1
weight_classification = 0.9
#first_cutoff = False
#first_time = False
#contrastive_patience = 0
#best_contrastive = float('inf')

patience = args.patience  # patience for early stopping (validation)
counter = 0
best_model_loss = float('inf')
best_model_state = None

# optionally freeze gnn later: we will check a flag
# freeze_gcn_when = None  # if you want to explicitly set epoch to freeze GCN; else keep None

# Example dynamic-multi-task schedule variables (you can tune)
#weight_decay_step = 0.001
#weight_decay_min = 0.2
#weight_inc_step = 0.001
#weight_inc_max = 0.8

# Some bookkeeping arrays for plotting later
training_losses = []
validation_losses = []
contrastive_losses = []
classification_losses = []
total_losses = []

#args.gene_rate = 0.1
#args.cell_rate = 0.1

# Set random seed for reproducibility
random_seed = 42
random.seed(random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)


def pregenerate_jobs(graphs, gene_rate, cell_rate, device):
    jobs_dict = {}
    for patient in graphs:
        patient_name = patient['name']
        # Get graph and gene_cell matrix
        graph = patient['graph']
        #load_file = [f for f in csv_files if patient_name in f][0]
        #transformed_matrix = pd.read_csv(load_file)
        #transformed_matrix = transformed_matrix.iloc[:, 1:].values
        adata_subset = adata[adata.obs[patient_id].isin([patient_name])]
        transformed_matrix = adata_subset.layers[count_layer].A.T
        gene_cell = transformed_matrix
        #
        # Generate jobs
        jobs = []
        gene_num = int(gene_cell.shape[0] * gene_rate)
        cell_num = int(gene_cell.shape[1] * cell_rate)
        jobs.append(sub_sample(graph, gene_cell, gene_cell.shape[1], gene_cell.shape[0], gene_cell.shape[0], gene_cell.shape[1], query=True))
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
            #print(f"Jobs: edge_index - {edge_index}")
            node_num = 0
            types = graph.get_types()
            for t in types:
                node_dict[t] = [node_num, len(node_dict)]
                node_num += len(feature[t])
            #
            for t in types:
                #t_i = node_dict[t][1]
                #node_feature.append(torch.tensor(feature[t], dtype=torch.float32).to(device))
                #node_time += list(time_info[t])
                #node_type += [node_dict[t][1] for _ in range(len(feature[t]))]
                t_i = node_dict[t][1]
                node_feature.insert(t_i, torch.tensor(feature[t], dtype=torch.float32).to(device))  # Saves cells or genes matrices
                node_time += list(time_info[t])  # Saves node time
                node_type += [node_dict[t][1] for _ in range(len(feature[t]))]  # Saves node type
            #
            edge_dict = {e[2]: i for i, e in enumerate(graph.get_meta_graph())}
            edge_dict['self'] = len(edge_dict)
            #if formatted_jobs == []:
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
            #node_feature = torch.cat(tuple(node_feature), 0).to(device)
            node_feature = torch.cat((node_feature[0], node_feature[1]), 0)
            node_type = torch.LongTensor(node_type).to(device)
            edge_time = torch.LongTensor(edge_time).to(device)
            edge_index = torch.LongTensor(edge_index).t().to(device)
            edge_type = torch.LongTensor(edge_type).to(device)
            #print(f"Jobs: edge_index - {edge_index.max()}")
            formatted_jobs.append([node_feature, node_type, edge_time, edge_index, edge_type, og_gene_indxs, og_cell_indxs])
        #
        jobs_dict[patient_name] = formatted_jobs
    #
    print(edge_dict)
    return jobs_dict

# Generate and format jobs
print("Pre-generating and formatting jobs...")
#args.gene_rate = 0.1
#args.cell_rate = 0.1
train_jobs = pregenerate_jobs(train_graphs, args.gene_rate, args.cell_rate, device)
val_jobs = pregenerate_jobs(val_graphs, args.gene_rate, args.cell_rate, device)
# print("Pre-generated and formatted jobs for all patients")
debuginfoStr('Pre-generated and formatted jobs for all patients')
# model0 = f'Perez_epoch_{args.epoch}_n_hid_{args.n_hid}_nheads_{args.n_heads}_lr_01_n_batch{args.n_batch}'
# state = torch.load(model_dir+model0, map_location=lambda storage, loc: storage) # Loads model
# gnn.load_state_dict(state['model'])
# Move GNN to device
gnn.to(device)

import psutil
import torch
import os
def get_memory_usage():
    # System RAM in GB
    process = psutil.Process(os.getpid())
    mem_gb = process.memory_info().rss / (1024 ** 3)
    # GPU RAM in GB (if using CUDA)
    gpu_mem_gb = 0
    if torch.cuda.is_available():
        gpu_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        # Reset the peak monitor so we see the max per epoch
        torch.cuda.reset_peak_memory_stats()
    #
    return mem_gb, gpu_mem_gb

print("\nSTARTING FLOREN TRAINING")
# Training loop
loop_start = time.time()
for epoch in range(nb_epochs):
    epoch_start = time.time()
    gnn.train()
    random.shuffle(train_graphs)
    epoch_contrastive = 0.0
    epoch_classification = 0.0
    epoch_total = 0.0
    #
    for patient in train_graphs:
        patient_name = patient['name']
        # print(f"Processing patient: {patient_name}")
        grp = metadata.loc[metadata[patient_id] == patient_name, 'group'].values
        if len(grp) == 0:
            raise ValueError(f"Patient {patient_name} not found in metadata")
        group_id = int(grp[0])
        num_groups = int(metadata['group'].max()) + 1
        labels_class = torch.zeros(num_groups, device=device)
        labels_class[group_id] = 1.0
        #
        # Get pre-formatted jobs
        gnn_input = train_jobs[patient_name]
        graph = patient['graph']
        #
        # Forward pass
        features = torch.cat([graph.node_feature['gene'], graph.node_feature['cell']], dim=0)
        logits, ret_class = gnn.forward(
            gnn_input[0][0], gnn_input[0][1], gnn_input[0][2], gnn_input[0][3], gnn_input[0][4],
            gnn_input[1][0], gnn_input[1][1], gnn_input[1][2], gnn_input[1][3], gnn_input[1][4],
            gnn_input[2][0], gnn_input[2][1], gnn_input[2][2], gnn_input[2][3], gnn_input[2][4]
        )
        #
        # Construct labels for contrastive loss
        num_nodes = gnn_input[0][0].shape[0]
        lbl_1 = torch.ones(1, num_nodes).to(device)
        lbl_2 = torch.zeros(1, num_nodes).to(device)
        lbl = torch.cat((lbl_1, lbl_2), dim=1)
        #
        # Compute losses
        contrastive_loss = contrastive_loss_fn(logits, lbl)
        classification_loss = classification_loss_fn(ret_class, labels_class)
        loss = weight_contrastive * contrastive_loss + weight_classification * classification_loss
        #
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #
        epoch_contrastive += contrastive_loss.item()
        epoch_classification += classification_loss.item()
        epoch_total += loss.item()
    #
    epoch_contrastive /= len(train_graphs)
    epoch_classification /= len(train_graphs)
    epoch_total /= len(train_graphs)
    #
    training_losses.append(epoch_total)
    contrastive_losses.append(epoch_contrastive)
    classification_losses.append(epoch_classification)
    total_losses.append(epoch_total)
    #
    # Validation
    gnn.eval()
    val_contrastive = 0.0
    val_classification = 0.0
    val_total = 0.0
    val_correct = 0
    val_total_count = 0
    #
    with torch.no_grad():
        for patient in val_graphs:
            patient_name = patient['name']
            grp = metadata.loc[metadata[patient_id] == patient_name, 'group'].values
            if len(grp) == 0:
                raise ValueError(f"Patient {patient_name} not found in metadata")
            group_id = int(grp[0])
            labels_class = torch.zeros(num_groups, device=device)
            labels_class[group_id] = 1.0
            #
            # Get pre-formatted jobs
            gnn_input = val_jobs[patient_name]
            graph = patient['graph']
            #
            # Forward pass
            features = torch.cat([graph.node_feature['gene'], graph.node_feature['cell']], dim=0)
            logits, ret_class = gnn.forward(
                gnn_input[0][0], gnn_input[0][1], gnn_input[0][2], gnn_input[0][3], gnn_input[0][4],
                gnn_input[1][0], gnn_input[1][1], gnn_input[1][2], gnn_input[1][3], gnn_input[1][4],
                gnn_input[2][0], gnn_input[2][1], gnn_input[2][2], gnn_input[2][3], gnn_input[2][4]
            )
            #
            # Construct labels for contrastive loss
            num_nodes = gnn_input[0][0].shape[0]
            lbl_1 = torch.ones(1, num_nodes).to(device)
            lbl_2 = torch.zeros(1, num_nodes).to(device)
            lbl = torch.cat((lbl_1, lbl_2), dim=1)
            #
            # Compute validation losses
            contrastive_loss = contrastive_loss_fn(logits, lbl)
            classification_loss = classification_loss_fn(ret_class, labels_class)
            loss = weight_contrastive * contrastive_loss + weight_classification * classification_loss
            #
            # Accumulate losses
            val_contrastive += contrastive_loss.item()
            val_classification += classification_loss.item()
            val_total += loss.item()
            #
            # Compute accuracy
            outputs = torch.softmax(ret_class, dim=-1)
            predicted = torch.argmax(outputs).item()
            true_label = torch.argmax(labels_class).item()
            val_correct += (predicted == true_label)
            val_total_count += 1
    #
    # Finalize validation metrics
    val_contrastive /= len(val_graphs)
    val_classification /= len(val_graphs)
    val_total /= len(val_graphs)
    val_acc = val_correct / val_total_count
    validation_losses.append(val_total)
    #
    # Scheduler step
    scheduler.step(torch.tensor(epoch_total))
    #
    # Print epoch summary
    print(f"Epoch {epoch + 1}/{nb_epochs} | Train loss: {epoch_total:.6f} | "
          f"Train Contr: {epoch_contrastive:.6f} | Train Class: {epoch_classification:.6f} | "
          f"Val loss: {val_total:.6f} | "  # Val Contr: {val_contrastive:.6f} | "
          # f"Val Class: {val_classification:.6f} | Val acc: {val_acc:.4f} | "
          f"time: {(time.time() - epoch_start) / 60:.1f} min")
    #
    # Early stopping and best model saving
    if epoch_total < best_model_loss:
        best_model_loss = epoch_total
        counter = 0
        best_model_state = copy.deepcopy(gnn.state_dict())
        state = {
            'model': gnn.state_dict(),
            'optimizer': scheduler.state_dict(),
            'epoch': epoch
        }
        model0 = f'{args.data_name}_epoch_{args.epochs}_n_hid_{args.n_hid}_nheads_{args.n_heads}_lr_01_n_batch{args.n_batch}'
        os.makedirs(model_dir, exist_ok=True)
        torch.save(state, os.path.join(model_dir, model0))
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}. Best val loss: {best_model_loss:.6f}")
            break
            #
    if epoch_classification < 1e-12:
        state = {
            'model': gnn.state_dict(),
            'optimizer': scheduler.state_dict(),
            'epoch': epoch
        }
        model0 = f'{args.data_name}_epoch_{args.epochs}_n_hid_{args.n_hid}_nheads_{args.n_heads}_lr_01_n_batch{args.n_batch}'
        os.makedirs(model_dir, exist_ok=True)
        torch.save(state, os.path.join(model_dir, model0))
        print("Classification loss reached zero. Stopping early.")
        break
        #
    if abs(epoch_contrastive - epoch_classification) < 0.001:
        state = {
            'model': gnn.state_dict(),
            'optimizer': scheduler.state_dict(),
            'epoch': epoch
        }
        model0 = f'{args.data_name}_epoch_{args.epochs}_n_hid_{args.n_hid}_nheads_{args.n_heads}_lr_01_n_batch{args.n_batch}'
        os.makedirs(model_dir, exist_ok=True)
        torch.save(state, os.path.join(model_dir, model0))
        print("Loss balanced. Stopping early.")
        break

# End of epochs
#print("Training finished. Final training losses:", training_losses[-5:])
#print("Validation losses:", validation_losses[-5:])

# Save loss history
loss_history = pd.DataFrame({
    "epoch": list(range(1, len(training_losses) + 1)),
    "train_total": training_losses,
    "train_contrastive": contrastive_losses,
    "train_classification": classification_losses,
    "val_total": validation_losses
})
model0 = f'{args.data_name}_epoch_{args.epochs}_n_hid_{args.n_hid}_nheads_{args.n_heads}_lr_01_n_batch{args.n_batch}'
loss_history.to_csv(os.path.join(loss_dir, model0 + ".csv"), index=False)

debuginfoStr('FloREN training finished')

ram_usage, vram_usage = get_memory_usage()
# Update your Print summary
print(f"Epoch {epoch + 1}/{nb_epochs} | "
      f"Train loss: {epoch_total:.4f} | "
      f"Val loss: {val_total:.4f} | "
      f"RAM: {ram_usage:.2f}GB | "   # New
      f"VRAM: {vram_usage:.2f}GB | " # New
      f"Time: {(time.time() - loop_start) / 60:.1f} min")
# end of epochs
# if best_model_state is not None:
#    gnn.load_state_dict(best_model_state)
#    print("Loaded best model from training (best_model.pth).")


state = {'model': gnn.state_dict(), 'optimizer': scheduler.state_dict(),
         'epoch': epoch}  # Saves a dictionary of model
model0 = f'Final_{args.data_name}_epoch_{args.epochs}_n_hid_{args.n_hid}_nheads_{args.n_heads}_lr_01_n_batch{args.n_batch}'
torch.save(state, model_dir + model0)  # Saves model in folder
# pd.DataFrame(training_loss).to_csv(loss_dir+model0+".csv")
#debuginfoStr('Graph Embedding training finished')



################################################################
#
# APPLY HGT TO WHOLE GRAPH
#
################################################################

# Directories for saving results
gene_dir = args.result_dir + '/floren_gene_embeddings/'
cell_dir = args.result_dir + '/floren_cell_embeddings/'
att_dir = args.result_dir + '/floren_attention_embeddings/'
patient_emb_dir = args.result_dir + '/floren_patient_embeddings/'
patient_emb_dir_split = args.result_dir + '/floren_patient_embeddings/split/'
os.makedirs(gene_dir, exist_ok=True)
os.makedirs(cell_dir, exist_ok=True)
os.makedirs(att_dir, exist_ok=True)
os.makedirs(patient_emb_dir, exist_ok=True)
os.makedirs(patient_emb_dir_split, exist_ok=True)

# Load metadata
# metadata = pd.read_csv("/home/inigo/Desktop/FloREN3.0/Binvignat/metadata_updated.csv")

# Device setup
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)

# Load trained model
# model_name = f'Binvignat_epoch_{args.epoch}_n_hid_{args.n_hid}_nheads_{args.n_heads}_lr_01_n_batch{args.n_batch}'
# model_path = os.path.join(model_dir, model_name)
#model0 = f'Perez_epoch_{args.epoch}_n_hid_{args.n_hid}_nheads_{args.n_heads}_lr_01_n_batch{args.n_batch}'
state = torch.load(model_dir + model0, map_location=lambda storage, loc: storage)  # Loads model
print(f"Loaded model from {model_dir + model0}")

test_jobs = pregenerate_jobs(test_graphs, args.gene_rate, args.cell_rate, device)

# Initialize GNN model
gnn = GNN(
    conv_name=args.layer_type,
    in_dim=args.in_dim,  # Should match training (e.g., 256)
    n_hid=args.n_hid,  # Should match training (e.g., 128)
    n_heads=args.n_heads,  # Should match training (e.g., 8)
    n_layers=args.n_layers,
    dropout=args.dropout,
    num_types=2,
    num_relations=7,
    n_labels=(np.max(metadata.group) + 1),  # Should match training (e.g., 2 for Control vs RA)
    use_RTE=False
).to(device)
gnn.load_state_dict(state['model'])
gnn.eval()

# Combine train and validation graphs for inference
all_graphs = train_graphs + val_graphs + test_graphs
all_jobs = {**train_jobs, **val_jobs, **test_jobs}  # Combine train_jobs and val_jobs

print("\nSAVING PATIENT REPRESENTATIONS")
# Process each patient
for patient in all_graphs:
    patient_name = patient['name']
    print(f"    Saving patient: {patient_name}")
    # Get pre-formatted job for full graph
    #gnn_input = all_jobs[patient_name][0]  # Use first job (full graph)
    gnn_input = all_jobs[patient_name]
    edge_index = gnn_input[0][3]  # Save edge_index for attention scores
    # Run GNN with full graph inputs repeated three times (mimicking training)
    with torch.no_grad():
        # Forward pass
        logits, ret_class = gnn.forward(
            gnn_input[0][0], gnn_input[0][1], gnn_input[0][2], gnn_input[0][3], gnn_input[0][4],
            gnn_input[1][0], gnn_input[1][1], gnn_input[1][2], gnn_input[1][3], gnn_input[1][4],
            gnn_input[2][0], gnn_input[2][1], gnn_input[2][2], gnn_input[2][3], gnn_input[2][4]
        )
        # Extract results
        att_full = gnn.att_0.abs() + gnn.att_n.abs()
        att_final = (att_full/2).mean(axis=1)
        cell_embs = (gnn.cells_0 + gnn.cells_n)/2
        gene_embs = (gnn.genes_0 + gnn.genes_n)/2
        # Patient embeddings
        emb_104 = gnn.emb_104.cpu().numpy()  # [104]
        g_32 = gnn.g_32.cpu().numpy()        # [32]
        l_32 = gnn.l_32.cpu().numpy()        # [32]
        g_64 = gnn.g_64.cpu().numpy()        # [64]
        l_64 = gnn.l_64.cpu().numpy()        # [64]
        g_128 = gnn.g_128.cpu().numpy()      # [128]
        l_128 = gnn.l_128.cpu().numpy()      # [128]
    #
    # Prepare patient embedding
    patient_embedding = np.concatenate([emb_104, g_32, l_32, g_64, l_64, g_128, l_128, ret_class], axis=0)  # [552]
    # Save results
    file_name = f'sample_{patient_name}_epoch_{args.epochs}_n_hid_{args.n_hid}_nheads_{args.n_heads}_lr_01_n_batch{args.n_batch}'
    # Save edge atts
    positions = pd.DataFrame(gnn_input[0][3].T)
    df = pd.DataFrame(att_final)
    df2 = pd.concat([positions, df], axis=1)
    df2.to_csv(os.path.join(att_dir, f"{patient_name}_edge_att.csv"), sep=",", index=True)
    # Save cell embeddings
    adata_subset = adata[adata.obs[patient_id].isin([patient_name])]
    #np.savetxt(os.path.join(cell_dir, f"{patient_name}_cell_embs.csv"), cell_embs, delimiter=",")
    pd.DataFrame(cell_embs, index=adata_subset.obs_names).to_csv(os.path.join(cell_dir, f"{patient_name}_cell_embs.csv"))
    # Save gene embeddings
    #np.savetxt(os.path.join(gene_dir, f"{patient_name}_gene_embs.csv"), gene_embs, delimiter=",")
    pd.DataFrame(gene_embs, index=adata_subset.var_names).to_csv(os.path.join(gene_dir, f"{patient_name}_gene_embs.csv"))
    # Save node attentions
    num_nodes = gnn_input[0][0].shape[0]
    node_attention = np.zeros(num_nodes, dtype=np.float64)
    np.add.at(node_attention, df2.iloc[:, 0].values, df2.iloc[:, 2].values)
    np.add.at(node_attention, df2.iloc[:, 1].values, df2.iloc[:, 2].values)
    num_genes = n_genes
    genes_grad = node_attention[:num_genes]  # (num_genes, in_dim)
    cells_grad = node_attention[num_genes:]  # (num_cells, in_dim)
    patient_out = os.path.join(inter_dir, patient_name)
    os.makedirs(patient_out, exist_ok=True)
    #np.savetxt(os.path.join(patient_out, "genes_atts.csv"), genes_grad, delimiter=",")
    #np.savetxt(os.path.join(patient_out, "cells_atts.csv"), cells_grad, delimiter=",")
    pd.DataFrame(genes_grad, index=adata_subset.var_names).to_csv(os.path.join(patient_out, "genes_atts.csv"))
    pd.DataFrame(cells_grad, index=adata_subset.obs_names).to_csv(os.path.join(patient_out, "cells_atts.csv"))
    # Save patient embeddings
    np.savetxt(os.path.join(patient_emb_dir_split, f"{patient_name}_emb_ssl.csv"), emb_104, delimiter=",")
    np.savetxt(os.path.join(patient_emb_dir_split, f"{patient_name}_emb_g32.csv"), g_32, delimiter=",")
    np.savetxt(os.path.join(patient_emb_dir_split, f"{patient_name}_emb_l32.csv"), l_32, delimiter=",")
    np.savetxt(os.path.join(patient_emb_dir_split, f"{patient_name}_emb_g64.csv"), g_64, delimiter=",")
    np.savetxt(os.path.join(patient_emb_dir_split, f"{patient_name}_emb_l64.csv"), l_64, delimiter=",")
    np.savetxt(os.path.join(patient_emb_dir_split, f"{patient_name}_emb_g128.csv"), g_128, delimiter=",")
    np.savetxt(os.path.join(patient_emb_dir_split, f"{patient_name}_emb_l128.csv"), l_128, delimiter=",")
    np.savetxt(os.path.join(patient_emb_dir_split, f"{patient_name}_emb_class.csv"), ret_class, delimiter=",")
    # Save patient embeddings
    #np.savetxt(os.path.join(patient_emb_dir, f"{file_name}.csv"), patient_embedding, delimiter=",")

print("Saved embeddings and attention scores for all patients")
