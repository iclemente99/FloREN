################################################################
#
# MODULES
#
################################################################

import gc
import pandas as pd
import numpy as np
import os
import re
import random
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from warnings import filterwarnings
filterwarnings("ignore")

from sklearn.model_selection import train_test_split

def _t2np(t):
    """Tensor → numpy, compatible with numpy 1.x and 2.x."""
    t = t.detach().cpu().float()
    try:
        return t.numpy()
    except RuntimeError:
        return np.asarray(t)

import scanpy as sc
import psutil

from scipy.sparse import load_npz

from reduction_leaky import reduction, apply_AE
from utils import debuginfoStr, build_graph
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
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Arguments
parser = argparse.ArgumentParser(description='Training GNN on gene-cell graph')

# Sampling
parser.add_argument('--n_batch', type=int, default=25, help='Number of batches (sampled graphs) per epoch')
parser.add_argument('--cell_rate', type=float, default=0.1)
parser.add_argument('--gene_rate', type=float, default=0.1)

# Result
parser.add_argument('--data_name', type=str, default='FloREN', help='Dataset name tag')
parser.add_argument('--reduction', type=str, default='AE', help='Feature extraction method: pca, raw, AE')
parser.add_argument('--in_dim', type=int, default=256, help='AE bottleneck dimension')
parser.add_argument('--floren_grn', type=str, default='True', help='Use FloREN GRN inference')

# HGTSSL
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--n_hid', type=int, default=128, help='GNN hidden dimension')
parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
parser.add_argument('--n_layers', type=int, default=2, help='Number of GNN layers')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout ratio')
parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
parser.add_argument('--batch_size', type=int, help='Number of output nodes for training')
parser.add_argument('--layer_type', type=str, default='hgt', help='GNN layer type')
parser.add_argument('--loss', type=str, default='kl', help='Loss for GAE')
parser.add_argument('--factor', type=float, default=0.5, help='LR scheduler attenuation factor')
parser.add_argument('--rf', type=float, default=0.0, help='Regularization weight')
parser.add_argument('--cuda', type=int, default=1, help='0 = GPU0, else CPU')
parser.add_argument('--rep', type=str, default='T', help='Precision truncation')
parser.add_argument('--AEtype', type=int, default=1, help='AE type: 1=embedding AE, 2=HGT node AE')
parser.add_argument('--optimizer', type=str, default='adamw', help='Optimizer')
parser.add_argument('--test_size', type=float, default=0.2, help='Fraction of data for test set')
parser.add_argument('--val_size', type=float, default=0.2, help='Fraction of data for validation set')
parser.add_argument('--patience', type=int, default=30, help='Epochs without val improvement before early stop')
parser.add_argument('--resume', action='store_true',
                    help='Resume training from checkpoint_latest.pt saved in model_dir')

# Data directories
parser.add_argument('--data_path', default='./data/', type=str, help='Path to FloREN data')
parser.add_argument('--output_path', default='./floren_output', type=str, help='Path to graph construction outputs')
parser.add_argument('--cell_comm_path', default=None, type=str, help='Path to cell-cell communication matrices')
parser.add_argument('--result_dir', default=None, type=str, help='Path for FloREN outputs (defaults to output_path)')
parser.add_argument('--patient_id', default='patient_id', type=str,
                    help='adata.obs column with the patient identifier')
parser.add_argument('--metadata_group', default='disease', type=str,
                    help='adata.obs column with the group label')
parser.add_argument('--count_layer', default='logcounts', type=str,
                    help='adata.layer name with log-normalized counts')
parser.add_argument('--min_count', default=0, type=int,
                    help='Minimum expression count to establish a gene-cell edge (exclusive)')
parser.add_argument('--adata_path', default='./data/binvignat_example.h5ad', type=str,
                    help='Path to adata object')
parser.add_argument('--job_cache_dir', default=None, type=str,
                    help='Directory for pre-computed subgraph job cache (defaults to result_dir/job_cache)')

args = parser.parse_args()

sample_name = args.data_name
patient_id = args.patient_id
count_layer = args.count_layer
metadata_group = args.metadata_group
min_count = args.min_count

# result_dir falls back to output_path if not explicitly provided
if args.result_dir is None:
    args.result_dir = args.output_path

job_cache_dir = args.job_cache_dir if args.job_cache_dir else os.path.join(args.result_dir, "job_cache")
os.makedirs(job_cache_dir, exist_ok=True)

output_path = args.output_path
embeddings_path = os.path.join(output_path, "embeddings")
connections_path = os.path.join(output_path, "connections")
gene_embeddings_norm_path = os.path.join(embeddings_path, "gene_embeddings_norm")
cell_embeddings_norm_path = os.path.join(embeddings_path, "cell_embeddings_norm")

if args.cuda == 0:
    device = torch.device("cuda:0")
    print("Using CUDA GPU0")
else:
    device = torch.device("cpu")
    print("Using CPU")

# AMP setup
use_amp = (device.type == 'cuda')
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

#---------------------------------------------------------------
#
#                 DATA LOADING
#
#---------------------------------------------------------------

print("Loading adata object")
adata = sc.read_h5ad(args.adata_path)
inds = np.unique(adata.obs[patient_id].values.astype(str))
gene_names = adata.var_names
n_genes = len(adata.var_names)

h_n = min(n_genes, args.in_dim)

#---------------------------------------------------------------
#
#            GENERATE HETEROGENEOUS PATIENT GRAPHS
#
#---------------------------------------------------------------

cell_names = pd.read_csv(os.path.join(output_path, "All_AUC_Cell_names.csv"))
samples_names = []
for name in cell_names['0']:
    samples_names.append(re.split('__', name, 2)[0])

files = list(set(samples_names))
files = [f[1:] if f.startswith("\\") else f for f in files]

patient_graphs = []

print("\nGENERATING PATIENT GRAPHS")
for patient_name in files:
    print(f"    Processing patient: {patient_name}")

    # Gene-cell adjacency: binarize expression matrix by count threshold
    adata_subset = adata[adata.obs[patient_id].isin([patient_name])]
    _layer = adata_subset.layers[count_layer]
    transformed_matrix = (_layer.toarray() if hasattr(_layer, "toarray") else np.asarray(_layer)).T.copy()
    gene_cell = transformed_matrix
    gene_cell[gene_cell > min_count] = 1   # strictly greater than: zeros stay zero
    gene_cell[gene_cell <= min_count] = 0

    # Gene-gene adjacency
    npz_path = os.path.join(connections_path, f"{patient_name}_gene_gene_connections.npz")
    csv_path = os.path.join(connections_path, f"{patient_name}_gene_gene_connections.csv")

    if os.path.exists(npz_path):
        genes_f = pd.DataFrame(
            load_npz(npz_path).toarray(),
            index=sorted(gene_names), columns=sorted(gene_names)
        )
    else:
        # Fallback to CSV for backward compatibility
        gene_gene = pd.read_csv(csv_path)
        gene_gene.drop(columns=gene_gene.columns[0], axis=1, inplace=True)
        gene_gene.set_index(gene_gene.columns, inplace=True)
        genes_intrsct = list(set(gene_names) & set(gene_gene.columns))
        genes_f = gene_gene[genes_intrsct]
        genes_f = genes_f.loc[genes_intrsct]
        missing = list(set(gene_names) - set(genes_intrsct))
        genes_f = np.concatenate([np.array(genes_f), np.zeros((genes_f.shape[0], len(missing)))], axis=1)
        genes_f = pd.DataFrame(
            np.concatenate([genes_f, np.zeros((len(missing), genes_f.shape[1]))], axis=0))
        genes_f.columns = genes_intrsct + missing
        genes_f.set_index(genes_f.columns, inplace=True)
        genes_f = genes_f.reindex(sorted(genes_f.columns), axis=1)
        genes_f = genes_f.reindex(sorted(genes_f.index), axis=0)
        np.fill_diagonal(genes_f.values, 0)
        genes_f[genes_f > 0] = 1

    # Cell-cell adjacency
    if args.cell_comm_path is None:
        cells_f = pd.DataFrame(np.zeros([gene_cell.shape[1], gene_cell.shape[1]]))
    else:
        cells_f = pd.read_csv(os.path.join(args.cell_comm_path, f"{patient_name}.csv"))
        cells_f.set_index(cells_f.columns[0], inplace=True)

    # Gene and cell embeddings — kept on CPU
    genes_files = pd.read_csv(os.path.join(gene_embeddings_norm_path, f"{patient_name}_AE_Emb_Genes.csv"))
    genes_files.drop(genes_files.columns[0], axis=1, inplace=True)
    encoded = torch.tensor(genes_files.values, dtype=torch.float32)   # stays on CPU

    cells_files = pd.read_csv(os.path.join(cell_embeddings_norm_path, f"{patient_name}_AE_Emb_Cells.csv"))
    cells_files.drop(cells_files.columns[0], axis=1, inplace=True)
    encoded2 = torch.tensor(cells_files.values, dtype=torch.float32)   # stays on CPU

    graph = build_graph(gene_cell, genes_f, cells_f, encoded, encoded2)

    patient_graphs.append({
        "name": patient_name,
        "graph": graph,
        "cells_idx": cell_names[cell_names['0'].str.contains(patient_name)].index.values
    })

print(f"Built {len(patient_graphs)} patient graphs")
debuginfoStr('Build Graph finished')

################################################################
#
# OUTPUT DIRECTORY SETUP
#
################################################################

file0 = f'sample_{sample_name}_epoch_{args.epochs}_n_hid_{args.n_hid}_nheads_{args.n_heads}_lr_01_n_batch{args.n_batch}'
gene_dir  = args.result_dir + '/gene/'
cell_dir  = args.result_dir + '/cell/'
model_dir = args.result_dir + '/model/'
att_dir   = args.result_dir + '/att/'
loss_dir  = args.result_dir + '/loss/'
inter_dir = args.result_dir + '/interpretability/'
os.makedirs(gene_dir, exist_ok=True)
os.makedirs(cell_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(att_dir, exist_ok=True)
os.makedirs(loss_dir, exist_ok=True)
os.makedirs(inter_dir, exist_ok=True)

checkpoint_path = os.path.join(model_dir, 'checkpoint_latest.pt')

################################################################
#
# TRAIN / VAL / TEST SPLIT
#
################################################################

print("Splitting into training, validation, and test sets")
metadata = adata.obs[[patient_id, metadata_group]]
metadata = metadata.groupby(patient_id).first().reset_index()
metadata["group"] = metadata[metadata_group].astype("category").cat.codes
num_groups = int(metadata['group'].max()) + 1


def stratified_split_graphs(patient_graphs, metadata_df, test_size=0.15, val_size=0.15, random_state=42):
    meta_lookup = dict(zip(metadata_df[patient_id], metadata_df["group"]))
    patient_names = [pg["name"] for pg in patient_graphs]
    labels = [meta_lookup[name] for name in patient_names]
    n_classes = len(set(labels))
    # Fall back to non-stratified split if dataset is too small for stratified sampling
    use_stratify = int(len(labels) * test_size) >= n_classes
    train_val_graphs, test_graphs, train_val_labels, test_labels = train_test_split(
        patient_graphs, labels,
        test_size=test_size, stratify=labels if use_stratify else None, random_state=random_state
    )
    relative_val_size = val_size / (1 - test_size)
    use_stratify_val = int(len(train_val_labels) * relative_val_size) >= n_classes
    train_graphs, val_graphs, train_labels, val_labels = train_test_split(
        train_val_graphs, train_val_labels,
        test_size=relative_val_size, stratify=train_val_labels if use_stratify_val else None, random_state=random_state
    )
    return train_graphs, val_graphs, test_graphs


train_graphs, val_graphs, test_graphs = stratified_split_graphs(
    patient_graphs, metadata, test_size=args.test_size, val_size=args.val_size
)
print(f"Train: {len(train_graphs)} | Val: {len(val_graphs)} | Test: {len(test_graphs)}")
debuginfoStr('Graph split finished')

################################################################
#
# MODEL AND OPTIMIZER
#
################################################################

# FIX 2: gradient checkpointing trades recomputation for activation memory.
# For large graphs (thousands of nodes, millions of edges) this can cut backward-pass
# RAM by 50-60% at the cost of ~30% extra forward compute per layer.
gnn = GNN(
    conv_name=args.layer_type,
    in_dim=h_n,
    n_hid=args.n_hid,
    n_heads=args.n_heads,
    n_layers=args.n_layers,
    dropout=args.dropout,
    num_types=2,
    num_relations=7,
    use_RTE=False,
    n_labels=num_groups,
    use_grad_checkpoint=True,
).to(device)

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
# LOSS FUNCTIONS
#
################################################################

contrastive_loss_fn = nn.BCEWithLogitsLoss()
classification_loss_fn = nn.CrossEntropyLoss()

nb_epochs = args.epochs
weight_contrastive = 0.1
weight_classification = 0.9

patience = args.patience
counter = 0
best_model_loss = float('inf')

training_losses = []
validation_losses = []
contrastive_losses = []
classification_losses = []
total_losses = []

start_epoch = 0
best_model_path = None   # set when a new best is saved; persisted in checkpoint
if args.resume:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"--resume set but no checkpoint found at {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    gnn.load_state_dict(ckpt['gnn_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    scheduler.load_state_dict(ckpt['scheduler_state'])
    start_epoch           = ckpt['epoch'] + 1
    best_model_loss       = ckpt['best_model_loss']
    best_model_path       = ckpt.get('best_model_path')
    counter               = ckpt['counter']
    training_losses       = ckpt['training_losses']
    validation_losses     = ckpt['validation_losses']
    contrastive_losses    = ckpt['contrastive_losses']
    classification_losses = ckpt['classification_losses']
    print(f"Resumed from epoch {start_epoch} | Best val loss so far: {best_model_loss:.4f}")

################################################################
#
# DISK-CACHED SUBGRAPH JOBS
#
################################################################

def save_jobs_to_disk(graphs, gene_rate, cell_rate, cache_dir):
    """Generate subgraph jobs for all patients and save each as a .pt file on CPU.
    Skips patients whose cache file already exists."""
    for patient in graphs:
        patient_name = patient['name']
        cache_path = os.path.join(cache_dir, f"{patient_name}.pt")
        if os.path.exists(cache_path):
            continue

        graph = patient['graph']
        adata_subset = adata[adata.obs[patient_id].isin([patient_name])]
        _layer = adata_subset.layers[count_layer]
        transformed_matrix = (_layer.toarray() if hasattr(_layer, "toarray") else np.asarray(_layer)).T
        gene_cell = transformed_matrix

        jobs = []
        gene_num = int(gene_cell.shape[0] * gene_rate)
        cell_num = int(gene_cell.shape[1] * cell_rate)
        jobs.append(sub_sample(graph, gene_cell, gene_cell.shape[1], gene_cell.shape[0], gene_cell.shape[0], gene_cell.shape[1], query=True))
        jobs.append(sub_sample(graph, gene_cell, cell_num, gene_num, gene_cell.shape[0], gene_cell.shape[1]))
        jobs.append(sub_sample(graph, gene_cell, cell_num, gene_num, gene_cell.shape[0], gene_cell.shape[1]))

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
            node_num = 0
            types = graph.get_types()
            for t in types:
                node_dict[t] = [node_num, len(node_dict)]
                node_num += len(feature[t])
            for t in types:
                t_i = node_dict[t][1]
                # Store as float16 on CPU to halve disk/RAM usage
                node_feature.insert(t_i, torch.tensor(feature[t], dtype=torch.float16))
                node_time += list(time_info[t])
                node_type += [node_dict[t][1] for _ in range(len(feature[t]))]
            edge_dict = {e[2]: i for i, e in enumerate(graph.get_meta_graph())}
            edge_dict['self'] = len(edge_dict)
            for target_type in edge_list:
                for source_type in edge_list[target_type]:
                    for relation_type in edge_list[target_type][source_type]:
                        for ti, si in edge_list[target_type][source_type][relation_type]:
                            tid = ti + node_dict[target_type][0]
                            sid = si + node_dict[source_type][0]
                            edge_index.append([sid, tid])
                            edge_type.append(edge_dict.get(relation_type, 0))
                            edge_time.append(120)
            node_feature = torch.cat((node_feature[0], node_feature[1]), 0)  # float16 on CPU
            node_type = torch.IntTensor(node_type)   # int32 on CPU
            edge_time = torch.IntTensor(edge_time)   # int32 on CPU
            edge_index = torch.LongTensor(edge_index).t()  # keep int64 (2 x E)
            edge_type = torch.IntTensor(edge_type)   # int32 on CPU
            formatted_jobs.append([node_feature, node_type, edge_time, edge_index, edge_type, og_gene_indxs, og_cell_indxs])

        torch.save(formatted_jobs, cache_path)


def load_patient_jobs(patient_name, cache_dir, device):
    """Load one patient's jobs from disk and move tensors to device.
    Casts float16 -> float32 and int32 -> int64 on load."""
    cache_path = os.path.join(cache_dir, f"{patient_name}.pt")
    formatted_jobs = torch.load(cache_path, map_location='cpu')
    result = []
    for job in formatted_jobs:
        node_feature, node_type, edge_time, edge_index, edge_type, og_gene_indxs, og_cell_indxs = job
        node_feature = node_feature.float().to(device)   # float16 -> float32
        node_type    = node_type.long().to(device)       # int32 -> int64
        edge_time    = edge_time.long().to(device)       # int32 -> int64
        edge_index   = edge_index.to(device)             # already int64
        edge_type    = edge_type.long().to(device)       # int32 -> int64
        result.append([node_feature, node_type, edge_time, edge_index, edge_type, og_gene_indxs, og_cell_indxs])
    return result


# Cache ALL patients to disk once before training
print("Caching subgraph jobs to disk...")
save_jobs_to_disk(patient_graphs, args.gene_rate, args.cell_rate, job_cache_dir)
debuginfoStr('Pre-generated jobs for all patients')

# FIX 1+4: free the heavy Graph objects and adata expression layers now that all
# subgraph jobs are safely cached to disk.  save_jobs_to_disk() was the last
# consumer of both; training only needs patient names and the .pt files.
for _p in patient_graphs:
    _p.pop('graph', None)
    _p.pop('cells_idx', None)
del patient_graphs
for _lyr in list(adata.layers.keys()):
    del adata.layers[_lyr]
gc.collect()
debuginfoStr('Graph objects and adata layers freed')

gnn.to(device)


def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_gb = process.memory_info().rss / (1024 ** 3)
    gpu_mem_gb = 0
    if torch.cuda.is_available():
        gpu_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        torch.cuda.reset_peak_memory_stats()
    return mem_gb, gpu_mem_gb

################################################################
#
# TRAINING LOOP
#
################################################################

print("\nSTARTING FLOREN TRAINING")
loop_start = time.time()
for epoch in range(start_epoch, nb_epochs):
    epoch_start = time.time()
    gnn.train()
    random.shuffle(train_graphs)
    epoch_contrastive = 0.0
    epoch_classification = 0.0
    epoch_total = 0.0

    for patient in train_graphs:
        patient_name = patient['name']
        grp = metadata.loc[metadata[patient_id] == patient_name, 'group'].values
        if len(grp) == 0:
            raise ValueError(f"Patient {patient_name} not found in metadata")
        group_id = int(grp[0])
        labels_class = torch.tensor([group_id], device=device, dtype=torch.long)

        gnn_input = load_patient_jobs(patient_name, job_cache_dir, device)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits, ret_class = gnn.forward(
                gnn_input[0][0], gnn_input[0][1], gnn_input[0][2], gnn_input[0][3], gnn_input[0][4],
                gnn_input[1][0], gnn_input[1][1], gnn_input[1][2], gnn_input[1][3], gnn_input[1][4],
                gnn_input[2][0], gnn_input[2][1], gnn_input[2][2], gnn_input[2][3], gnn_input[2][4]
            )

            num_nodes = gnn_input[0][0].shape[0]
            lbl_1 = torch.ones(1, num_nodes).to(device)
            lbl_2 = torch.zeros(1, num_nodes).to(device)
            lbl = torch.cat((lbl_1, lbl_2), dim=1)

            contrastive_loss = contrastive_loss_fn(logits, lbl)
            classification_loss = classification_loss_fn(ret_class.view(1, -1), labels_class)
            loss = weight_contrastive * contrastive_loss + weight_classification * classification_loss

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_contrastive += contrastive_loss.item()
        epoch_classification += classification_loss.item()
        epoch_total += loss.item()

        del gnn_input
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()  # FIX 3: prompt Python GC to release freed tensors between patients

    epoch_contrastive /= len(train_graphs)
    epoch_classification /= len(train_graphs)
    epoch_total /= len(train_graphs)

    training_losses.append(epoch_total)
    contrastive_losses.append(epoch_contrastive)
    classification_losses.append(epoch_classification)
    total_losses.append(epoch_total)

    # Validation
    gnn.eval()
    val_contrastive = 0.0
    val_classification = 0.0
    val_total = 0.0
    val_correct = 0
    val_total_count = 0

    with torch.no_grad():
        for patient in val_graphs:
            patient_name = patient['name']
            grp = metadata.loc[metadata[patient_id] == patient_name, 'group'].values
            if len(grp) == 0:
                raise ValueError(f"Patient {patient_name} not found in metadata")
            group_id = int(grp[0])
            labels_class = torch.tensor([group_id], device=device, dtype=torch.long)

            gnn_input = load_patient_jobs(patient_name, job_cache_dir, device)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits, ret_class = gnn.forward(
                    gnn_input[0][0], gnn_input[0][1], gnn_input[0][2], gnn_input[0][3], gnn_input[0][4],
                    gnn_input[1][0], gnn_input[1][1], gnn_input[1][2], gnn_input[1][3], gnn_input[1][4],
                    gnn_input[2][0], gnn_input[2][1], gnn_input[2][2], gnn_input[2][3], gnn_input[2][4]
                )

                num_nodes = gnn_input[0][0].shape[0]
                lbl_1 = torch.ones(1, num_nodes).to(device)
                lbl_2 = torch.zeros(1, num_nodes).to(device)
                lbl = torch.cat((lbl_1, lbl_2), dim=1)

                contrastive_loss = contrastive_loss_fn(logits, lbl)
                classification_loss = classification_loss_fn(ret_class.view(1, -1), labels_class)
                loss = weight_contrastive * contrastive_loss + weight_classification * classification_loss

            val_contrastive += contrastive_loss.item()
            val_classification += classification_loss.item()
            val_total += loss.item()

            outputs = torch.softmax(ret_class, dim=-1)
            predicted = torch.argmax(outputs).item()
            true_label = labels_class.item()
            val_correct += (predicted == true_label)
            val_total_count += 1

            del gnn_input
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

    val_contrastive /= len(val_graphs)
    val_classification /= len(val_graphs)
    val_total /= len(val_graphs)
    val_acc = val_correct / val_total_count
    validation_losses.append(val_total)

    # Step scheduler on validation loss
    scheduler.step(val_total)

    print(f"Epoch {epoch + 1}/{nb_epochs} | Train loss: {epoch_total:.6f} | "
          f"Train Contr: {epoch_contrastive:.6f} | Train Class: {epoch_classification:.6f} | "
          f"Val loss: {val_total:.6f} | Val acc: {val_acc:.4f} | "
          f"time: {(time.time() - epoch_start) / 60:.1f} min")

    # Early stopping on validation loss
    model0 = f'{args.data_name}_epoch_{args.epochs}_n_hid_{args.n_hid}_nheads_{args.n_heads}_lr_01_n_batch{args.n_batch}'
    if val_total < best_model_loss:
        best_model_loss = val_total
        counter = 0
        best_model_path = os.path.join(model_dir, model0)
        state = {
            'model': gnn.state_dict(),
            'optimizer': scheduler.state_dict(),
            'epoch': epoch
        }
        os.makedirs(model_dir, exist_ok=True)
        torch.save(state, best_model_path)
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}. Best val loss: {best_model_loss:.6f}")
            break

    # Save latest checkpoint so training can be resumed if killed
    torch.save({
        'epoch':               epoch,
        'gnn_state':           gnn.state_dict(),
        'optimizer_state':     optimizer.state_dict(),
        'scheduler_state':     scheduler.state_dict(),
        'best_model_loss':     best_model_loss,
        'best_model_path':     best_model_path,
        'counter':             counter,
        'training_losses':     training_losses,
        'validation_losses':   validation_losses,
        'contrastive_losses':  contrastive_losses,
        'classification_losses': classification_losses,
    }, checkpoint_path)

# Save loss history
loss_history = pd.DataFrame({
    "epoch": list(range(1, len(training_losses) + 1)),
    "train_total": training_losses,
    "train_contrastive": contrastive_losses,
    "train_classification": classification_losses,
    "val_total": validation_losses
})
loss_history.to_csv(os.path.join(loss_dir, model0 + ".csv"), index=False)
debuginfoStr('FloREN training finished')

ram_usage, vram_usage = get_memory_usage()
print(f"Training complete | Best val loss: {best_model_loss:.4f} | "
      f"RAM: {ram_usage:.2f}GB | VRAM: {vram_usage:.2f}GB | "
      f"Time: {(time.time() - loop_start) / 60:.1f} min")

# Save final model checkpoint
state = {'model': gnn.state_dict(), 'optimizer': scheduler.state_dict(), 'epoch': epoch}
final_model0 = f'Final_{args.data_name}_epoch_{args.epochs}_n_hid_{args.n_hid}_nheads_{args.n_heads}_lr_01_n_batch{args.n_batch}'
torch.save(state, model_dir + final_model0)

################################################################
#
# APPLY GNN TO FULL GRAPHS (INFERENCE)
#
################################################################

gene_dir       = args.result_dir + '/floren_gene_embeddings/'
cell_dir       = args.result_dir + '/floren_cell_embeddings/'
att_dir        = args.result_dir + '/floren_attention_embeddings/'
patient_emb_dir       = args.result_dir + '/floren_patient_embeddings/'
patient_emb_dir_split    = args.result_dir + '/floren_patient_embeddings/split/'
patient_emb_dir_patients = args.result_dir + '/floren_patient_embeddings/patients/'
patient_emb_dir_layers   = args.result_dir + '/floren_patient_embeddings/layers/'
os.makedirs(gene_dir, exist_ok=True)
os.makedirs(cell_dir, exist_ok=True)
os.makedirs(att_dir, exist_ok=True)
os.makedirs(patient_emb_dir, exist_ok=True)
os.makedirs(patient_emb_dir_split, exist_ok=True)
os.makedirs(patient_emb_dir_patients, exist_ok=True)
os.makedirs(patient_emb_dir_layers, exist_ok=True)

# Load best saved model (use tracked path so resume runs find the original best)
_best_path = best_model_path if best_model_path and os.path.exists(best_model_path) \
             else os.path.join(model_dir, model0)
state = torch.load(_best_path, map_location=lambda storage, loc: storage)
print(f"Loaded best model from {_best_path}")

# Re-instantiate GNN with the same in_dim used during training (h_n)
gnn = GNN(
    conv_name=args.layer_type,
    in_dim=h_n,
    n_hid=args.n_hid,
    n_heads=args.n_heads,
    n_layers=args.n_layers,
    dropout=args.dropout,
    num_types=2,
    num_relations=7,
    n_labels=num_groups,
    use_RTE=False
).to(device)
gnn.load_state_dict(state['model'])
gnn.eval()

all_graphs = train_graphs + val_graphs + test_graphs

_layer_accum = {n: {} for n in ['ssl', 'g32', 'l32', 'g64', 'l64', 'g128', 'l128', 'class']}

print("\nSAVING PATIENT REPRESENTATIONS")
for patient in all_graphs:
    patient_name = patient['name']
    print(f"    Saving patient: {patient_name}")
    gnn_input = load_patient_jobs(patient_name, job_cache_dir, device)
    edge_index = gnn_input[0][3]

    with torch.no_grad():
        logits, ret_class = gnn.forward(
            gnn_input[0][0], gnn_input[0][1], gnn_input[0][2], gnn_input[0][3], gnn_input[0][4],
            gnn_input[1][0], gnn_input[1][1], gnn_input[1][2], gnn_input[1][3], gnn_input[1][4],
            gnn_input[2][0], gnn_input[2][1], gnn_input[2][2], gnn_input[2][3], gnn_input[2][4]
        )
        att_full  = gnn.att_0.abs() + gnn.att_n.abs()
        att_final = (att_full / 2).mean(axis=1)
        cell_embs = (gnn.cells_0 + gnn.cells_n) / 2
        gene_embs = (gnn.genes_0 + gnn.genes_n) / 2
        emb_nhid = _t2np(gnn.emb_104)
        g_32  = _t2np(gnn.g_32)
        l_32  = _t2np(gnn.l_32)
        g_64  = _t2np(gnn.g_64)
        l_64  = _t2np(gnn.l_64)
        g_128 = _t2np(gnn.g_128)
        l_128 = _t2np(gnn.l_128)

    patient_embedding = np.concatenate(
        [arr.flatten() for arr in [emb_nhid, g_32, l_32, g_64, l_64, g_128, l_128, _t2np(ret_class)]]
    )

    # Save edge attentions
    positions = pd.DataFrame(_t2np(gnn_input[0][3].T))
    df = pd.DataFrame(_t2np(att_final))
    df2 = pd.concat([positions, df], axis=1)
    df2.to_csv(os.path.join(att_dir, f"{patient_name}_edge_att.csv"), sep=",", index=True)

    adata_subset = adata[adata.obs[patient_id].isin([patient_name])]

    # Save cell and gene embeddings
    pd.DataFrame(_t2np(cell_embs), index=adata_subset.obs_names).to_csv(
        os.path.join(cell_dir, f"{patient_name}_cell_embs.csv"))
    pd.DataFrame(_t2np(gene_embs), index=adata_subset.var_names).to_csv(
        os.path.join(gene_dir, f"{patient_name}_gene_embs.csv"))

    # Save node attention scores
    num_nodes = gnn_input[0][0].shape[0]
    node_attention = np.zeros(num_nodes, dtype=np.float64)
    np.add.at(node_attention, df2.iloc[:, 0].values.astype(int), df2.iloc[:, 2].values)
    np.add.at(node_attention, df2.iloc[:, 1].values.astype(int), df2.iloc[:, 2].values)
    genes_grad = node_attention[:n_genes]
    cells_grad = node_attention[n_genes:]
    patient_out = os.path.join(inter_dir, patient_name)
    os.makedirs(patient_out, exist_ok=True)
    pd.DataFrame(genes_grad, index=adata_subset.var_names).to_csv(
        os.path.join(patient_out, "genes_atts.csv"))
    pd.DataFrame(cells_grad, index=adata_subset.obs_names).to_csv(
        os.path.join(patient_out, "cells_atts.csv"))

    # Save per-patient split components (one file per layer per patient)
    np.savetxt(os.path.join(patient_emb_dir_split, f"{patient_name}_emb_ssl.csv"),  emb_nhid, delimiter=",")
    np.savetxt(os.path.join(patient_emb_dir_split, f"{patient_name}_emb_g32.csv"),  g_32,     delimiter=",")
    np.savetxt(os.path.join(patient_emb_dir_split, f"{patient_name}_emb_l32.csv"),  l_32,     delimiter=",")
    np.savetxt(os.path.join(patient_emb_dir_split, f"{patient_name}_emb_g64.csv"),  g_64,     delimiter=",")
    np.savetxt(os.path.join(patient_emb_dir_split, f"{patient_name}_emb_l64.csv"),  l_64,     delimiter=",")
    np.savetxt(os.path.join(patient_emb_dir_split, f"{patient_name}_emb_g128.csv"), g_128,    delimiter=",")
    np.savetxt(os.path.join(patient_emb_dir_split, f"{patient_name}_emb_l128.csv"), l_128,    delimiter=",")
    np.savetxt(os.path.join(patient_emb_dir_split, f"{patient_name}_emb_class.csv"), _t2np(ret_class), delimiter=",")

    # Save concatenated embedding per patient in /patients/
    np.savetxt(os.path.join(patient_emb_dir_patients, f"{patient_name}.csv"), patient_embedding, delimiter=",")

    # Accumulate layer vectors for the /layers/ CSVs (rows=patients, cols=dims)
    _layer_accum['ssl'][patient_name]   = emb_nhid.flatten()
    _layer_accum['g32'][patient_name]   = g_32.flatten()
    _layer_accum['l32'][patient_name]   = l_32.flatten()
    _layer_accum['g64'][patient_name]   = g_64.flatten()
    _layer_accum['l64'][patient_name]   = l_64.flatten()
    _layer_accum['g128'][patient_name]  = g_128.flatten()
    _layer_accum['l128'][patient_name]  = l_128.flatten()
    _layer_accum['class'][patient_name] = _t2np(ret_class).flatten()

    # Keep legacy flat file in root for backwards compatibility
    file_name = f'sample_{patient_name}_epoch_{args.epochs}_n_hid_{args.n_hid}_nheads_{args.n_heads}_lr_01_n_batch{args.n_batch}'
    np.savetxt(os.path.join(patient_emb_dir, f"{file_name}.csv"), patient_embedding, delimiter=",")

    del gnn_input
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

print("Saved embeddings and attention scores for all patients")

# Save /layers/ CSVs: one file per layer, rows=patients, columns=embedding dims
for _lname, _rows in _layer_accum.items():
    _df = pd.DataFrame(_rows).T   # patients × dims
    _df.index.name = 'patient'
    _df.to_csv(os.path.join(patient_emb_dir_layers, f"layer_{_lname}.csv"))
print(f"Saved per-layer patient matrices → {patient_emb_dir_layers}")
