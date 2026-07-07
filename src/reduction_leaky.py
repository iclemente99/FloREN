import torch
import torch.utils.data as data
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import pandas as pd
import glob
import os

class AE(nn.Module):
    def __init__(self, dim, h_n):
        super(AE, self).__init__()
        self.dim = dim
        self.fc1 = nn.Linear(dim, 512)
        self.fc2 = nn.Linear(512, h_n)
        self.fc3 = nn.Linear(h_n, 512)
        self.fc4 = nn.Linear(512, dim)

    def encode(self, x):
        h1 = F.leaky_relu(self.fc1(x))
        return F.leaky_relu(self.fc2(h1))

    def decode(self, z):
        h3 = F.leaky_relu(self.fc3(z))
        return F.leaky_relu(self.fc4(h3))

    def forward(self, x):
        z = self.encode(x.view(-1, self.dim))
        return self.decode(z), z

def reduction(method, gene_cell, device, h_n, epochs):
    if method == 'AE':
        model_gene, model_cell, losses_gene, losses_cell = train_AE(gene_cell, device, h_n, epochs)
        return model_gene, model_cell, losses_gene, losses_cell
    elif method == 'raw':
        encoded = torch.tensor(gene_cell, dtype=torch.float32).to(device)
        encoded2 = torch.tensor(np.transpose(gene_cell), dtype=torch.float32).to(device)
        return None, None, [0], [0]

def train_AE(gene_cell, device, h_n, epochs=2000):
    gene = torch.tensor(gene_cell, dtype=torch.float32).to(device)
    cell = torch.tensor(np.transpose(gene_cell), dtype=torch.float32).to(device)

    # Batch sizes
    ba_gene = min(gene_cell.shape[0], 5000)
    ba_cell = min(gene_cell.shape[1], 5000)

    # Initialize models
    model_gene = AE(dim=gene_cell.shape[1], h_n=h_n).to(device)  # Input dim = total_cells
    model_cell = AE(dim=gene_cell.shape[0], h_n=h_n).to(device)  # Input dim = n_genes

    optimizer_gene = optim.Adam(model_gene.parameters(), lr=1e-3)
    optimizer_cell = optim.Adam(model_cell.parameters(), lr=1e-3)
    loss_func = nn.MSELoss()
    #EPOCH_AE = 2000
    EPOCH_AE = epochs
    losses_gene = np.zeros(EPOCH_AE)
    losses_cell = np.zeros(EPOCH_AE)

    use_amp = (device.type == 'cuda')
    scaler_gene = torch.cuda.amp.GradScaler(enabled=use_amp)
    scaler_cell = torch.cuda.amp.GradScaler(enabled=use_amp)

    loader_gene = data.DataLoader(gene, batch_size=ba_gene)
    loader_cell = data.DataLoader(cell, batch_size=ba_cell)

    for epoch in range(EPOCH_AE):
        epoch_loss_gene = 0.0
        epoch_loss_cell = 0.0
        num_batches = 0

        # Train gene AE
        for batch_x in loader_gene:
            with torch.autocast(device_type=device.type, enabled=use_amp):
                decoded, _ = model_gene(batch_x)
                loss = loss_func(batch_x, decoded)
            optimizer_gene.zero_grad()
            scaler_gene.scale(loss).backward()
            scaler_gene.step(optimizer_gene)
            scaler_gene.update()
            epoch_loss_gene += loss.item()
            num_batches += 1

        # Train cell AE
        for batch_x in loader_cell:
            with torch.autocast(device_type=device.type, enabled=use_amp):
                decoded, _ = model_cell(batch_x)
                loss = loss_func(batch_x, decoded)
            optimizer_cell.zero_grad()
            scaler_cell.scale(loss).backward()
            scaler_cell.step(optimizer_cell)
            scaler_cell.update()
            epoch_loss_cell += loss.item()
            num_batches += 1

        # Average losses
        losses_gene[epoch] = epoch_loss_gene / num_batches if num_batches > 0 else 0
        losses_cell[epoch] = epoch_loss_cell / num_batches if num_batches > 0 else 0
        print(f'Epoch {epoch+1}/{EPOCH_AE}: Gene loss: {losses_gene[epoch]:.12f}, Cell loss: {losses_cell[epoch]:.12f}')

    return model_gene, model_cell, losses_gene, losses_cell

def apply_AE(data, device, h_n, model, cell=False):
    # Transpose data for cell embeddings
    if cell:
        data = np.transpose(data)

    data_tensor = torch.tensor(data, dtype=torch.float32).to(device).contiguous()
    with torch.no_grad():
        _, encoded = model(data_tensor)

    return encoded
