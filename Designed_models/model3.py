import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scanpy as sc
import scipy
import anndata
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import os
import pandas as pd
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from torch_geometric.nn import GCNConv, GATConv
import optuna

train_adata = None
train_dataset = None
test_dataset = None
device = None

class GeneExpressionDataset(Dataset):
    def __init__(self, adata, perturbation_key='perturbation', scaler=None):
        self.adata = adata
        self.perturbation_key = perturbation_key
        
        if scipy.sparse.issparse(adata.X):
            data = adata.X.toarray()
        else:
            data = adata.X
            
        data = np.maximum(data, 0)
        data = np.maximum(data, 1e-10)
        data = np.log1p(data)
        
        if scaler is None:
            self.scaler = StandardScaler()
            self.expression_data = self.scaler.fit_transform(data)
        else:
            self.scaler = scaler
            self.expression_data = self.scaler.transform(data)
        
        self.expression_data = np.clip(self.expression_data, -10, 10)
        self.expression_data = self.expression_data / 10.0
        self.perturbations = pd.get_dummies(adata.obs[perturbation_key]).values
        
    def __len__(self):
        return len(self.adata)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.expression_data[idx]), torch.FloatTensor(self.perturbations[idx])

class EnhancedGEARS(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, num_genes=33694, n_layers=2, n_heads=8, dropout=0.1):
        super(EnhancedGEARS, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.expression_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.pert_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=n_heads, dropout=dropout)
        
        self.gat_layers = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim // n_heads, heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        self.combination_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def build_gene_graph(self, expression_data):
        corr_matrix = np.corrcoef(expression_data.T)
        threshold = np.percentile(np.abs(corr_matrix), 95)
        adj_matrix = (np.abs(corr_matrix) > threshold).astype(float)
        np.fill_diagonal(adj_matrix, 0)
        edge_weights = np.abs(corr_matrix)[adj_matrix > 0]
        edge_index = torch.tensor(scipy.sparse.csr_matrix(adj_matrix).nonzero(), dtype=torch.long)
        edge_weight = torch.tensor(edge_weights, dtype=torch.float)
        return edge_index, edge_weight
        
    def forward(self, x, pert, edge_index, edge_weight=None):
        print(f"Input shapes - x: {x.shape}, pert: {pert.shape}")
        
        if pert.shape[1] != self.input_dim:
            pert = F.pad(pert, (0, self.input_dim - pert.shape[1]))
        
        gene_features = self.expression_encoder(x)
        pert_features = self.pert_encoder(pert)
        
        gene_features = gene_features.unsqueeze(0)
        gene_features, _ = self.attention(gene_features, gene_features, gene_features)
        gene_features = gene_features.squeeze(0)
        
        for gat_layer in self.gat_layers:
            gene_features = gat_layer(gene_features, edge_index, edge_weight)
            gene_features = F.relu(gene_features)
        
        combined_pert = self.combination_encoder(torch.cat([gene_features, pert_features], dim=1))
        combined = torch.cat([gene_features, combined_pert], dim=1)
        fused = self.fusion(combined)
        output = self.output(fused)
        
        return output

def train_model(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        x, pert = batch
        x, pert = x.to(device), pert.to(device)
        
        edge_index, edge_weight = model.build_gene_graph(x.cpu().numpy())
        edge_index, edge_weight = edge_index.to(device), edge_weight.to(device)
        
        optimizer.zero_grad()
        output = model(x, pert, edge_index, edge_weight)
        loss = F.mse_loss(output, x)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(train_loader)

def evaluate_model(model, test_loader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in test_loader:
            x, pert = batch
            x, pert = x.to(device), pert.to(device)
            
            edge_index, edge_weight = model.build_gene_graph(x.cpu().numpy())
            edge_index, edge_weight = edge_index.to(device), edge_weight.to(device)
            
            output = model(x, pert, edge_index, edge_weight)
            loss = F.mse_loss(output, x)
            
            total_loss += loss.item()
            
    return total_loss / len(test_loader)

def main():
    global train_adata, train_dataset, test_dataset, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    print('Loading data...')
    train_path = "$path/to/your/dataset$/train.h5ad"
    test_path = "$path/to/your/dataset$/test.h5ad"
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Data files not found: {train_path} or {test_path}")
    
    train_adata = sc.read_h5ad(train_path)
    test_adata = sc.read_h5ad(test_path)
    
    print(f'Training data shape: {train_adata.X.shape}')
    print(f'Test data shape: {test_adata.X.shape}')
    
    scaler = StandardScaler()
    
    train_dataset = GeneExpressionDataset(train_adata, scaler=scaler)
    test_dataset = GeneExpressionDataset(test_adata, scaler=scaler)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = EnhancedGEARS(
        input_dim=train_dataset.expression_data.shape[1],
        hidden_dim=1024,
        n_layers=2,
        n_heads=8,
        dropout=0.1
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    print('Starting training...')
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    max_epochs = 100
    
    for epoch in range(max_epochs):
        train_loss = train_model(model, train_loader, optimizer, device)
        val_loss = evaluate_model(model, test_loader, device)
        
        print(f'Epoch {epoch+1}/{max_epochs}:')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'gears_best_model.pt')
            print(f"Saved best model with loss: {best_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    print('Training completed!')

if __name__ == '__main__':
    main()