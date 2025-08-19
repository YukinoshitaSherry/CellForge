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
from transformers import AutoModel, AutoTokenizer
from torch_geometric.nn import GCNConv, GATConv
import optuna


train_adata = None
train_dataset = None
test_dataset = None
device = None


class SimpleScGPT(nn.Module):
    def __init__(self, n_genes, n_hidden=1024, n_layers=4, n_heads=8, dropout=0.1):
        super().__init__()
        self.n_genes = n_genes
        self.n_hidden = n_hidden
        self.embedding = nn.Linear(n_genes, n_hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_hidden,
            nhead=n_heads,
            dim_feedforward=n_hidden*2,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(n_hidden)
        self.head = nn.Linear(n_hidden, n_hidden)

    def forward(self, x):
        
        x_emb = self.embedding(x)  
        
        x_emb = x_emb.unsqueeze(1)  
        x_trans = self.transformer(x_emb)  
        x_trans = self.norm(x_trans)
        x_out = self.head(x_trans.squeeze(1))  
        return x_out

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

class EnhancedScGPT(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, num_genes=33694):
        super(EnhancedScGPT, self).__init__()
        
        
        self.scgpt = SimpleScGPT(
            n_genes=input_dim,
            n_hidden=hidden_dim,
            n_layers=4,
            n_heads=8,
            dropout=0.1
        )
        
        
        self.expression_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        
        self.pert_encoder = nn.Sequential(
            nn.Linear(48, hidden_dim),  
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim)
        )
        
        
        print(f"Model input dimension: {input_dim}")
        print(f"Model hidden dimension: {hidden_dim}")
        print(f"Model gene count: {num_genes}")
        print(f"Fusion layer input dimension: {hidden_dim * 3}")
    
    def forward(self, x, pert, edge_index=None):
        batch_size = x.size(0)
        
        
        scgpt_features = self.scgpt(x)
        
        
        gene_features = self.expression_encoder(x)
        
        
        pert_features = self.pert_encoder(pert)
        
        
        
        gene_features = gene_features.view(batch_size, 1, -1)
        
        
        attn_output, _ = self.self_attention(
            gene_features, gene_features, gene_features
        )
        
        
        attn_features = attn_output.squeeze(1)
        
        
        combined = torch.cat([scgpt_features, attn_features, pert_features], dim=1)
        
        
        fused = self.fusion(combined)
        
        
        output = self.output(fused)
        
        return output

def train_model(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc='Training', leave=False):
        x, pert = batch
        x, pert = x.to(device), pert.to(device)
        optimizer.zero_grad()
        output = model(x, pert)
        loss = F.mse_loss(output, x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate_model(model, test_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Validating', leave=False):
            x, pert = batch
            x, pert = x.to(device), pert.to(device)
            output = model(x, pert)
            loss = F.mse_loss(output, x)
            total_loss += loss.item()
    return total_loss / len(test_loader)

def calculate_metrics(pred, true):
    mse = np.mean((pred - true) ** 2)
    pcc = np.mean([pearsonr(p, t)[0] for p, t in zip(pred.T, true.T)])
    r2 = np.mean([r2_score(t, p) for p, t in zip(pred.T, true.T)])
    
    
    std = np.std(true, axis=0)
    de_mask = np.abs(true - np.mean(true, axis=0)) > std
    if np.any(de_mask):
        mse_de = np.mean((pred[de_mask] - true[de_mask]) ** 2)
        pcc_de = np.mean([pearsonr(p[m], t[m])[0] for p, t, m in zip(pred.T, true.T, de_mask.T)])
        r2_de = np.mean([r2_score(t[m], p[m]) for p, t, m in zip(pred.T, true.T, de_mask.T)])
    else:
        mse_de = pcc_de = r2_de = np.nan
        
    return {
        'MSE': mse,
        'PCC': pcc,
        'R2': r2,
        'MSE_DE': mse_de,
        'PCC_DE': pcc_de,
        'R2_DE': r2_de
    }

def evaluate_and_save_model(model, test_loader, device, save_path='method1_scgpt_best.pt'):

    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            x, pert = batch
            x, pert = x.to(device), pert.to(device)
            output = model(x, pert)
            
            
            all_predictions.append(output.cpu().numpy())
            all_targets.append(x.cpu().numpy())
    
    
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    
    results = calculate_metrics(all_predictions, all_targets)
    
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'evaluation_results': results
    }, save_path)
    
    
    metrics_df = pd.DataFrame({
        'Metric': ['MSE', 'PCC', 'R2', 'MSE_DE', 'PCC_DE', 'R2_DE'],
        'Value': [results['MSE'], results['PCC'], results['R2'], 
                 results['MSE_DE'], results['PCC_DE'], results['R2_DE']]
    })
    
    print("\nEvaluation Results:")
    print(metrics_df.to_string(index=False, float_format=lambda x: '{:.6f}'.format(x)))
    print(f"\nModel and evaluation results saved to: {save_path}")
    
    return results

def objective(trial):
    global train_adata, train_dataset, test_dataset
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    n_hidden = trial.suggest_int('n_hidden', 256, 512, step=256)  
    n_layers = trial.suggest_int('n_layers', 2, 4)  
    n_heads = trial.suggest_int('n_heads', 4, 8, step=4)
    dropout = trial.suggest_float('dropout', 0.1, 0.3)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True)
    batch_size = trial.suggest_int('batch_size', 32, 64, step=32)  
    
    
    model = EnhancedScGPT(
        input_dim=train_adata.n_vars,
        hidden_dim=n_hidden,
        num_genes=train_adata.n_vars,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(20):
        train_loss = train_model(model, train_loader, optimizer, device)
        val_loss = evaluate_model(model, test_loader, device)
        
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
            
        
        trial.report(val_loss, epoch)
        
        
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return best_val_loss

def main():
    global train_adata, train_dataset, test_dataset
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    train_adata = sc.read_h5ad("$path/to/dataset.h5ad$")
    test_adata = sc.read_h5ad("$path/to/dataset.h5ad$")
    
    
    print(f"Training data shape: {train_adata.X.shape}")
    print(f"Test data shape: {test_adata.X.shape}")
    
    
    train_dataset = GeneExpressionDataset(train_adata)
    test_dataset = GeneExpressionDataset(test_adata, scaler=train_dataset.scaler)
    
    
    print(f"Training set perturbation dimension: {train_dataset.perturbations.shape}")
    
    
    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=5,
            interval_steps=1
        )
    )
    
    
    study.optimize(objective, n_trials=50, timeout=3600)  
    
    print('Best parameters:', study.best_params)
    print('Best validation loss:', study.best_value)
    
    
    best_params = study.best_params
    model = EnhancedScGPT(
        input_dim=train_adata.n_vars,
        hidden_dim=best_params['n_hidden'],
        num_genes=train_adata.n_vars,
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['learning_rate'])
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)
    
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(50):  
        train_loss = train_model(model, train_loader, optimizer, device)
        val_loss = evaluate_model(model, test_loader, device)
        
        print(f'Epoch {epoch+1}/50: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'method1_scgpt_best.pt')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    model = EnhancedScGPT(
        input_dim=train_adata.n_vars,
        hidden_dim=512,  
        num_genes=train_adata.n_vars,
    ).to(device)
    
    
    checkpoint = torch.load('method1_scgpt_best.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    
    results = evaluate_and_save_model(model, test_loader, device)
    
    
    # 使用相对路径，基于项目根目录
    project_root = Path(__file__).parent.parent.parent  # 项目根目录
    results_dir = project_root / "results"
    
    # 创建目录（如果不存在）
    results_dir.mkdir(parents=True, exist_ok=True)
    
    import json
    results_file = results_dir / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nEvaluation results saved to {results_file}")

if __name__ == "__main__":
    main()