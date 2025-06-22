import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scanpy as sc
import scipy
import anndata
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
import os
import pandas as pd
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import optuna
from IPython.display import display
import time

def print_log(message):
    """Custom print function to simulate notebook output style"""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

train_adata = None
train_dataset = None
test_dataset = None
device = None
pca_model = None

class GeneExpressionDataset(Dataset):
    def __init__(self, adata, perturbation_key='perturbation', scaler=None, pca_model=None, pca_dim=128, fit_pca=False, augment=False, is_train=True):
        self.adata = adata
        self.perturbation_key = perturbation_key
        self.augment = augment
        self.training = True
        self.pca_dim = pca_dim
        self.is_train = is_train  
        
        
        if scipy.sparse.issparse(adata.X):
            data = adata.X.toarray()
        else:
            data = adata.X
        data = np.maximum(data, 0)
        data = np.maximum(data, 1e-10)
        data = np.log1p(data)
        
        
        if scaler is None:
            self.scaler = StandardScaler()
            data = self.scaler.fit_transform(data)
        else:
            self.scaler = scaler
            data = self.scaler.transform(data)
            
        data = np.clip(data, -10, 10)
        data = data / 10.0
        
        
        if pca_model is None:
            if fit_pca:
                self.pca = PCA(n_components=pca_dim)
                self.expression_data = self.pca.fit_transform(data)
            else:
                raise ValueError('pca_model must be provided for test set')
        else:
            self.pca = pca_model
            self.expression_data = self.pca.transform(data)
            
        
        self.perturbations = pd.get_dummies(adata.obs[perturbation_key]).values
        print(f"{'Training' if is_train else 'Testing'} set perturbation dimensions: {self.perturbations.shape[1]}")
        
        
        if not all(x.shape[1] == self.perturbations.shape[1] for x in [self.perturbations]):
            raise ValueError("All samples must have consistent perturbation dimensions")
    def __len__(self):
        return len(self.adata)
    def __getitem__(self, idx):
        x, pert = self.expression_data[idx], self.perturbations[idx]
        
        if self.augment and self.training:
            
            noise = np.random.normal(0, 0.1, x.shape)
            x = x + noise
            
            
            mask = np.random.random(x.shape) > 0.1
            x = x * mask
            
            
            scale = np.random.uniform(0.9, 1.1)
            x = x * scale
        
        return torch.FloatTensor(x), torch.FloatTensor(pert)


class VAEEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
    def forward(self, x):
        h = F.gelu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

class VAEDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, z):
        h = F.gelu(self.fc1(z))
        return self.fc2(h)


class PerturbationEmbedding(nn.Module):
    def __init__(self, pert_dim, emb_dim):
        super().__init__()
        self.embedding = nn.Linear(pert_dim, emb_dim)
    def forward(self, pert):
        return self.embedding(pert)


class HybridAttentionModel(nn.Module):
    def __init__(self, input_dim, train_pert_dim, test_pert_dim, hidden_dim=512, n_layers=2, n_heads=8, dropout=0.1, attention_dropout=0.1, ffn_dropout=0.1, activation='gelu', use_transformer=True, use_vae=False, vae_latent_dim=64, vae_hidden_dim=256, use_pert_emb=False, pert_emb_dim=32, vae_beta=1.0):
        super(HybridAttentionModel, self).__init__()
        self.input_dim = input_dim
        self.train_pert_dim = train_pert_dim
        self.test_pert_dim = test_pert_dim
        self.hidden_dim = hidden_dim
        self.use_transformer = use_transformer
        self.use_vae = use_vae
        self.vae_beta = vae_beta
        self.use_pert_emb = use_pert_emb
        
        
        if use_vae:
            self.vae_encoder = VAEEncoder(input_dim, vae_latent_dim, vae_hidden_dim)
            self.vae_decoder = VAEDecoder(vae_latent_dim, input_dim, vae_hidden_dim)
            expr_out_dim = vae_latent_dim
        else:
            self.expression_encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            expr_out_dim = hidden_dim
            
        
        if use_pert_emb:
            self.train_pert_encoder = PerturbationEmbedding(train_pert_dim, pert_emb_dim)
            self.test_pert_encoder = PerturbationEmbedding(test_pert_dim, pert_emb_dim)
            pert_out_dim = pert_emb_dim
        else:
            self.train_pert_encoder = nn.Sequential(
                nn.Linear(train_pert_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            self.test_pert_encoder = nn.Sequential(
                nn.Linear(test_pert_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            pert_out_dim = hidden_dim
            
        
        fusion_dim = expr_out_dim + pert_out_dim
        
        self.fusion_dim = ((fusion_dim + n_heads - 1) // n_heads) * n_heads
        if self.fusion_dim != fusion_dim:
            self.fusion_proj = nn.Linear(fusion_dim, self.fusion_dim)
        else:
            self.fusion_proj = nn.Identity()
            
        if use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.fusion_dim,  
                nhead=n_heads,
                dim_feedforward=hidden_dim*4,
                dropout=ffn_dropout,
                activation=activation,
                batch_first=True,
                norm_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        else:
            self.self_attention = nn.MultiheadAttention(self.fusion_dim, n_heads, dropout=attention_dropout, batch_first=True)
            
        
        self.fusion = nn.Sequential(
            nn.Linear(self.fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
        
        
        self.train_perturbation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, train_pert_dim)
        )
        
        self.test_perturbation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, test_pert_dim)
        )
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    def forward(self, x, pert, is_train=True):
        vae_kl = 0
        vae_recon = None
        
        
        if self.use_vae:
            z, mu, logvar = self.vae_encoder(x)
            vae_recon = self.vae_decoder(z)
            vae_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
            expr_feat = z
        else:
            expr_feat = self.expression_encoder(x)
            
        
        if is_train:
            pert_feat = self.train_pert_encoder(pert)
        else:
            pert_feat = self.test_pert_encoder(pert)
        
        
        fusion_input = torch.cat([expr_feat, pert_feat], dim=1)
        
        fusion_input = self.fusion_proj(fusion_input)
        fusion_input = fusion_input.unsqueeze(1)  
        
        if self.use_transformer:
            x_trans = self.transformer(fusion_input).squeeze(1)
        else:
            x_trans, _ = self.self_attention(fusion_input, fusion_input, fusion_input)
            x_trans = x_trans.squeeze(1)
            
        fused = self.fusion(x_trans)
        output = self.output(fused)
        
        
        if is_train:
            pert_pred = self.train_perturbation_head(fused)
        else:
            pert_pred = self.test_perturbation_head(fused)
        
        return output, pert_pred, vae_recon, vae_kl

def train_model(model, train_loader, optimizer, scheduler, device, aux_weight=0.1):
    model.train()
    total_loss = 0
    accumulation_steps = 4
    optimizer.zero_grad()
    
    for i, batch in enumerate(train_loader):
        x, pert = batch
        x, pert = x.to(device), pert.to(device)
        
        output, pert_pred, _, _ = model(x, pert, is_train=True)
        
        main_loss = F.mse_loss(output, x)
        aux_loss = F.mse_loss(pert_pred, pert)
        loss = main_loss + aux_weight * aux_loss
        loss = loss / accumulation_steps
        
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
    
    return total_loss / len(train_loader)

def evaluate_model(model, test_loader, device, aux_weight=0.1):
    model.eval()
    total_loss = 0
    total_r2 = 0
    total_pearson = 0
    total_pert_r2 = 0
    with torch.no_grad():
        for batch in test_loader:
            x, pert = batch
            x, pert = x.to(device), pert.to(device)
            output, pert_pred, _, _ = model(x, pert, is_train=False)
            
            main_loss = F.mse_loss(output, x)
            aux_loss = F.mse_loss(pert_pred, pert)
            loss = main_loss + aux_weight * aux_loss
            total_loss += loss.item()
            r2 = r2_score(x.cpu().numpy(), output.cpu().numpy())
            total_r2 += r2
            pearson = np.mean([pearsonr(x[i].cpu().numpy(), output[i].cpu().numpy())[0] for i in range(x.size(0))])
            total_pearson += pearson
            pert_r2 = r2_score(pert.cpu().numpy(), pert_pred.cpu().numpy())
            total_pert_r2 += pert_r2
    return {
        'loss': total_loss / len(test_loader),
        'r2': total_r2 / len(test_loader),
        'pearson': total_pearson / len(test_loader),
        'pert_r2': total_pert_r2 / len(test_loader)
    }

def validate_data_consistency(train_adata, test_adata):
    train_pert = set(train_adata.obs['perturbation'].unique())
    test_pert = set(test_adata.obs['perturbation'].unique())
    
    print(f"Number of perturbation types in training set: {len(train_pert)}")
    print(f"Number of perturbation types in test set: {len(test_pert)}")

def objective(trial):
    global train_dataset, test_dataset, device, pca_model
    
    params = {
        'pca_dim': 128,
        'n_hidden': trial.suggest_int('n_hidden', 256, 1024),
        'n_layers': trial.suggest_int('n_layers', 1, 3),
        'n_heads': trial.suggest_int('n_heads', 4, 8),
        'dropout': trial.suggest_float('dropout', 0.1, 0.3),
        'attention_dropout': trial.suggest_float('attention_dropout', 0.1, 0.2),
        'ffn_dropout': trial.suggest_float('ffn_dropout', 0.1, 0.2),
        'aux_weight': trial.suggest_float('aux_weight', 0.05, 0.15),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-4, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        'use_transformer': trial.suggest_categorical('use_transformer', [True, False]),
        'use_vae': trial.suggest_categorical('use_vae', [True, False]),
        'vae_latent_dim': trial.suggest_int('vae_latent_dim', 32, 64),
        'vae_hidden_dim': trial.suggest_int('vae_hidden_dim', 128, 256),
        'use_pert_emb': trial.suggest_categorical('use_pert_emb', [True, False]),
        'pert_emb_dim': trial.suggest_int('pert_emb_dim', 16, 32),
        'vae_beta': trial.suggest_float('vae_beta', 0.1, 0.5)
    }
    
    model = HybridAttentionModel(
        input_dim=128,
        train_pert_dim=train_dataset.perturbations.shape[1],
        test_pert_dim=test_dataset.perturbations.shape[1],
        hidden_dim=params['n_hidden'],
        n_layers=params['n_layers'],
        n_heads=params['n_heads'],
        dropout=params['dropout'],
        attention_dropout=params['attention_dropout'],
        ffn_dropout=params['ffn_dropout'],
        use_transformer=params['use_transformer'],
        use_vae=params['use_vae'],
        vae_latent_dim=params['vae_latent_dim'],
        vae_hidden_dim=params['vae_hidden_dim'],
        use_pert_emb=params['use_pert_emb'],
        pert_emb_dim=params['pert_emb_dim'],
        vae_beta=params['vae_beta']
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params['learning_rate'],
        weight_decay=params['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=params['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    max_epochs = 100
    
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{max_epochs}'):
            x, pert = batch
            x, pert = x.to(device), pert.to(device)
            
            optimizer.zero_grad()
            output, pert_pred, vae_recon, vae_kl = model(x, pert, is_train=True)
            
            mse_loss = F.mse_loss(output, x)
            pert_loss = F.binary_cross_entropy_with_logits(pert_pred, pert)
            
            if params['use_vae']:
                vae_recon_loss = F.mse_loss(vae_recon, x)
                loss = mse_loss + params['aux_weight'] * pert_loss + vae_recon_loss + params['vae_beta'] * vae_kl
            else:
                loss = mse_loss + params['aux_weight'] * pert_loss
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                x, pert = batch
                x, pert = x.to(device), pert.to(device)
                
                output, pert_pred, vae_recon, vae_kl = model(x, pert, is_train=False)
                
                mse_loss = F.mse_loss(output, x)
                pert_loss = F.binary_cross_entropy_with_logits(pert_pred, pert)
                
                if params['use_vae']:
                    vae_recon_loss = F.mse_loss(vae_recon, x)
                    loss = mse_loss + params['aux_weight'] * pert_loss + vae_recon_loss + params['vae_beta'] * vae_kl
                else:
                    loss = mse_loss + params['aux_weight'] * pert_loss
                
                val_loss += loss.item()
        
        val_loss /= len(test_loader)
        
        scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'best_model_trial_{trial.number}.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        print(f'Epoch {epoch+1}/{max_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}')
    
    return best_val_loss

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

def evaluate_and_save_model(model, test_loader, device, save_path='method3_hybrid_best.pt'):
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            x, pert = batch
            x, pert = x.to(device), pert.to(device)
            output, _, _, _ = model(x, pert, False)
            
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

def main():
    global train_adata, train_dataset, test_dataset, device, pca_model
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    print('Loading data...')
    train_path = "autodl-tmp/NormanWeissman2019_filtered_train_processed_unseenpert.h5ad"
    test_path = "autodl-tmp/NormanWeissman2019_filtered_test_processed_unseenpert.h5ad"
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Data files not found: {train_path} or {test_path}")
    
    train_adata = sc.read_h5ad(train_path)
    test_adata = sc.read_h5ad(test_path)
    
    validate_data_consistency(train_adata, test_adata)
    
    pca_model = PCA(n_components=128)
    
    if scipy.sparse.issparse(train_adata.X):
        train_data = train_adata.X.toarray()
    else:
        train_data = train_adata.X
    train_data = np.maximum(train_data, 0)
    train_data = np.maximum(train_data, 1e-10)
    train_data = np.log1p(train_data)
    
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    train_data = np.clip(train_data, -10, 10)
    train_data = train_data / 10.0
    
    pca_model.fit(train_data)
    
    train_dataset = GeneExpressionDataset(
        train_adata,
        perturbation_key='perturbation',
        scaler=scaler,
        pca_model=pca_model,
        pca_dim=128,
        fit_pca=False,
        augment=True,
        is_train=True
    )
    
    test_dataset = GeneExpressionDataset(
        test_adata,
        perturbation_key='perturbation',
        scaler=scaler,
        pca_model=pca_model,
        pca_dim=128,
        fit_pca=False,
        augment=False,
        is_train=False
    )
    
    print(f'Training data shape: {train_adata.X.shape}')
    print(f'Test data shape: {test_adata.X.shape}')
    
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=5,
            interval_steps=1
        )
    )
    
    print('Starting hyperparameter optimization...')
    study.optimize(objective, n_trials=50)
    
    print('Best parameters:')
    for key, value in study.best_params.items():
        print(f'{key}: {value}')
    
    best_params = study.best_params
    final_model = HybridAttentionModel(
        input_dim=128,
        train_pert_dim=train_dataset.perturbations.shape[1],
        test_pert_dim=test_dataset.perturbations.shape[1],
        hidden_dim=best_params['n_hidden'],
        n_layers=best_params['n_layers'],
        n_heads=best_params['n_heads'],
        dropout=best_params['dropout'],
        attention_dropout=best_params['attention_dropout'],
        ffn_dropout=best_params['ffn_dropout'],
        use_transformer=best_params['use_transformer'],
        use_vae=best_params['use_vae'],
        vae_latent_dim=best_params['vae_latent_dim'],
        vae_hidden_dim=best_params['vae_hidden_dim'],
        use_pert_emb=best_params['use_pert_emb'],
        pert_emb_dim=best_params['pert_emb_dim'],
        vae_beta=best_params['vae_beta']
    ).to(device)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=best_params['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=best_params['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    optimizer = torch.optim.AdamW(
        final_model.parameters(),
        lr=best_params['learning_rate'],
        weight_decay=best_params['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )
    
    print('Training final model...')
    best_loss = float('inf')
    best_model = None
    max_epochs = 100
    
    for epoch in range(max_epochs):
        train_loss = train_model(final_model, train_loader, optimizer, scheduler, device, 
                               aux_weight=best_params['aux_weight'])
        eval_metrics = evaluate_model(final_model, test_loader, device, 
                                    aux_weight=best_params['aux_weight'])
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{max_epochs}:')
            print(f'Training Loss: {train_loss:.4f}')
            print(f'Test Loss: {eval_metrics["loss"]:.4f}')
            print(f'R2 Score: {eval_metrics["r2"]:.4f}')
            print(f'Pearson Correlation: {eval_metrics["pearson"]:.4f}')
            print(f'Perturbation R2: {eval_metrics["pert_r2"]:.4f}')
        
        if eval_metrics["loss"] < best_loss:
            best_loss = eval_metrics["loss"]
            best_model = final_model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': final_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
                'metrics': eval_metrics,
                'best_params': best_params
            }, 'hybrid_best_model.pt')
            print(f"Saved best model with loss: {best_loss:.4f}")
    
    final_model.load_state_dict(best_model)
    
    print('Evaluating final model...')
    results = evaluate_and_save_model(final_model, test_loader, device, 'hybrid_final_model.pt')
    
    results_df = pd.DataFrame({
        'Metric': ['MSE', 'PCC', 'R2', 'MSE_DE', 'PCC_DE', 'R2_DE'],
        'Value': [results['MSE'], results['PCC'], results['R2'], 
                 results['MSE_DE'], results['PCC_DE'], results['R2_DE']],
        'Best_Params': [str(best_params)] * 6
    })
    
    results_df.to_csv('hybrid_evaluation_results.csv', index=False)
    
    print("\nFinal Evaluation Results:")
    display(results_df)
    
    return results_df

if __name__ == '__main__':
    results_df = main() 