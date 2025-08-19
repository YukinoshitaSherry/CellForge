import torch
import torch.nn as nn
import torch.optim as optim
import scanpy as sc
import numpy as np
import pandas as pd
import scipy.sparse
import os
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
from torch.utils.data import Dataset, DataLoader
import shap
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json
from IPython.display import display

def setup_device():
    if torch.cuda.is_available():
        
        gpu_count = torch.cuda.device_count()
        if gpu_count > 0:
            
            device = torch.device('cuda:0')
            print(f"Use GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            print("No GPU，Use CPU")
    else:
        device = torch.device('cpu')
        print("CUDA unavailable，use CPU")
    return device

class PerturbDataset(Dataset):
    def __init__(self, rna_data, protein_data, guide_ids, cell_types, perturb_genes):
        self.rna_data = torch.FloatTensor(rna_data)
        self.protein_data = torch.FloatTensor(protein_data)
        self.guide_ids = guide_ids.values if isinstance(guide_ids, pd.Series) else guide_ids
        self.cell_types = cell_types.values if isinstance(cell_types, pd.Series) else cell_types
        self.perturb_genes = perturb_genes.values if isinstance(perturb_genes, pd.Series) else perturb_genes
        
    def __len__(self):
        return len(self.rna_data)
    
    def __getitem__(self, idx):
        return {
            'rna': self.rna_data[idx],
            'protein': self.protein_data[idx],
            'guide_id': self.guide_ids[idx],
            'cell_type': self.cell_types[idx],
            'perturb_gene': self.perturb_genes[idx]
        }

class PerturbXGBModel:
    def __init__(self, rna_dim, protein_dim, num_cell_types, num_perturb_genes):
        self.rna_dim = rna_dim
        self.protein_dim = protein_dim
        self.num_cell_types = num_cell_types
        self.num_perturb_genes = num_perturb_genes
        
        
        self.rna_scaler = StandardScaler()
        self.protein_scaler = StandardScaler()
        
        
        self.models = {}
        for cell_type in range(num_cell_types):
            for perturb_gene in range(num_perturb_genes):
                self.models[(cell_type, perturb_gene)] = xgb.XGBRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=6,
                    min_child_weight=1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='reg:squarederror',
                    tree_method='hist',
                    device='cuda'
                )
    
    def prepare_features(self, rna, protein, cell_type, perturb_gene):
        
        
        rna_scaled = self.rna_scaler.transform(rna)
        protein_scaled = self.protein_scaler.transform(protein)
        
        
        cell_type_onehot = np.zeros((len(cell_type), self.num_cell_types))
        cell_type_onehot[np.arange(len(cell_type)), cell_type] = 1
        
        perturb_gene_onehot = np.zeros((len(perturb_gene), self.num_perturb_genes))
        perturb_gene_onehot[np.arange(len(perturb_gene)), perturb_gene] = 1
        
        
        features = np.hstack([
            rna_scaled,
            protein_scaled,
            cell_type_onehot,
            perturb_gene_onehot
        ])
        
        return features
    
    def train(self, train_loader, val_loader):
        
        all_rna = []
        all_protein = []
        all_cell_types = []
        all_perturb_genes = []
        all_targets = []
        
        for batch in train_loader:
            all_rna.append(batch['rna'].numpy())
            all_protein.append(batch['protein'].numpy())
            all_cell_types.append(batch['cell_type'].numpy())
            all_perturb_genes.append(batch['perturb_gene'].numpy())
            all_targets.append(torch.cat([batch['rna'], batch['protein']], dim=1).numpy())
        
        all_rna = np.concatenate(all_rna)
        all_protein = np.concatenate(all_protein)
        all_cell_types = np.concatenate(all_cell_types)
        all_perturb_genes = np.concatenate(all_perturb_genes)
        all_targets = np.concatenate(all_targets)
        
        
        self.rna_scaler.fit(all_rna)
        self.protein_scaler.fit(all_protein)
        
        
        for cell_type in range(self.num_cell_types):
            for perturb_gene in range(self.num_perturb_genes):
                
                mask = (all_cell_types == cell_type) & (all_perturb_genes == perturb_gene)
                if np.sum(mask) > 0:
                    features = self.prepare_features(
                        all_rna[mask],
                        all_protein[mask],
                        all_cell_types[mask],
                        all_perturb_genes[mask]
                    )
                    targets = all_targets[mask]
                    
                    
                    self.models[(cell_type, perturb_gene)].fit(
                        features, targets,
                        eval_set=[(features, targets)],
                        verbose=False
                    )
    
    def predict(self, test_loader):
        
        all_preds = []
        all_targets = []
        
        for batch in test_loader:
            rna = batch['rna'].numpy()
            protein = batch['protein'].numpy()
            cell_type = batch['cell_type'].numpy()
            perturb_gene = batch['perturb_gene'].numpy()
            target = torch.cat([batch['rna'], batch['protein']], dim=1).numpy()
            
            
            batch_preds = []
            for i in range(len(rna)):
                features = self.prepare_features(
                    rna[i:i+1],
                    protein[i:i+1],
                    cell_type[i:i+1],
                    perturb_gene[i:i+1]
                )
                pred = self.models[(cell_type[i], perturb_gene[i])].predict(features)
                batch_preds.append(pred)
            
            batch_preds = np.concatenate(batch_preds)
            all_preds.append(batch_preds)
            all_targets.append(target)
        
        return np.concatenate(all_preds), np.concatenate(all_targets)
    
    def analyze_perturbation_effects(self, test_loader, output_dir='./results/cite_analysis'):
        preds, targets = self.predict(test_loader)

        gene_effects = {}
        for i in range(self.num_perturb_genes):
            mask = test_loader.dataset.perturb_genes == i
            if np.sum(mask) > 0:
                gene_effects[i] = {
                    'mean_effect': np.mean(preds[mask] - targets[mask], axis=0),
                    'std_effect': np.std(preds[mask] - targets[mask], axis=0)
                }
        
        
        effects_df = pd.DataFrame({
            'gene': list(gene_effects.keys()),
            'mean_effect': [e['mean_effect'] for e in gene_effects.values()],
            'std_effect': [e['std_effect'] for e in gene_effects.values()]
        })
        effects_df.to_csv(os.path.join(output_dir, 'perturbation_effects.csv'))
        
        
        cell_type_effects = {}
        for i in range(self.num_cell_types):
            mask = test_loader.dataset.cell_types == i
            if np.sum(mask) > 0:
                cell_type_effects[i] = {
                    'mean_effect': np.mean(preds[mask] - targets[mask], axis=0),
                    'std_effect': np.std(preds[mask] - targets[mask], axis=0)
                }
        
        
        cell_effects_df = pd.DataFrame({
            'cell_type': list(cell_type_effects.keys()),
            'mean_effect': [e['mean_effect'] for e in cell_type_effects.values()],
            'std_effect': [e['std_effect'] for e in cell_type_effects.values()]
        })
        cell_effects_df.to_csv(os.path.join(output_dir, 'cell_type_effects.csv'))
        
        
        G = nx.Graph()
        for i in range(self.num_perturb_genes):
            for j in range(self.num_perturb_genes):
                if i != j:
                    
                    mask_i = test_loader.dataset.perturb_genes == i
                    mask_j = test_loader.dataset.perturb_genes == j
                    if np.sum(mask_i) > 0 and np.sum(mask_j) > 0:
                        corr = np.corrcoef(
                            preds[mask_i].flatten(),
                            preds[mask_j].flatten()
                        )[0, 1]
                        if not np.isnan(corr) and abs(corr) > 0.3:
                            G.add_edge(i, j, weight=corr)
        
        
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue',
                node_size=500, font_size=8)
        plt.savefig(os.path.join(output_dir, 'perturbation_network.png'))
        plt.close()
        
        return {
            'gene_effects': gene_effects,
            'cell_type_effects': cell_type_effects,
            'perturbation_network': G
        }

def evaluate_model(model, test_loader, device, ctrl_data):
    model.eval()
    all_preds = []
    all_truths = []
    
    with torch.no_grad():
        for batch in test_loader:
            rna = batch['rna'].to(device)
            protein = batch['protein'].to(device)
            
            outputs = model(rna, protein)
            pred = outputs['perturbation_pred']
            target = torch.cat([rna, protein], dim=1)
            
            all_preds.append(pred.cpu().numpy())
            all_truths.append(target.cpu().numpy())
    
    pred = np.concatenate(all_preds, axis=0)
    truth = np.concatenate(all_truths, axis=0)
    rna_dim = pred.shape[1] - batch['protein'].shape[1]
    
    
    pred_rna = pred[:, :rna_dim]
    truth_rna = truth[:, :rna_dim]
    
    
    pred_protein = pred[:, rna_dim:]
    truth_protein = truth[:, rna_dim:]
    
    
    pred_de = pred_rna[:, :1000]
    truth_de = truth_rna[:, :1000]
    ctrl_data_de = ctrl_data[:1000]
    
    
    delta_pred = pred_rna - ctrl_data
    delta_truth = truth_rna - ctrl_data
    delta_pred_de = pred_de - ctrl_data_de
    delta_truth_de = truth_de - ctrl_data_de
    
    
    metrics = {
        
        "MSE": mean_squared_error(truth_rna.flatten(), pred_rna.flatten()),
        "PCC": pearsonr(pred_rna.flatten(), truth_rna.flatten())[0],
        "R2": r2_score(truth_rna.flatten(), pred_rna.flatten()),
        
        
        "MSE_DE": mean_squared_error(truth_de.flatten(), pred_de.flatten()),
        "PCC_DE": pearsonr(pred_de.flatten(), truth_de.flatten())[0],
        "R2_DE": r2_score(truth_de.flatten(), pred_de.flatten()),
        
        
        "MSE_DELTA": mean_squared_error(delta_truth.flatten(), delta_pred.flatten()),
        "PCC_DELTA": pearsonr(delta_pred.flatten(), delta_truth.flatten())[0],
        "R2_DELTA": r2_score(delta_truth.flatten(), delta_pred.flatten()),
        
        
        "MSE_DE_DELTA": mean_squared_error(delta_truth_de.flatten(), delta_pred_de.flatten()),
        "PCC_DE_DELTA": pearsonr(delta_pred_de.flatten(), delta_truth_de.flatten())[0],
        "R2_DE_DELTA": r2_score(delta_truth_de.flatten(), delta_pred_de.flatten()),
        
        
        "PROTEIN_MSE": mean_squared_error(truth_protein.flatten(), pred_protein.flatten()),
        "PROTEIN_PCC": pearsonr(pred_protein.flatten(), truth_protein.flatten())[0],
        "PROTEIN_R2": r2_score(truth_protein.flatten(), pred_protein.flatten()),
    }
    
    return metrics

def main():
    device = setup_device()
    
    try:
        rna_train = sc.read("$path/to/dataset.h5ad$")
        rna_test = sc.read("$path/to/dataset.h5ad$")
        protein_train = sc.read("$path/to/dataset.h5ad$")
        protein_test = sc.read("$path/to/dataset.h5ad$")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return
    
    
    ctrl_data = np.mean(rna_train.X.toarray() if scipy.sparse.issparse(rna_train.X) else rna_train.X, axis=0)
    
    
    def get_col(adata, col):
        if col in adata.obs:
            return adata.obs[col].values
        else:
            return np.zeros(len(adata), dtype=int)
    try:
        train_dataset = PerturbDataset(
            rna_train.X, protein_train.X,
            rna_train.obs['guide_id'].values,
            get_col(rna_train, 'cell_type'),
            get_col(rna_train, 'perturb_gene')
        )
        test_dataset = PerturbDataset(
            rna_test.X, protein_test.X,
            rna_test.obs['guide_id'].values,
            get_col(rna_test, 'cell_type'),
            get_col(rna_test, 'perturb_gene')
        )
    except Exception as e:
        print(f"Error creating dataset: {str(e)}")
        return
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    
    num_cell_types = int(np.max(get_col(rna_train, 'cell_type'))) + 1
    num_perturb_genes = int(np.max(get_col(rna_train, 'perturb_gene'))) + 1
    model = PerturbXGBModel(
        rna_dim=rna_train.shape[1],
        protein_dim=protein_train.shape[1],
        num_cell_types=num_cell_types,
        num_perturb_genes=num_perturb_genes
    )
    model.train(train_loader, test_loader)
    
    metrics = evaluate_model(model, test_loader, device, ctrl_data)
    
    
    # 使用相对路径，基于项目根目录
    project_root = Path(__file__).parent.parent.parent  # 项目根目录
    results_dir = project_root / "results" / "cite_analysis"
    models_dir = project_root / "results" / "models" / "cite"
    
    # 创建目录（如果不存在）
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    
    model.save_model(str(models_dir / "xgb_model.json"))
    
    
    model_params = {
        'rna_dim': rna_train.shape[1],
        'protein_dim': protein_train.shape[1],
        'metrics': metrics
    }
    with open(models_dir / "xgb_params.json", 'w') as f:
        json.dump(model_params, f)
    
    
    results_df = pd.DataFrame([metrics])
    results_df.to_csv(results_dir / "perturbation_analysis_xgb.csv", index=False)
    display(results_df)
    print("\nAnalysis complete!")
    print(f"Model saved to {models_dir / 'xgb_model.json'}")
    print(f"Parameters saved to {models_dir / 'xgb_params.json'}")
    print(f"Results saved to {results_dir / 'perturbation_analysis_xgb.csv'}")

if __name__ == "__main__":
    main() 