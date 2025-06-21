import scanpy as sc
import anndata as ad

adata = sc.read_h5ad("/path/to/your/dataset.h5ad")
import anndata as ad

print(adata.obs.head())
print("---")
# see the columns of the obs
print(adata.obs.columns)
print("---")
# see the first few rows of the var
print(adata.var.head())
print("---")
# see the shape of the adata
print(adata.shape)