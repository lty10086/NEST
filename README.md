# NEST

NEST (**N**ative **E**xpression reconstruction for **S**patial **T**ranscriptomics)
provides one interface for two reconstruction tasks:

- `mode="missing_gene"`: completes unmeasured genes with a matched scRNA-seq
  reference, cross-modal mapping, gene-domain correction, consensus-graph
  construction, and graph-guided reconstruction.
- `mode="dropout"`: recovers false zeros from the ST matrix alone. Training
  losses are computed on observed non-zero entries and, by default, only input
  zeros are replaced.

## Installation

```bash
pip install nest-st
```

For a CUDA environment, install the PyTorch build appropriate for the local CUDA
version first, then install `nest-st`. NEST does not normalize or log-transform
data implicitly; `adata_st` and `adata_sc` must use the same non-negative scale.

## Python API

### Complete unmeasured genes

```python
import scanpy as sc
from nest_st import NEST, NESTConfig

adata_st = sc.read_h5ad("st_log1p.h5ad")
adata_sc = sc.read_h5ad("scrna_log1p.h5ad")

model = NEST(NESTConfig(mode="missing_gene", device="cuda"))
result = model.fit_transform(adata_st, adata_sc)
result.write_h5ad("nest_missing_gene.h5ad")
```

By default, all reference genes absent from ST are reconstructed. To perform a
gene-wise validation split, pass measured genes as `target_genes`; NEST excludes
them from training and replaces them with predictions.

```python
result = model.fit_transform(
    adata_st,
    adata_sc,
    target_genes=["GeneA", "GeneB"],
)
```

### Recover dropout zeros

```python
from nest_st import reconstruct

result = reconstruct(
    adata_st,
    mode="dropout",
    device="cuda",
)
result.write_h5ad("nest_dropout.h5ad")
```

Observed non-zero values are retained. Pass `fill_mask=` to replace a specific
boolean mask, or set `fill_zeros_only=False` to return the complete model output.

## Command line

```bash
nest-st --mode missing_gene --st st.h5ad --sc scrna.h5ad --output result.h5ad
nest-st --mode dropout --st st.h5ad --output result.h5ad
```

Every output stores resolved parameters in `result.uns["nest"]`, the learned
embedding in `result.obsm["X_nest_latent"]`, and preserves the input ST
observation and spatial metadata.

## Input requirements

- Inputs are `AnnData` objects with unique observation and gene names.
- `adata_st.obsm["spatial"]` contains finite 2-D coordinates.
- Expression values are finite and non-negative.
- In `missing_gene` mode, ST and scRNA-seq matrices use the same transformation
  and share at least two non-target genes.

## Data Availability
The processed datasets supporting this study are
available on Zenodo at [https://doi.org/10.5281/zenodo.XXXXXXXX](https://zenodo.org/records/23438059?preview=1&token=eyJhbGciOiJIUzUxMiIsImlhdCI6MTc5MjMyMTYzMCwiZXhwIjoxODExNjM1MTk5fQ.eyJpZCI6IjUwNmJmOTQ2LWMyZjAtNGM5MC1hOTdiLTk4YTcwMWJiZmY2NyIsImRhdGEiOnt9LCJyYW5kb20iOiI4NTJjMTljNmVkMWRiZWFlNzMxNmE2NDlhM2U3Y2VhYyJ9.-7ldL-JuTKA23Y8AFmUa-CkUivU7HXwsr0non5m9SbxNOPwbxXKrJcII5WzlyD9CS7RUqqBpPtb99aG8vyz9Qw).

## Development build

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
```

