"""Command-line entry point for NEST."""

from __future__ import annotations

import argparse

import scanpy as sc

from .nest import NEST, NESTConfig


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NEST spatial transcriptome reconstruction")
    parser.add_argument("--mode", required=True, choices=("missing_gene", "dropout"))
    parser.add_argument("--st", required=True, help="Input ST .h5ad file")
    parser.add_argument("--sc", help="Matched scRNA-seq .h5ad file (missing_gene only)")
    parser.add_argument("--output", required=True, help="Output .h5ad file")
    parser.add_argument("--target-genes", nargs="*", help="Genes to reconstruct in missing_gene mode")
    parser.add_argument("--spatial-key", default="spatial")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--n-neighbors", type=int)
    parser.add_argument("--latent-dim", type=int)
    parser.add_argument("--n-pcs", type=int)
    parser.add_argument("--ae-epochs", type=int)
    parser.add_argument("--gae-epochs", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--replace-all",
        action="store_true",
        help="In dropout mode, return the complete model output instead of replacing zeros only",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.mode == "missing_gene" and not args.sc:
        raise SystemExit("--sc is required when --mode missing_gene")
    if args.mode == "dropout" and args.sc:
        raise SystemExit("--sc is not used when --mode dropout")
    adata_st = sc.read_h5ad(args.st)
    adata_sc = sc.read_h5ad(args.sc) if args.sc else None
    config = NESTConfig(
        mode=args.mode,
        spatial_key=args.spatial_key,
        device=args.device,
        learning_rate=args.learning_rate,
        n_neighbors=args.n_neighbors,
        latent_dim=args.latent_dim,
        n_pcs=args.n_pcs,
        ae_epochs=args.ae_epochs,
        gae_epochs=args.gae_epochs,
        random_state=args.seed,
        fill_zeros_only=not args.replace_all,
    )
    result = NEST(config).fit_transform(
        adata_st, adata_sc, target_genes=args.target_genes
    )
    result.write_h5ad(args.output)


if __name__ == "__main__":
    main()

