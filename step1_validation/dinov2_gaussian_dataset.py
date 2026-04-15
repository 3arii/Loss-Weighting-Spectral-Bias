"""Synthetic Gaussian dataset with DINOv2 mean and covariance.

Loads pre-computed mean/covariance from DINOv2-B patch tokens (norm layer,
ImageNet-1K) and draws synthetic Gaussian samples:

    X ~ N(mu, Sigma)  where Sigma = U diag(eigval) U^T

The eigenbasis (eigval, eigvec) is returned directly — no need to re-run PCA
on training data since we already have the exact covariance.

Default: norm layer, patch tokens, d=768.
Results stored at $STORE_DIR/DL_Projects/DINOv2_ImageNet1k_Covariance/
"""

import torch
import numpy as np

_DEFAULT_COV_DIR = (
    "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang"
    "/DL_Projects/DINOv2_ImageNet1k_Covariance"
)


def load_dinov2_stats(layer: str = "norm",
                      token: str = "patch",
                      cov_dir: str = _DEFAULT_COV_DIR) -> dict:
    """Load mean and covariance from a saved DINOv2 .pt file.

    Args:
        layer:   one of block1/block3/block5/block7/block9/block11/norm
        token:   one of cls/reg/patch/all
        cov_dir: path to DINOv2_ImageNet1k_Covariance directory
    Returns:
        dict with keys: mean [768], cov [768,768], n (sample count)
    """
    path = f"{cov_dir}/{layer}.pt"
    data = torch.load(path, weights_only=True)
    entry = data[token]
    return {
        "mean": entry["mean"].float(),   # [768]
        "cov":  entry["cov"].float(),    # [768, 768]
        "n":    int(entry["n"]),
    }


def compute_eigenbasis(cov: torch.Tensor) -> tuple:
    """Eigendecompose covariance matrix, sorted descending.

    Returns:
        eigval [d]   eigenvalues (descending)
        eigvec [d,d] eigenvectors (columns), same ordering
    """
    eigval, eigvec = torch.linalg.eigh(cov)   # ascending
    idx    = eigval.argsort(descending=True)
    eigval = eigval[idx]
    eigvec = eigvec[:, idx]
    # Clamp tiny negatives from numerical noise
    eigval = eigval.clamp(min=0.0)
    return eigval, eigvec


def sample_gaussian(mean: torch.Tensor,
                    eigval: torch.Tensor,
                    eigvec: torch.Tensor,
                    n_samples: int,
                    seed: int = 42) -> torch.Tensor:
    """Draw n_samples from N(mean, U diag(eigval) U^T).

    X = mean + U @ diag(sqrt(eigval)) @ z,  z ~ N(0, I)

    Returns: [n_samples, d]
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    d   = mean.shape[0]
    z   = torch.randn(n_samples, d, generator=rng)         # [N, d]
    std = eigval.sqrt()                                    # [d]
    X   = z * std.unsqueeze(0)                             # [N, d] in eigenbasis
    X   = X @ eigvec.T                                     # rotate to data space
    X   = X + mean.unsqueeze(0)                            # add mean
    return X


def make_dinov2_gaussian_dataset(n_samples: int = 10000,
                                  layer: str = "norm",
                                  token: str = "patch",
                                  seed: int = 42,
                                  cov_dir: str = _DEFAULT_COV_DIR,
                                  device: str = "cpu") -> dict:
    """Full pipeline: load stats -> eigenbasis -> draw samples.

    Args:
        n_samples: number of synthetic Gaussian draws (default 10k)
        layer:     DINOv2 layer (default "norm")
        token:     token type (default "patch")
        seed:      random seed for reproducibility
        cov_dir:   path to covariance directory
        device:    "cpu" or "cuda"
    Returns:
        dict with:
            X_train  [n_samples, 768]  synthetic training data
            mean     [768]             DINOv2 mean
            eigval   [768]             eigenvalues (descending)
            eigvec   [768, 768]        eigenvectors (columns)
            n_real   int               original sample count from DINOv2
    """
    stats  = load_dinov2_stats(layer=layer, token=token, cov_dir=cov_dir)
    mean   = stats["mean"]
    cov    = stats["cov"]
    eigval, eigvec = compute_eigenbasis(cov)

    X_train = sample_gaussian(mean, eigval, eigvec, n_samples, seed=seed)

    return {
        "X_train": X_train.to(device),
        "mean":    mean.to(device),
        "eigval":  eigval.to(device),
        "eigvec":  eigvec.to(device),
        "n_real":  stats["n"],
        "layer":   layer,
        "token":   token,
    }


if __name__ == "__main__":
    ds = make_dinov2_gaussian_dataset(n_samples=10000)
    print(f"X_train: {ds['X_train'].shape}")
    print(f"mean range: [{ds['mean'].min():.3f}, {ds['mean'].max():.3f}]")
    print(f"eigval[:5]: {ds['eigval'][:5].tolist()}")
    print(f"eigval[-5:]: {ds['eigval'][-5:].tolist()}")
    print(f"X_train mean: {ds['X_train'].mean():.4f}, std: {ds['X_train'].std():.4f}")
