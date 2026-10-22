"""Distribution metrics for synthetic image sets, with patient-level bootstrap intervals.

All metrics compare features of a real and a generated set, so they are defined for sets, not
for single images. Every image is first brought to the 512 px evaluation resolution with an
antialiased (PIL bicubic) resize, so models that sample at 768 or 1024 px are scored on the same
footing as the 512 px real references.

Feature extractors (`BACKBONES`):
    inception  Inception-v3 pool features (2048-d), FID weights as used by torchmetrics and
               torch-fidelity (input resized internally to 299 px). Primary, pre-specified.
    uni        UNI ViT-L/16 (1024-d), pathology foundation model (MahmoodLab/UNI, gated).
    phikon     Phikon ViT-B/16 (768-d CLS token), pathology foundation model (owkin/phikon).

Metrics:
    KID (primary): unbiased squared MMD, cubic polynomial kernel k(x, y) = (x.y / d + 1)^3, on the
        FULL sets (no overlapping subsets). Bootstrap: real images resampled by patient (cluster
        bootstrap), generated images by image; pairs of the same underlying image are excluded
        from within-set kernel means. Duplicating a patient raises the share of within-patient
        pairs, so resampled KID values are shifted upward relative to the estimate; the interval
        for one KID is therefore a normal bootstrap interval, estimate +- 1.96 bootstrap SD
        (`normal_ci`). Paired differences between runs share the real resample, so that term
        cancels and their draws stay centered; their intervals are percentile intervals.
    FD: Frechet distance between Gaussian fits (FID for Inception features). Computed exactly
        through the singular values of the centered cross-product, which is cheap when, as here,
        each set has far fewer images than feature dimensions; this makes bootstraps affordable.
        With ~80 references the covariance is rank-deficient, so FD is secondary.
    Precision / recall (Kynkaanniemi et al., 2019) and density / coverage (Naeem et al., 2020):
        k-nearest-neighbor manifold estimates (k = 5) that separate fidelity from diversity.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

EVAL_RESOLUTION = 512
BACKBONES = ("inception", "uni", "phikon")
PRIMARY_BACKBONE = "inception"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_MODELS: dict[tuple[str, str], torch.nn.Module] = {}


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_eval_image(path: Path) -> Image.Image:
    """RGB image at the evaluation resolution (antialiased downscale if larger)."""
    img = Image.open(path).convert("RGB")
    if img.size != (EVAL_RESOLUTION, EVAL_RESOLUTION):
        img = img.resize((EVAL_RESOLUTION, EVAL_RESOLUTION), Image.Resampling.BICUBIC)
    return img


def _model(backbone: str, device: str) -> torch.nn.Module:
    key = (backbone, device)
    if key not in _MODELS:
        if backbone == "inception":
            from torchmetrics.image.fid import FrechetInceptionDistance
            net = FrechetInceptionDistance(feature=2048).inception
        elif backbone == "uni":
            import timm
            net = timm.create_model("hf-hub:MahmoodLab/UNI", pretrained=True, init_values=1e-5,
                                    dynamic_img_size=True)
        elif backbone == "phikon":
            from transformers import ViTModel
            net = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
        else:
            raise ValueError(f"unknown backbone {backbone!r}; choose from {BACKBONES}")
        _MODELS[key] = net.to(device).eval()
    return _MODELS[key]


def _batch_tensor(imgs: list[Image.Image], backbone: str) -> torch.Tensor:
    if backbone == "inception":
        # uint8 [0, 255]; the network resizes to 299 internally (torch-fidelity convention).
        return torch.from_numpy(np.stack([np.asarray(i, dtype=np.uint8) for i in imgs])).permute(0, 3, 1, 2)
    arr = np.stack([np.asarray(i.resize((224, 224), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
                    for i in imgs])
    arr = (arr - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
    return torch.from_numpy(arr).permute(0, 3, 1, 2)


@torch.no_grad()
def extract_features(paths: list[Path], backbone: str = PRIMARY_BACKBONE, device: str | None = None,
                     batch_size: int = 32) -> np.ndarray:
    """Features, shape (len(paths), d), float64."""
    device = device or default_device()
    net = _model(backbone, device)
    out = []
    for i in range(0, len(paths), batch_size):
        x = _batch_tensor([load_eval_image(p) for p in paths[i : i + batch_size]], backbone).to(device)
        if backbone == "phikon":
            f = net(pixel_values=x).last_hidden_state[:, 0]
        else:
            f = net(x)
        out.append(f.reshape(len(x), -1).cpu().double().numpy())  # MPS has no float64
    return np.concatenate(out)


# --- KID --------------------------------------------------------------------------------

def poly_kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a @ b.T / a.shape[1] + 1.0) ** 3


class KernelSet:
    """Precomputed kernel matrices for one (real, generated) pair; cheap to resample."""

    def __init__(self, real: np.ndarray, gen: np.ndarray):
        self.kxx = poly_kernel(real, real)
        self.kyy = poly_kernel(gen, gen)
        self.kxy = poly_kernel(real, gen)

    def kid(self, i: np.ndarray | None = None, j: np.ndarray | None = None) -> float:
        """Unbiased MMD^2 on real rows i and generated rows j (default: all rows)."""
        i = np.arange(len(self.kxx)) if i is None else i
        j = np.arange(len(self.kyy)) if j is None else j
        xx = self.kxx[np.ix_(i, i)][i[:, None] != i[None, :]].mean()
        yy = self.kyy[np.ix_(j, j)][j[:, None] != j[None, :]].mean()
        return float(xx + yy - 2.0 * self.kxy[np.ix_(i, j)].mean())


def cluster_resamples(groups: np.ndarray, n_boot: int, seed: int) -> list[np.ndarray]:
    """Row indices for n_boot cluster-bootstrap resamples (clusters drawn with replacement)."""
    rng = np.random.default_rng(seed)
    members = {g: np.flatnonzero(groups == g) for g in np.unique(groups)}
    keys = list(members)
    return [np.concatenate([members[keys[k]] for k in rng.integers(0, len(keys), len(keys))])
            for _ in range(n_boot)]


def image_resamples(n: int, n_boot: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n, n) for _ in range(n_boot)]


def bootstrap_kid(ks: KernelSet, real_idx: list[np.ndarray], gen_idx: list[np.ndarray]) -> np.ndarray:
    return np.array([ks.kid(i, j) for i, j in zip(real_idx, gen_idx)])


def normal_ci(point: float, draws: np.ndarray) -> tuple[float, float]:
    """Normal bootstrap interval centered on the full-sample estimate (estimate +- 1.96 SD)."""
    sd = float(np.std(draws, ddof=1))
    return point - 1.96 * sd, point + 1.96 * sd


# --- Frechet distance ---------------------------------------------------------------------

def frechet_distance(real: np.ndarray, gen: np.ndarray) -> float:
    """||mu_r - mu_g||^2 + Tr(S_r) + Tr(S_g) - 2 Tr((S_r S_g)^(1/2)), exactly.

    With centered data A (n_r x d) and B (n_g x d), S_r = A'A/(n_r-1) and S_g = B'B/(n_g-1); the
    nonzero eigenvalues of S_r S_g are the squared singular values of A B' / sqrt((n_r-1)(n_g-1)),
    so the trace term is a sum of singular values of an n_r x n_g matrix (no d x d square root).
    """
    a = real - real.mean(0)
    b = gen - gen.mean(0)
    nr, ng = len(real) - 1, len(gen) - 1
    tr_cross = np.linalg.svd(a @ b.T, compute_uv=False).sum() / np.sqrt(nr * ng)
    diff = real.mean(0) - gen.mean(0)
    return float(diff @ diff + (a * a).sum() / nr + (b * b).sum() / ng - 2.0 * tr_cross)


def bootstrap_fd(real: np.ndarray, gen: np.ndarray, real_idx: list[np.ndarray],
                 gen_idx: list[np.ndarray]) -> np.ndarray:
    return np.array([frechet_distance(real[i], gen[j]) for i, j in zip(real_idx, gen_idx)])


# --- precision / recall / density / coverage ------------------------------------------------

def _pairwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d2 = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * a @ b.T
    return np.sqrt(np.maximum(d2, 0.0))


def prdc(real: np.ndarray, gen: np.ndarray, k: int = 5) -> dict[str, float]:
    """Precision, recall (Kynkaanniemi et al. 2019), density, coverage (Naeem et al. 2020)."""
    d_rr, d_gg, d_rg = _pairwise(real, real), _pairwise(gen, gen), _pairwise(real, gen)
    r_real = np.sort(d_rr, axis=1)[:, k]          # k-th neighbor, index 0 is the point itself
    r_gen = np.sort(d_gg, axis=1)[:, k]
    inside_real = d_rg <= r_real[:, None]         # (real j, gen i): gen i inside real j's ball
    return {
        "precision": float(inside_real.any(axis=0).mean()),
        "recall": float((d_rg <= r_gen[None, :]).any(axis=1).mean()),
        "density": float(inside_real.sum(axis=0).mean() / k),
        "coverage": float((d_rg.min(axis=1) <= r_real).mean()),
    }
