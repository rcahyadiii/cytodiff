"""Checks of the distribution metrics (cytodiff.eval.metrics) against independent implementations.

KID and FD are compared with the formulas written out directly, with scipy's matrix square root,
and, when torchmetrics is installed, with torchmetrics' KID and FID on the same feature vectors.

Run:  PYTHONPATH=src python -m pytest tests -q
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import linalg

from cytodiff.eval.metrics import KernelSet, frechet_distance

RNG = np.random.default_rng(0)
D = 64
REAL = RNG.normal(size=(83, D))                       # held-out set size of the study (ADC)
GEN = RNG.normal(loc=0.3, scale=1.2, size=(200, D))   # 200 generated images per class


def kid_direct(x: np.ndarray, y: np.ndarray) -> float:
    """Unbiased squared MMD with the cubic polynomial kernel k(a, b) = (a.b / d + 1)^3."""
    d = x.shape[1]
    kxx, kyy, kxy = (x @ x.T / d + 1) ** 3, (y @ y.T / d + 1) ** 3, (x @ y.T / d + 1) ** 3
    m, n = len(x), len(y)
    return float((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (n * (n - 1))
                 - 2 * kxy.mean())


def fd_scipy(x: np.ndarray, y: np.ndarray) -> float:
    """Frechet distance with scipy's matrix square root (the textbook FID formula)."""
    s1, s2 = np.cov(x, rowvar=False), np.cov(y, rowvar=False)
    covmean = linalg.sqrtm(s1 @ s2).real
    return float(((x.mean(0) - y.mean(0)) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


def test_kid_matches_direct_formula():
    assert KernelSet(REAL, GEN).kid() == pytest.approx(kid_direct(REAL, GEN), rel=1e-9)


def test_kid_is_near_zero_for_one_distribution():
    a, b = RNG.normal(size=(400, D)), RNG.normal(size=(400, D))
    assert abs(KernelSet(a, b).kid()) < 0.05 * kid_direct(REAL, GEN)


def test_fd_matches_scipy():
    assert frechet_distance(REAL, GEN) == pytest.approx(fd_scipy(REAL, GEN), rel=1e-6)


def test_fd_rank_deficient_case_matches_scipy():
    """Fewer images than dimensions, as with ~80 held-out images and 2048-d Inception features."""
    x, y = RNG.normal(size=(40, 128)), RNG.normal(loc=0.2, size=(60, 128))
    assert frechet_distance(x, y) == pytest.approx(fd_scipy(x, y), rel=1e-5)


def _torch_features(torch):
    class Features(torch.nn.Module):
        """Passes feature vectors through; answers torchmetrics' 299 px probe image with D zeros."""

        def forward(self, x):
            return x if x.ndim == 2 else torch.zeros(x.shape[0], D, dtype=torch.float64)

    return Features()


def test_kid_matches_torchmetrics():
    torch = pytest.importorskip("torch")
    kid_mod = pytest.importorskip("torchmetrics.image.kid")
    x, y = REAL[:80], GEN[:80]   # torchmetrics draws equal-size subsets; one subset of all images
    metric = kid_mod.KernelInceptionDistance(feature=_torch_features(torch), subsets=1, subset_size=80)
    metric.update(torch.tensor(x), real=True)
    metric.update(torch.tensor(y), real=False)
    mean, _ = metric.compute()
    assert KernelSet(x, y).kid() == pytest.approx(float(mean), rel=1e-5)


def test_fd_matches_torchmetrics():
    torch = pytest.importorskip("torch")
    fid_mod = pytest.importorskip("torchmetrics.image.fid")
    metric = fid_mod.FrechetInceptionDistance(feature=_torch_features(torch))
    metric.update(torch.tensor(REAL), real=True)
    metric.update(torch.tensor(GEN), real=False)
    assert frechet_distance(REAL, GEN) == pytest.approx(float(metric.compute()), rel=1e-5)
