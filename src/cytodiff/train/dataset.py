"""Training dataset over cached VAE latents + text embeddings (stage 1d cache).

Each item is a precomputed latent *distribution* (so we still sample fresh z0 per step) and the
cached caption embedding. The VAE and text encoder never touch the GPU during training, which is
what keeps the big models within budget.
"""

from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset

from cytodiff.data.cache import latent_cache_path, text_cache_path
from cytodiff.data.split import MANIFEST_PATH


class CachedLatentDataset(Dataset):
    def __init__(self, model_name: str):
        df = pd.read_csv(MANIFEST_PATH)
        self.rows = df[df["split"] == "train"].reset_index(drop=True)
        self.model_name = model_name
        # Sanity: cache must exist for every training row.
        missing = [
            r["path"] for _, r in self.rows.iterrows()
            if not latent_cache_path(model_name, _stem(r["path"])).exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} latents not cached for {model_name}; run cache.build_caches first "
                f"(e.g. {missing[0]})."
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows.iloc[idx]
        lat = torch.load(latent_cache_path(self.model_name, _stem(row["path"])), weights_only=True)
        emb = torch.load(text_cache_path(self.model_name, row["label"]), weights_only=True)
        # Single-encoder caches store a bare Tensor; SDXL/SD3 store {emb, pooled}.
        if isinstance(emb, dict):
            text_emb, pooled = emb["emb"], emb["pooled"]
        else:
            text_emb, pooled = emb, torch.zeros(0)  # placeholder; unused for unet_single
        return {
            "params": lat["params"],
            "scaling": float(lat["scaling"]),
            "shift": float(lat.get("shift", 0.0)),
            "text_emb": text_emb,
            "pooled": pooled,
        }

    def class_balanced_weights(self) -> torch.Tensor:
        # Per-row weights inversely proportional to class frequency. With a
        # WeightedRandomSampler both classes are drawn ~50/50 (224 ADC / 260 SCC train rows
        # on LungHist700), so batches are class-balanced on average. The eval-reference
        # asymmetry in split.py handles the eval side.
        counts = self.rows["class"].value_counts().to_dict()
        weights = self.rows["class"].map(lambda c: 1.0 / counts[c]).to_numpy(copy=True)
        return torch.as_tensor(weights, dtype=torch.double)


def _stem(path: str) -> str:
    return path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
