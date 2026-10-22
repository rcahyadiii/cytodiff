"""Stage 6 — histogram-matching color transfer (best model only).

Per-channel CDF matching of each generated image to a class-matched real reference
(RGB and L*a*b* variants; keep lower-FID). Reinhard noted as ablation alternative.
"""

from cytodiff.colortransfer.histogram import apply_to_model, match_one

__all__ = ["apply_to_model", "match_one"]
