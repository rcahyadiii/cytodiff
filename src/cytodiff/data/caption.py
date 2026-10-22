"""Stage 1c — captioning (proposal §3.3.1).

A single fixed per-class label is used for all four models so the comparison isolates the
base architecture. This module is the one place prompt content lives, so the prompt-variant
ablation (§3.9 scenario 4) extends here without touching the rest of the pipeline.
"""

from __future__ import annotations

from cytodiff.constants import CLASS_LABELS


def caption_for(class_name: str) -> str:
    """Return the fixed text label for a class ('ADC' or 'SCC')."""
    try:
        return CLASS_LABELS[class_name]
    except KeyError as e:
        raise ValueError(f"Unknown class {class_name!r}; expected one of {list(CLASS_LABELS)}") from e


def caption_variants(class_name: str) -> list[str]:
    """Prompt variants for the ablation. Baseline returns the single fixed label.

    Extend with paraphrases here to run §3.9 scenario 4; keep the first entry as the
    canonical label so baseline runs stay reproducible.
    """
    return [caption_for(class_name)]
