"""Stage 4 — synthetic image generation + classifier-free-guidance sweep.

Generate per class across CFG_SWEEP; DDIM (50 steps) for U-Net models, native
scheduler for SD 3. Outputs to experiments/samples/<model>/<cfg>/<class>/.
"""
