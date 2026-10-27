"""CytoDiff: educational synthetic ADC/SCC H&E lung histopathology generation (LungHist700).

Pipeline stages map to subpackages:
    data          - standardization, patient/patch split, captioning  (stage 1)
    train         - LoRA fine-tuning of SD v1.5 / v2.1 / SDXL / SD 3   (stage 3)
    generate      - inference + CFG sweep                              (stage 4)
    eval          - FID / KID + qualitative grids                      (stage 4-5)
    colortransfer - histogram-matching color transfer on best model    (stage 6)
    utils         - shared config / logging helpers

Generated images are for EDUCATION, VISUALIZATION, and RESEARCH only — never clinical diagnosis.
"""

__version__ = "0.1.0"
