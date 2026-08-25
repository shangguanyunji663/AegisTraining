# Archived Training Assets

This directory contains historical training datasets and the distillation corpus kept for local reproducibility and audit history.

- `risk_sft_v1/`, `risk_sft_v2/`, `risk_sft_v2_round2/`, and `risk_sft_v3/` are superseded dataset builds.
- `distill_psychology-10k-r1.json` is a generation-oriented corpus without risk labels; it is not a current risk-training input.
- The current recommended pipeline uses `training/data/consolidated_risk_v1/` and `training/scripts/prepare_risk_sft_v4.py`.

These files are local research assets and are ignored by the Git repository. Do not publish them without checking authorization, licensing, privacy, and de-identification requirements.
