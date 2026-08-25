# Legacy Tools

The scripts in this directory are historical v1-v3 inspection and data-build tools. They are retained for audit and old experiment reproduction, but they are not the current training entry points.

Use the versioned workflow under `training/scripts/` instead:

- `prepare_risk_sft.py` for the older candidate-pool pipeline
- `prepare_risk_sft_v4.py` for the current consolidated risk pipeline
- `train_risk_qlora.py` for isolated CUDA QLoRA training
- `merge_risk_qlora.py` for adapter export
- `eval_risk_qlora.py` and `serve_risk_qlora.py` for evaluation and inference

Legacy tools may reference archived data under `data/archive/` or external data under `external-data/`. Treat their outputs as historical artifacts and do not use them as the default production path.
