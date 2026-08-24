"""Merge a validated PEFT adapter into the pinned Qwen base snapshot."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TRAINING_SRC = Path(__file__).resolve().parents[1] / "src"
if str(TRAINING_SRC) not in sys.path:
    sys.path.insert(0, str(TRAINING_SRC))

from aegis_training.base_model_gate import verify_snapshot
from aegis_training.paths import training_root, under

TRAIN_ROOT = training_root()


def _assert_empty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"output directory already contains files: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge a risk QLoRA adapter")
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-shard-size", default="2GB")
    parser.add_argument("--preserve-multimodal-architecture", action="store_true")
    args = parser.parse_args()

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

        snapshot = under(args.snapshot_dir, TRAIN_ROOT, "snapshot")
        adapter = under(args.adapter_dir, TRAIN_ROOT, "adapter")
        output = under(args.output_dir, TRAIN_ROOT, "output")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required to merge the 2B base safely")
        if not (adapter / "adapter_config.json").is_file() or not (adapter / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(f"adapter files are missing from {adapter}")
        _assert_empty(output)
        gate = verify_snapshot(snapshot, provenance_path=None, require_exact_ollama_provenance=False)
        tokenizer = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        loader = AutoModelForImageTextToText if args.preserve_multimodal_architecture else AutoModelForCausalLM
        base_model = loader.from_pretrained(
            snapshot,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            trust_remote_code=False,
            low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(base_model, adapter)
        merged = model.merge_and_unload(safe_merge=True)
        if args.preserve_multimodal_architecture:
            merged.config.architectures = ["Qwen3_5ForConditionalGeneration"]
        else:
            merged.config.architectures = [type(merged).__name__]
        merged.generation_config.do_sample = False
        merged.generation_config.temperature = None
        output.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(output, safe_serialization=True, max_shard_size=args.max_shard_size)
        tokenizer.save_pretrained(output)
        provenance = {
            "kind": "merged-risk-qlora-export",
            "official_base": gate.__dict__,
            "adapter_dir": str(adapter),
            "output_dir": str(output),
            "dtype": "bfloat16",
            "merged_model_class": type(merged).__name__,
            "max_shard_size": args.max_shard_size,
        }
        (output / "aegis-export-manifest.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(provenance, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f"MERGE EXPORT FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        for name in ("merged", "model", "base_model"):
            if name in locals():
                del locals()[name]
        if "torch" in locals() and torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
