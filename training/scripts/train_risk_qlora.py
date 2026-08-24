"""Train an isolated NF4 QLoRA risk adapter."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

TRAINING_SRC = Path(__file__).resolve().parents[1] / "src"
if str(TRAINING_SRC) not in sys.path:
    sys.path.insert(0, str(TRAINING_SRC))

from aegis_training.paths import training_root, under

TRAIN_ROOT = training_root()


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("install PyYAML in the isolated training environment") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must be a YAML object")
    return payload


def _cuda_dtype(name: str, torch):
    try:
        return {"bfloat16": torch.bfloat16, "float16": torch.float16}[name]
    except KeyError as exc:
        raise ValueError(f"unsupported bnb_4bit_compute_dtype={name!r}") from exc


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: invalid JSONL") from exc
        if not isinstance(row, dict) or not isinstance(row.get("messages"), list):
            raise ValueError(f"{path}:{number}: missing messages")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no SFT rows")
    return rows


def _target_modules(model, requested: list[str]) -> list[str]:
    import torch.nn as nn

    try:
        import bitsandbytes as bnb

        bnb_types = tuple(
            candidate
            for candidate in (getattr(bnb.nn, "Linear4bit", None), getattr(bnb.nn, "Linear8bitLt", None))
            if isinstance(candidate, type)
        )
    except ImportError:
        bnb_types = ()

    forbidden = ("vision", "visual", "image", "video")
    suffixes = {
        name.rsplit(".", 1)[-1]
        for name, module in model.named_modules()
        if isinstance(module, (nn.Linear, *bnb_types))
        and not any(token in name.lower() for token in forbidden)
    }
    matched = [name for name in requested if name in suffixes]
    if not matched:
        raise RuntimeError(
            "no configured LoRA target suffix exists in the quantized text model; "
            f"available suffixes include {sorted(suffixes)[:40]}"
        )
    return matched


def _render_chat(tokenizer, messages: list[dict]) -> str:
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except AttributeError as exc:
        raise RuntimeError("the tokenizer must provide apply_chat_template") from exc


def _tokenize_row(row: dict, tokenizer, cutoff_len: int) -> dict:
    messages = row["messages"]
    prompt = _render_chat(tokenizer, messages[:-1])
    full = _render_chat(tokenizer, messages)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    encoded = tokenizer(full, add_special_tokens=False, truncation=True, max_length=cutoff_len)
    labels = list(encoded["input_ids"])
    labels[: min(len(prompt_ids), len(labels))] = [-100] * min(len(prompt_ids), len(labels))
    if all(value == -100 for value in labels):
        raise ValueError("assistant target was truncated; increase cutoff_len")
    encoded["labels"] = labels
    return encoded


def _assert_text_causal_model(model) -> None:
    forbidden = {"vision", "visual", "image", "video"}
    if any(token in name.lower() for name, _ in model.named_modules() for token in forbidden):
        raise RuntimeError("the selected checkpoint exposes vision modules; use the text-only base model")


def _build_training_arguments(TrainingArguments, train: dict, output_root: Path, compute_dtype, train_rows: int):
    arguments = {
        "output_dir": str(output_root),
        "per_device_train_batch_size": train["per_device_train_batch_size"],
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": train["gradient_accumulation_steps"],
        "learning_rate": train["learning_rate"],
        "weight_decay": train["weight_decay"],
        "num_train_epochs": train["num_train_epochs"],
        "warmup_steps": 0,
        "lr_scheduler_type": train["lr_scheduler_type"],
        "optim": train["optim"],
        "max_grad_norm": train["max_grad_norm"],
        "logging_steps": train["logging_steps"],
        "eval_strategy": train["eval_strategy"],
        "save_strategy": train["save_strategy"],
        "save_total_limit": train["save_total_limit"],
        "seed": train["seed"],
        "bf16": str(compute_dtype).endswith("bfloat16"),
        "fp16": str(compute_dtype).endswith("float16") and not str(compute_dtype).endswith("bfloat16"),
        "gradient_checkpointing": train["gradient_checkpointing"],
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": "none",
        "remove_unused_columns": False,
    }
    import inspect

    supported = set(inspect.signature(TrainingArguments).parameters)
    if "warmup_ratio" in supported:
        arguments["warmup_ratio"] = train["warmup_ratio"]
    else:
        effective_batch = train["per_device_train_batch_size"] * train["gradient_accumulation_steps"]
        total_steps = math.ceil(train["num_train_epochs"] * train_rows / effective_batch)
        arguments["warmup_steps"] = max(1, round(train["warmup_ratio"] * total_steps))
    try:
        return TrainingArguments(**arguments)
    except TypeError as exc:
        raise RuntimeError("installed Transformers does not support the pinned training arguments") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Train an isolated NF4 QLoRA risk adapter")
    parser.add_argument("--config", type=Path, default=TRAIN_ROOT / "training" / "configs" / "risk_qlora_4060.yaml")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForSeq2Seq,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )
        from aegis_training.base_model_gate import OFFICIAL_REVISION, verify_snapshot
    except ImportError as exc:
        print(f"TRAINING ENVIRONMENT FAILED: {exc}", file=sys.stderr)
        return 2

    if not torch.cuda.is_available():
        print("TRAINING ENVIRONMENT FAILED: CUDA PyTorch is required", file=sys.stderr)
        return 2

    try:
        config_path = under(args.config, TRAIN_ROOT, "config")
        data_root = under(args.data_root, TRAIN_ROOT, "data")
        snapshot_dir = under(args.snapshot_dir, TRAIN_ROOT, "snapshot")
        config = _load_yaml(config_path)
        base, quant, lora, train = (
            config["base_model"], config["quantization"], config["lora"], config["training"]
        )
        configured_output = args.output_root or Path(train["output_root"])
        output_root = under(
            configured_output if Path(configured_output).is_absolute() else TRAIN_ROOT / configured_output,
            TRAIN_ROOT,
            "output",
        )
        if base["revision"] != OFFICIAL_REVISION:
            raise ValueError("config revision differs from the pinned official Qwen3.5 Base revision")
        gate = verify_snapshot(snapshot_dir, provenance_path=None, require_exact_ollama_provenance=False)
        compute_dtype = _cuda_dtype(quant["bnb_4bit_compute_dtype"], torch)
        if compute_dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
            if not train["fp16_fallback"]:
                raise RuntimeError("bf16 is unavailable and fp16 fallback is disabled")
            compute_dtype = torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quant["load_in_4bit"],
            bnb_4bit_quant_type=quant["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=quant["bnb_4bit_use_double_quant"],
            bnb_4bit_compute_dtype=compute_dtype,
        )
        tokenizer = AutoTokenizer.from_pretrained(snapshot_dir, trust_remote_code=base["trust_remote_code"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            snapshot_dir,
            quantization_config=bnb_config,
            device_map={"": 0},
            trust_remote_code=base["trust_remote_code"],
            low_cpu_mem_usage=True,
        )
        _assert_text_causal_model(model)
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=train["gradient_checkpointing"])
        targets = _target_modules(model, lora["target_modules"])
        model = get_peft_model(
            model,
            LoraConfig(
                task_type="CAUSAL_LM",
                r=lora["rank"],
                lora_alpha=lora["alpha"],
                lora_dropout=lora["dropout"],
                bias=lora["bias"],
                target_modules=targets,
            ),
        )
        train_rows = _load_jsonl(data_root / "train.jsonl")
        dev_rows = _load_jsonl(data_root / "dev.jsonl")
        train_dataset = Dataset.from_list([_tokenize_row(row, tokenizer, train["cutoff_len"]) for row in train_rows])
        dev_dataset = Dataset.from_list([_tokenize_row(row, tokenizer, train["cutoff_len"]) for row in dev_rows])
        trainable, total = model.get_nb_trainable_parameters()
        report = {
            "gate": gate.__dict__,
            "cuda_device": torch.cuda.get_device_name(0),
            "compute_dtype": str(compute_dtype),
            "target_modules": targets,
            "trainable_params": trainable,
            "total_params": total,
            "trainable_percent": round(trainable / total * 100, 6),
            "train_rows": len(train_dataset),
            "dev_rows": len(dev_dataset),
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if args.dry_run:
            return 0

        output_root.mkdir(parents=True, exist_ok=True)
        trainer = Trainer(
            model=model,
            args=_build_training_arguments(TrainingArguments, train, output_root, compute_dtype, len(train_dataset)),
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, label_pad_token_id=-100),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=train["early_stopping_patience"])],
        )
        trainer.train()
        model.save_pretrained(output_root / "adapter")
        tokenizer.save_pretrained(output_root / "adapter")
        report["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated(0)
        (output_root / "training-manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"QLORA TRAINING FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
