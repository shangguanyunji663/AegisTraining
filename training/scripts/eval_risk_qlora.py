"""Evaluate an Ollama or merged Transformers risk model on a frozen corpus."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

TRAINING_SRC = Path(__file__).resolve().parents[1] / "src"
if str(TRAINING_SRC) not in sys.path:
    sys.path.insert(0, str(TRAINING_SRC))

from aegis_training.data_contract import RISK_SYSTEM_PROMPT
from aegis_training.metrics import Prediction, risk_eval_report, write_report
from aegis_training.paths import project_root, training_root, under

TRAIN_ROOT = training_root()
LEVELS = {"low", "medium", "high"}
MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}$")


def _load_holdout(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("holdout corpus must be a JSON array")
    holdout = [row for row in rows if isinstance(row, dict) and row.get("layer") == "stress"]
    if not holdout:
        raise ValueError("holdout corpus has no stress rows")
    return holdout


def _parse_risk(raw: str) -> tuple[str | None, str, bool]:
    if raw.startswith("__ERROR__"):
        return None, "", False
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None, "", False
    if not isinstance(data, dict):
        return None, "", False
    level = str(data.get("risk_level", "")).strip().lower()
    return (level if level in LEVELS else None), str(data.get("reason", "")).strip(), True


def _run_ollama(model: str, message: str, timeout: float) -> tuple[str, float]:
    executable = shutil.which("ollama")
    if not executable:
        return "__ERROR__:OllamaNotFound", 0.0
    started = time.perf_counter()
    prompt = f"{RISK_SYSTEM_PROMPT}\n\n用户输入：\n{message}"
    try:
        completed = subprocess.run(
            [executable, "run", model, prompt, "--format", "json", "--hidethinking", "--think=false"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return "__ERROR__:Timeout", (time.perf_counter() - started) * 1000
    elapsed = (time.perf_counter() - started) * 1000
    if completed.returncode:
        return f"__ERROR__:OllamaExit:{completed.stderr.strip()[:240]}", elapsed
    return completed.stdout.strip(), elapsed


def _build_transformers_runner(model_dir: Path, max_new_tokens: int):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Transformers evaluation requires the isolated QLoRA environment") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("Transformers evaluation requires CUDA")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map={"": 0},
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    model.eval()
    device = next(model.parameters()).device

    def run(message: str) -> tuple[str, float]:
        started = time.perf_counter()
        try:
            encoded = tokenizer.apply_chat_template(
                [{"role": "system", "content": RISK_SYSTEM_PROMPT}, {"role": "user", "content": message}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                generated = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None)
            prompt_len = encoded["input_ids"].shape[1]
            raw = tokenizer.decode(generated[0, prompt_len:], skip_special_tokens=True).strip()
        except Exception as exc:
            raw = f"__ERROR__:Transformers:{type(exc).__name__}:{exc}"
        return raw, (time.perf_counter() - started) * 1000

    return run


def _predict(rows: list[dict], runner: Callable[[str], tuple[str, float]]) -> tuple[list[Prediction], list[dict]]:
    predictions, raw_rows = [], []
    for row in rows:
        raw, latency = runner(row["message"])
        level, reason, valid = _parse_risk(raw)
        predictions.append(Prediction(row["id"], row["expected_risk"], level, raw, valid, reason, latency, row.get("category", ""), row.get("layer", "")))
        raw_rows.append({"id": row["id"], "raw": raw, "reason": reason, "latency_ms": latency})
    return predictions, raw_rows


def _rules_predict(message: str) -> str:
    app_root = project_root()
    if app_root is None:
        raise RuntimeError("AEGIS_PROJECT_ROOT is required for rules baseline evaluation")
    sys.path.insert(0, str(app_root))
    from app.assessment import assess_message

    return assess_message(message).risk_level.value


def _fuse(rules: list[Prediction], model: list[Prediction]) -> list[Prediction]:
    order = {"low": 1, "medium": 2, "high": 3}
    return [
        Prediction(
            rule.sample_id,
            rule.expected,
            model_item.predicted if model_item.predicted and order[model_item.predicted] > order[rule.predicted] else rule.predicted,
            model_item.raw_output,
            model_item.json_valid,
            model_item.reason,
            model_item.latency_ms,
            rule.category,
            rule.layer,
        )
        for rule, model_item in zip(rules, model, strict=True)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate risk models on the frozen stress holdout")
    parser.add_argument("--holdout", type=Path, default=None)
    parser.add_argument("--original-model", default="qwen3.5:2b")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--qlora-model")
    group.add_argument("--qlora-model-dir", type=Path)
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output", type=Path, default=TRAIN_ROOT / "reports" / "risk-qlora-eval.json")
    args = parser.parse_args()

    try:
        if not MODEL_NAME_RE.fullmatch(args.original_model.strip()):
            raise ValueError("original model must be an Ollama model identifier")
        configured_project = project_root()
        holdout = args.holdout or (configured_project / "eval" / "fixtures" / "representative_corpus.json" if configured_project else None)
        if holdout is None:
            raise ValueError("--holdout or AEGIS_PROJECT_ROOT is required")
        rows = _load_holdout(holdout.expanduser().resolve())
        rules = [Prediction(row["id"], row["expected_risk"], _rules_predict(row["message"]), category=row.get("category", ""), layer=row.get("layer", "")) for row in rows]
        original, original_raw = _predict(rows, lambda message: _run_ollama(args.original_model.strip(), message, args.timeout))
        if args.qlora_model:
            if not MODEL_NAME_RE.fullmatch(args.qlora_model.strip()):
                raise ValueError("QLoRA model must be an Ollama model identifier")
            qlora, qlora_raw = _predict(rows, lambda message: _run_ollama(args.qlora_model.strip(), message, args.timeout))
            descriptor = {"backend": "ollama", "model": args.qlora_model.strip()}
        else:
            model_dir = under(args.qlora_model_dir, TRAIN_ROOT, "model")
            if not (model_dir / "config.json").is_file():
                raise FileNotFoundError(f"missing config.json in {model_dir}")
            qlora, qlora_raw = _predict(rows, _build_transformers_runner(model_dir, args.max_new_tokens))
            descriptor = {"backend": "transformers", "model_dir": str(model_dir)}
    except Exception as exc:
        print(f"RISK EVALUATION FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    reports = {
        "rules_only": risk_eval_report(rules),
        "original_raw": risk_eval_report(original),
        "rules_union_original": risk_eval_report(_fuse(rules, original)),
        "qlora_raw": risk_eval_report(qlora),
        "rules_union_qlora": risk_eval_report(_fuse(rules, qlora)),
    }
    payload = {
        "holdout": str(holdout),
        "count": len(rows),
        "models": {"original": args.original_model.strip(), "qlora": descriptor},
        "reports": reports,
        "raw_predictions": {"original": original_raw, "qlora": qlora_raw},
    }
    output = under(args.output, TRAIN_ROOT, "output")
    write_report(payload, output)
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
