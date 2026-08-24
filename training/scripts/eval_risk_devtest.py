# -*- coding: utf-8 -*-
"""对 dev-test（consolidated 同规则重标后的 test.jsonl，1414 条）评测合并模型。

复用 eval_risk_qlora.py 的 transformers 推理器与 JSON 解析器，保证与冻结
stress 验收集完全同一口径。输出 per-class F1 / macro-F1 / 误升级率与延迟。

结果只写入 D:/AegisTraining/reports。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

TRAINING_SRC = Path(__file__).resolve().parents[1] / "src"
if str(TRAINING_SRC) not in sys.path:
    sys.path.insert(0, str(TRAINING_SRC))

from aegis_training.paths import training_root

TRAIN_ROOT = training_root()

from eval_risk_qlora import _build_transformers_runner, _parse_risk  # noqa: E402

LABELS = ("low", "medium", "high")


def _guard(path: Path, allowed_roots: tuple, role: str) -> Path:
    resolved = path.resolve()
    for root in allowed_roots:
        try:
            resolved.relative_to(root.resolve())
            return resolved
        except ValueError:
            continue
    raise ValueError(f"{role} path escapes allowed roots: {resolved}")


def _write_text(root: Path, filename: str, content: str) -> Path:
    target = _guard(root / filename, (TRAIN_ROOT,), "write")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target


def load_test_rows(path: Path) -> list:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        messages = row.get("messages") or []
        if len(messages) < 3:
            raise ValueError(f"{path}:{number}: expected 3 messages")
        gold = json.loads(messages[2]["content"])["risk_level"]
        rows.append({"user": messages[1]["content"], "gold": gold,
                     "system": messages[0]["content"]})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate merged model on dev-test")
    parser.add_argument("--model-dir", type=Path,
                        default=TRAIN_ROOT / "exports" / "aegis-risk-qwen3.5-2b-v4-merged")
    parser.add_argument("--test-jsonl", type=Path,
                        default=TRAIN_ROOT / "data" / "risk_sft_v4" / "test.jsonl")
    parser.add_argument("--output", type=Path,
                        default=TRAIN_ROOT / "reports" / "risk-qlora-devtest-v4.json")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    model_dir = _guard(args.model_dir, (TRAIN_ROOT,), "model")
    test_path = _guard(args.test_jsonl, (TRAIN_ROOT,), "test")
    out_path = _guard(args.output, (TRAIN_ROOT,), "output")

    rows = load_test_rows(test_path)
    run = _build_transformers_runner(model_dir, args.max_new_tokens)

    preds, latencies = [], []
    json_valid = valid_label = reason_over = 0
    confusion = Counter()
    errors = []
    for i, row in enumerate(rows):
        raw, latency_ms = run(row["user"])
        latencies.append(latency_ms)
        label, reason, parsed_ok = _parse_risk(raw)
        if parsed_ok:
            json_valid += 1
        if label in LABELS:
            valid_label += 1
        else:
            label = "low"  # 解析失败按协议回退 low
        if len(reason) > 20:
            reason_over += 1
        gold = row["gold"]
        confusion[(gold, label)] += 1
        if label != gold and len(errors) < 40:
            errors.append({"gold": gold, "pred": label, "user": row["user"][:60],
                           "reason": reason})
        preds.append((gold, label))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(rows)} done", flush=True)

    def prf(label: str) -> dict:
        tp = confusion[(label, label)]
        fp = sum(confusion[(g, label)] for g in LABELS if g != label)
        fn = sum(confusion[(label, p)] for p in LABELS if p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"precision": round(precision, 4), "recall": round(recall, 4),
                "f1": round(f1, 4), "support": tp + fn}

    n = len(rows)
    non_high = sum(1 for g, _ in preds if g != "high")
    medium_total = sum(1 for g, _ in preds if g == "medium")
    lat_sorted = sorted(latencies)
    report = {
        "model_dir": str(model_dir),
        "test_jsonl": str(test_path),
        "count": n,
        "json_valid_rate": round(json_valid / n, 4),
        "valid_label_rate": round(valid_label / n, 4),
        "reason_over_20_rate": round(reason_over / n, 4),
        "accuracy": round(sum(1 for g, p in preds if g == p) / n, 4),
        "per_class": {label: prf(label) for label in LABELS},
        "macro_f1": round(sum(prf(label)["f1"] for label in LABELS) / 3, 4),
        "non_high_to_high_fpr": round(
            sum(1 for g, p in preds if g != "high" and p == "high") / non_high, 4),
        "medium_to_high_rate": round(
            sum(1 for g, p in preds if g == "medium" and p == "high") / medium_total, 4),
        "low_to_medium_rate": round(
            sum(1 for g, p in preds if g == "low" and p != "low")
            / sum(1 for g, _ in preds if g == "low"), 4),
        "latency_ms": {"avg": round(sum(latencies) / n, 1),
                       "p95": round(lat_sorted[int(n * 0.95)], 1)},
        "confusion": {f"{g}->{p}": c for (g, p), c in sorted(confusion.items()) if c},
        "errors_sample": errors,
    }
    _write_text(out_path.parent, out_path.name,
                json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: report[k] for k in
                      ("count", "accuracy", "macro_f1", "non_high_to_high_fpr",
                       "medium_to_high_rate", "json_valid_rate")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
