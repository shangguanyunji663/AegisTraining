"""Prepare reproducible risk SFT data from approved external sources."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

TRAINING_SRC = Path(__file__).resolve().parents[1] / "src"
if str(TRAINING_SRC) not in sys.path:
    sys.path.insert(0, str(TRAINING_SRC))

from aegis_training.data_contract import DataContractError, RiskSample, label_distribution, write_jsonl
from aegis_training.leakage_guard import assert_no_final_holdout_leakage, dedupe_against_selected
from aegis_training.paths import project_root, training_root, under
from aegis_training.source_ingest import load_hongzhi_candidate_pool, load_project_candidate_pool

TRAIN_ROOT = training_root()


def _stable_order(samples: list[RiskSample], seed: int, label: str) -> list[RiskSample]:
    return sorted(
        samples,
        key=lambda sample: hashlib.sha256(f"{seed}:{label}:{sample.sample_id}".encode("utf-8")).hexdigest(),
    )


def stratified_split(samples: list[RiskSample], train_size: int, dev_size: int, seed: int):
    groups: dict[str, list[RiskSample]] = defaultdict(list)
    for sample in samples:
        groups[sample.risk_level].append(sample)
    levels = ("low", "medium", "high")
    if set(groups) != set(levels):
        raise DataContractError(f"candidate pool must contain {levels}, got {sorted(groups)}")
    train_quota = {level: train_size // 3 for level in levels}
    dev_quota = {level: dev_size // 3 for level in levels}
    for level in levels[: train_size % 3]:
        train_quota[level] += 1
    for level in levels[: dev_size % 3]:
        dev_quota[level] += 1
    train, dev = [], []
    for level in levels:
        ordered = _stable_order(groups[level], seed, level)
        needed = train_quota[level] + dev_quota[level]
        if len(ordered) < needed:
            raise DataContractError(f"{level} pool has {len(ordered)} samples, need {needed}")
        train.extend(ordered[: train_quota[level]])
        dev.extend(ordered[train_quota[level]:needed])
    return _stable_order(train, seed, "train"), _stable_order(dev, seed, "dev")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare isolated risk QLoRA SFT data")
    parser.add_argument("--source-root", type=Path, default=TRAIN_ROOT / "data" / "external" / "SupervisedVsLLM-EfficacyEval")
    parser.add_argument("--project-root", type=Path, default=None, help="optional production checkout; only committed base fixtures are read")
    parser.add_argument("--holdout", type=Path, default=None, help="frozen stress fixture used for leakage checks")
    parser.add_argument("--without-project-base", action="store_true")
    parser.add_argument("--output-root", type=Path, default=TRAIN_ROOT / "data" / "risk_sft_v2")
    parser.add_argument("--train-size", type=int, default=720)
    parser.add_argument("--dev-size", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.82)
    args = parser.parse_args()

    try:
        source_root = args.source_root.expanduser().resolve()
        output_root = under(args.output_root, TRAIN_ROOT, "output")
        configured_project = args.project_root or project_root()
        candidates = load_hongzhi_candidate_pool(source_root)
        project_samples: list[RiskSample] = []
        holdout = args.holdout.expanduser().resolve() if args.holdout else None
        if not args.without_project_base:
            if configured_project is None:
                raise ValueError("--project-root or AEGIS_PROJECT_ROOT is required unless --without-project-base is set")
            configured_project = configured_project.expanduser().resolve()
            project_samples = load_project_candidate_pool(configured_project)
            candidates.extend(project_samples)
            if holdout is None:
                holdout = configured_project / "eval" / "fixtures" / "representative_corpus.json"
        candidates = dedupe_against_selected(candidates)
        if holdout is not None:
            assert_no_final_holdout_leakage(
                candidates,
                near_duplicate_threshold=args.near_duplicate_threshold,
                holdout_path=holdout,
            )
        train, dev = stratified_split(candidates, args.train_size, args.dev_size, args.seed)
        write_jsonl((sample.to_sft_record() for sample in train), output_root / "train.jsonl")
        write_jsonl((sample.to_sft_record() for sample in dev), output_root / "dev.jsonl")
        manifest = {
            "schema_version": "risk_sft_v2",
            "seed": args.seed,
            "source_root": str(source_root),
            "project_root": str(configured_project) if configured_project else None,
            "final_holdout": str(holdout) if holdout else None,
            "candidate_count": len(candidates),
            "project_samples": len(project_samples),
            "train": {"count": len(train), "labels": label_distribution(train)},
            "dev": {"count": len(dev), "labels": label_distribution(dev)},
            "source_distribution": dict(Counter(sample.source for sample in candidates)),
        }
        (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    except (DataContractError, OSError, ValueError) as exc:
        print(f"DATA PREPARATION FAILED: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
