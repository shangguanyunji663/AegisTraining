r"""Build risk_sft_v3: v2 candidate pool + synthetic implicit high + PsySUICIDE implicit high + curated hard negatives.

Outputs train.jsonl / dev.jsonl / manifest.json / build_report.json under --output-root.
Existing datasets are never modified. Stress-holdout leakage is scanned and rejected per-sample.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(r"D:\PythonProject\aegis-psych-agent")
TRAINING_SRC = PROJECT_ROOT / "training" / "src"
if str(TRAINING_SRC) not in sys.path:
    sys.path.insert(0, str(TRAINING_SRC))

from aegis_training.data_contract import (  # noqa: E402
    DataContractError,
    RiskSample,
    label_distribution,
    normalize_message,
    parse_sample,
    write_jsonl,
    RISK_SYSTEM_PROMPT,
)
from aegis_training.leakage_guard import (  # noqa: E402
    dedupe_against_selected,
    scan_final_holdout_leakage,
)
from aegis_training.source_ingest import (  # noqa: E402
    load_hongzhi_candidate_pool,
    load_project_candidate_pool,
)

METAPHOR_SOURCES = frozenset({"synthetic-implicit-high", "psysuicide"})

REASON_BY_LEVEL = {
    "low": "无明确自身自伤意向",
    "medium": "明显痛苦但无自伤意向",
    "high": "存在自身自伤自杀意向",
}

HARD_NEGATIVES = [
    ("third_person_metaphor", "low", "third_party", "朋友说他最近总想消失，我该怎么安慰他"),
    ("third_person_metaphor", "low", "third_party", "室友说活着没意思，我已经陪他去了咨询中心"),
    ("third_person_explicit", "low", "third_party", "新闻里报道有人跳楼了，看完心情很沉重"),
    ("third_person_explicit", "low", "third_party", "电视剧里的女主角说想离开这个世界，看得我好难过"),
    ("academic_context", "low", "third_party", "论文里引用了青少年自杀率上升的数据"),
    ("academic_context", "low", "third_party", "同学在班会上分享了对轻生新闻的看法"),
    ("social_media", "low", "third_party", "网上刷到有人直播自杀的新闻，平台应该管管"),
    ("colloquial_hedge", "low", "third_party", "朋友开玩笑说不如死了算了，我知道他只是随口一说"),
    ("benign_hyperbole", "low", "self", "今天累死了，只想好好睡一觉"),
    ("benign_hyperbole", "low", "self", "这周作业多得要死，忙疯了"),
    ("benign_hyperbole", "low", "self", "笑死我了，这个视频太好笑了"),
    ("benign_hyperbole", "low", "self", "困死了，早八真的起不来"),
    ("fictional", "low", "fictional", "游戏里角色死了只能重来一局"),
    ("fictional", "low", "third_party", "小说里写到主人公结束了自己的生命，作者笔力真深"),
    ("distress_no_intent", "medium", "self", "最近考试压力好大，晚上翻来覆去睡不着"),
    ("distress_no_intent", "medium", "self", "感觉自己什么都做不好，特别沮丧"),
    ("distress_no_intent", "medium", "self", "和家里吵架了，心里堵得慌，不想说话"),
    ("distress_no_intent", "medium", "self", "实习被拒了七八次，觉得自己特别没用"),
    ("distress_no_intent", "medium", "self", "毕业论文一点头绪都没有，焦虑到胃疼"),
    ("distress_no_intent", "medium", "self", "每天都很低落，对以前喜欢的东西也提不起兴趣"),
    ("distress_no_intent", "medium", "self", "一个人在外地读书，孤独感特别重"),
    ("distress_no_intent", "medium", "self", "挂科之后整个人都很慌，怕毕不了业"),
]


def stable_order(samples: list[RiskSample], seed: int, tag: str) -> list[RiskSample]:
    return sorted(
        samples,
        key=lambda s: hashlib.sha256(f"{seed}:{tag}:{s.sample_id}".encode("utf-8")).hexdigest(),
    )


def read_jsonl_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if isinstance(record, dict):
                rows.append(record)
    return rows


def load_synthetic(path: Path) -> list[RiskSample]:
    return [parse_sample(row) for row in read_jsonl_rows(path)]


def load_psy_implicit(path: Path, target: int, max_chars: int, seed: int) -> tuple[list[RiskSample], dict]:
    drops: Counter[str] = Counter()
    metaphor: list[RiskSample] = []
    other: list[RiskSample] = []
    for row in read_jsonl_rows(path):
        note = str(row.get("adjudication_note") or "")
        sub_match = re.search(r"subtype=(\w+)", note)
        split_match = re.search(r"split=(\w+)", note)
        if not sub_match or not split_match:
            drops["missing_note"] += 1
            continue
        subtype = sub_match.group(1)
        if split_match.group(1) != "train":
            drops["split_not_train"] += 1
            continue
        if row.get("risk_level") != "high":
            drops["not_high"] += 1
            continue
        message = str(row.get("message") or "").strip()
        if len(message) < 2 or len(message) > max_chars:
            drops["message_length"] += 1
            continue
        trimmed = dict(row)
        trimmed["message"] = message
        try:
            sample = parse_sample(trimmed)
        except DataContractError:
            drops["contract_invalid"] += 1
            continue
        (metaphor if subtype == "metaphor_implicit" else other).append(sample)
    metaphor = stable_order(metaphor, seed, "psy-metaphor")
    other = stable_order(other, seed, "psy-other")
    selected = metaphor[:target]
    if len(selected) < target:
        selected = selected + other[: target - len(selected)]
    stats = {
        "metaphor_implicit_available": len(metaphor),
        "other_implicit_available": len(other),
        "selected_total": len(selected),
        "selected_metaphor_implicit": min(len(metaphor), target),
        "selected_other_implicit": max(0, target - len(metaphor)),
        "drops": dict(drops),
    }
    return selected, stats


def build_hard_negatives() -> list[RiskSample]:
    samples: list[RiskSample] = []
    for index, (category, level, scope, message) in enumerate(HARD_NEGATIVES, 1):
        samples.append(
            RiskSample(
                sample_id=f"hardneg-v3-{index:03d}",
                message=message,
                risk_level=level,
                reason=REASON_BY_LEVEL[level],
                source="hard-negative-synthetic",
                source_version="v3",
                label_method="synthetic",
                review_status="not_reviewed",
                annotator="hard-negative-curation-v1",
                speaker_scope=scope,
                adjudication_note=f"category={category}",
            )
        )
    return samples


def split_with_reserves(
    pool: list[RiskSample],
    train_size: int,
    dev_size: int,
    seed: int,
    dev_metaphor_reserve: int,
    train_metaphor_min: int,
    train_metaphor_cap_ratio: float,
) -> tuple[list[RiskSample], list[RiskSample], dict]:
    groups: dict[str, list[RiskSample]] = {"low": [], "medium": [], "high": []}
    for sample in pool:
        groups[sample.risk_level].append(sample)

    high_meta = stable_order(
        [s for s in groups["high"] if s.source in METAPHOR_SOURCES], seed, "high-metaphor"
    )
    high_other = stable_order(
        [s for s in groups["high"] if s.source not in METAPHOR_SOURCES], seed, "high-other"
    )
    lows = stable_order(groups["low"], seed, "low")
    meds = stable_order(groups["medium"], seed, "medium")

    levels = ["low", "medium", "high"]
    train_quotas = {level: train_size // 3 for level in levels}
    for level in levels[: train_size % 3]:
        train_quotas[level] += 1
    dev_quotas = {level: dev_size // 3 for level in levels}
    for level in levels[: dev_size % 3]:
        dev_quotas[level] += 1

    dev_meta_n = min(dev_metaphor_reserve, len(high_meta), dev_quotas["high"])
    dev_meta = high_meta[:dev_meta_n]
    high_meta_rest = high_meta[dev_meta_n:]

    train_cap = int(train_quotas["high"] * train_metaphor_cap_ratio)
    take = min(train_cap, len(high_meta_rest))
    take = max(take, min(train_metaphor_min, len(high_meta_rest)))
    train_meta = high_meta_rest[:take]
    high_meta_rest2 = high_meta_rest[len(train_meta):]

    dev_other_n = dev_quotas["high"] - len(dev_meta)
    dev_other = high_other[:dev_other_n]
    high_other_rest = high_other[len(dev_other):]

    train_other_n = train_quotas["high"] - len(train_meta)
    train_other = high_other_rest[:train_other_n]
    if len(train_other) < train_other_n:
        top_up = train_other_n - len(train_other)
        extra = high_meta_rest2[:top_up]
        train_meta = train_meta + extra

    train_lows = lows[: train_quotas["low"]]
    dev_lows = lows[train_quotas["low"]: train_quotas["low"] + dev_quotas["low"]]
    train_meds = meds[: train_quotas["medium"]]
    dev_meds = meds[train_quotas["medium"]: train_quotas["medium"] + dev_quotas["medium"]]

    if len(dev_lows) < dev_quotas["low"]:
        raise DataContractError("insufficient low candidates for dev")
    if len(dev_meds) < dev_quotas["medium"]:
        raise DataContractError("insufficient medium candidates for dev")
    if len(dev_meta) + len(dev_other) < dev_quotas["high"]:
        raise DataContractError("insufficient high candidates for dev")
    if len(train_meta) + len(train_other) < train_quotas["high"]:
        raise DataContractError("insufficient high candidates for train")

    train = train_lows + train_meds + train_meta + train_other
    dev = dev_lows + dev_meds + dev_meta + dev_other
    stats = {
        "quotas": {"train": train_quotas, "dev": dev_quotas},
        "high_metaphor_pool": len(high_meta),
        "high_other_pool": len(high_other),
        "train_metaphor_actual": len(train_meta),
        "dev_metaphor_actual": len(dev_meta),
    }
    return stable_order(train, seed, "train"), stable_order(dev, seed, "dev"), stats


def manifest_block(samples: list[RiskSample], split: str) -> dict:
    return {
        "split": split,
        "count": len(samples),
        "label_distribution": label_distribution(samples),
        "source_distribution": dict(sorted(Counter(s.source for s in samples).items())),
        "label_method_distribution": dict(sorted(Counter(s.label_method for s in samples).items())),
        "samples": [s.manifest_row() for s in samples],
    }


def normalized_hash(message: str) -> str:
    return hashlib.sha256(normalize_message(message).encode("utf-8")).hexdigest()


def verify_outputs(output_root: Path, train_size: int, dev_size: int) -> dict:
    def load_split(name: str) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for line_number, line in enumerate((output_root / name).read_text(encoding="utf-8").splitlines(), 1):
            record = json.loads(line)
            messages = record["messages"]
            assert [m["role"] for m in messages] == ["system", "user", "assistant"], f"{name}:{line_number} roles"
            assert messages[0]["content"] == RISK_SYSTEM_PROMPT, f"{name}:{line_number} system prompt mismatch"
            answer = json.loads(messages[2]["content"])
            assert answer["risk_level"] in {"low", "medium", "high"}, f"{name}:{line_number} bad label"
            assert isinstance(answer["reason"], str) and len(answer["reason"]) <= 20, f"{name}:{line_number} reason"
            out.append((messages[1]["content"], answer["risk_level"]))
        return out

    train = load_split("train.jsonl")
    dev = load_split("dev.jsonl")
    assert len(train) == train_size, f"train size {len(train)} != {train_size}"
    assert len(dev) == dev_size, f"dev size {len(dev)} != {dev_size}"
    overlap = {normalized_hash(t[0]) for t in train} & {normalized_hash(d[0]) for d in dev}
    assert not overlap, f"train/dev overlap: {len(overlap)} exact duplicates"
    train_labels = Counter(level for _, level in train)
    dev_labels = Counter(level for _, level in dev)
    return {
        "json_valid": True,
        "system_prompt_consistent": True,
        "assistant_schema_valid": True,
        "train_dev_disjoint": True,
        "train_count": len(train),
        "dev_count": len(dev),
        "train_labels": dict(sorted(train_labels.items())),
        "dev_labels": dict(sorted(dev_labels.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build risk_sft_v3 dataset")
    parser.add_argument("--source-root", type=Path, default=Path("D:/AegisTraining/data/external/SupervisedVsLLM-EfficacyEval"))
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--synthetic-path", type=Path, default=Path("D:/AegisTraining/data/risk_sft_v2_round2/synthetic_implicit_high.jsonl"))
    parser.add_argument("--metaphor-path", type=Path, default=Path("D:/AegisTraining/data/external/supplement/metaphor_corpus_v1.jsonl"))
    parser.add_argument("--output-root", type=Path, default=Path("D:/AegisTraining/data/risk_sft_v3"))
    parser.add_argument("--train-size", type=int, default=840)
    parser.add_argument("--dev-size", type=int, default=140)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--psy-target", type=int, default=120)
    parser.add_argument("--psy-max-chars", type=int, default=300)
    parser.add_argument("--dedup-threshold", type=float, default=0.92)
    parser.add_argument("--leak-threshold", type=float, default=0.82)
    parser.add_argument("--dev-metaphor-reserve", type=int, default=20)
    parser.add_argument("--dev-metaphor-min", type=int, default=15)
    parser.add_argument("--train-metaphor-min", type=int, default=60)
    parser.add_argument("--train-metaphor-cap-ratio", type=float, default=0.4)
    args = parser.parse_args()

    try:
        base = load_project_candidate_pool(args.project_root)
        hongzhi = load_hongzhi_candidate_pool(args.source_root)
        synthetic = load_synthetic(args.synthetic_path)
        psy, psy_stats = load_psy_implicit(args.metaphor_path, args.psy_target, args.psy_max_chars, args.seed)
        negatives = build_hard_negatives()

        stage_counts = {
            "project_base": len(base),
            "synthetic_implicit_high": len(synthetic),
            "psysuicide_implicit_selected": len(psy),
            "hard_negatives": len(negatives),
            "hongzhi_pool": len(hongzhi),
        }
        raw_combined = sum(stage_counts.values())

        combined = base + synthetic + psy + negatives + hongzhi
        before_ids = {s.sample_id for s in combined}
        combined_deduped = dedupe_against_selected(combined, near_duplicate_threshold=args.dedup_threshold)
        after_ids = {s.sample_id for s in combined_deduped}
        dedup_removed = sorted(before_ids - after_ids)

        leak_matches = scan_final_holdout_leakage(combined_deduped, root=args.project_root, near_duplicate_threshold=args.leak_threshold)
        leak_ids = {m.sample_id for m in leak_matches}
        pool = [s for s in combined_deduped if s.sample_id not in leak_ids]

        train, dev, split_stats = split_with_reserves(
            pool,
            args.train_size,
            args.dev_size,
            args.seed,
            args.dev_metaphor_reserve,
            args.train_metaphor_min,
            args.train_metaphor_cap_ratio,
        )

        args.output_root.mkdir(parents=True, exist_ok=True)
        write_jsonl((s.to_sft_record() for s in train), args.output_root / "train.jsonl")
        write_jsonl((s.to_sft_record() for s in dev), args.output_root / "dev.jsonl")

        pool_manifest = manifest_block(pool, "candidate_pool")
        pool_manifest.pop("split", None)
        pool_manifest.update(
            {
                "schema_lineage": (
                    "risk_sft_v3 = v2 pipeline pool (hongzhi suicide/socialcd/cognitive + project base)"
                    " + synthetic_implicit_high(v2_round2) + psysuicide implicit high (train split only)"
                    " + curated hard negatives"
                ),
                "stage_counts": stage_counts,
                "raw_combined": raw_combined,
                "dedup_removed_count": len(dedup_removed),
                "holdout_leakage_rejected_count": len(leak_ids),
                "final_pool": len(pool),
            }
        )

        train_block = manifest_block(train, "train")
        dev_block = manifest_block(dev, "dev")

        report = {
            "schema_version": "risk_sft_v3",
            "seed": args.seed,
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "psy_selection": psy_stats,
            "split": split_stats,
            "candidate_pool": {
                "stage_counts": stage_counts,
                "raw_combined": raw_combined,
                "dedup_removed": len(dedup_removed),
                "holdout_leakage_rejected": len(leak_ids),
                "final_pool": len(pool),
            },
            "train": {k: v for k, v in train_block.items() if k != "samples"},
            "dev": {k: v for k, v in dev_block.items() if k != "samples"},
            "reserve_compliance": {
                "train_metaphor_actual": split_stats["train_metaphor_actual"],
                "train_metaphor_min": args.train_metaphor_min,
                "train_metaphor_ge_min": split_stats["train_metaphor_actual"] >= args.train_metaphor_min,
                "dev_metaphor_actual": split_stats["dev_metaphor_actual"],
                "dev_metaphor_min": args.dev_metaphor_min,
                "dev_metaphor_ge_min": split_stats["dev_metaphor_actual"] >= args.dev_metaphor_min,
            },
            "leakage_matches": [
                {"sample_id": m.sample_id, "kind": m.kind, "score": m.score} for m in leak_matches
            ],
        }

        (args.output_root / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "risk_sft_v3",
                    "seed": args.seed,
                    "candidate_pool": pool_manifest,
                    "train": train_block,
                    "dev": dev_block,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        verification = verify_outputs(args.output_root, args.train_size, args.dev_size)
        report["verification"] = verification
        (args.output_root / "build_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except (DataContractError, ValueError, OSError, AssertionError) as exc:
        print(f"V3 BUILD FAILED: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nBUILD OK ->", args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
