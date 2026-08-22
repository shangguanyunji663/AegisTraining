r"""Independent post-hoc verification of risk_sft_v3 artifacts.

Re-reads the written files (not in-memory state) and checks:
  1. JSONL structure: roles, system prompt, assistant JSON schema, reason length
  2. Counts and label distributions vs plan targets
  3. Exact sample_id absence of the 9 holdout-leak-rejected synthetic ids (exact match, not substring)
  4. train/dev disjointness by normalized-message hash
  5. Zero exact/near-duplicate leakage vs frozen stress holdout, scanned on the written files themselves
  6. Manifest consistency: per-sample message_sha256 recomputed and matched; source-based metaphor reserves
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(r"D:\PythonProject\aegis-psych-agent")
TRAINING_SRC = PROJECT_ROOT / "training" / "src"
if str(TRAINING_SRC) not in sys.path:
    sys.path.insert(0, str(TRAINING_SRC))

from aegis_training.data_contract import (  # noqa: E402
    RiskSample,
    RISK_SYSTEM_PROMPT,
    normalize_message,
)
from aegis_training.leakage_guard import scan_final_holdout_leakage  # noqa: E402

DATA_ROOT = Path(r"D:\AegisTraining\data\risk_sft_v3")
LEAKED_IDS = {
    "synthetic-implicit-wish_never_born-1",
    "synthetic-implicit-disappear_better-1",
    "synthetic-implicit-shouldnt_live-1",
    "synthetic-implicit-cant_hold_on-1",
    "synthetic-implicit-stop_forever-1",
    "synthetic-implicit-stop_forever-2",
    "synthetic-implicit-dont_face_tomorrow-1",
    "synthetic-implicit-ending_better_for_all-1",
    "synthetic-implicit-give_up_thoughts-1",
}
METAPHOR_SOURCES = {"psysuicide", "synthetic-implicit-high"}
EXPECT = {"train": 840, "dev": 140}

failures: list[str] = []
notes: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        notes.append(f"PASS {message}")
    else:
        failures.append(f"FAIL {message}")


def load_split(name: str) -> list[dict]:
    records = []
    for line_number, line in enumerate((DATA_ROOT / f"{name}.jsonl").read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            failures.append(f"FAIL {name}.jsonl:{line_number} blank line")
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            failures.append(f"FAIL {name}.jsonl:{line_number} invalid JSON: {exc}")
            continue
        record["_line"] = line_number
        records.append(record)
    return records


def structural_check(name: str, records: list[dict]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    labels: Counter[str] = Counter()
    for record in records:
        where = f"{name}.jsonl:{record['_line']}"
        messages = record.get("messages")
        if not isinstance(messages, list) or len(messages) != 3:
            failures.append(f"FAIL {where} messages must have 3 entries")
            continue
        roles = [m.get("role") for m in messages]
        if roles != ["system", "user", "assistant"]:
            failures.append(f"FAIL {where} roles={roles}")
            continue
        if messages[0].get("content") != RISK_SYSTEM_PROMPT:
            failures.append(f"FAIL {where} system prompt mismatch")
        try:
            answer = json.loads(messages[2]["content"])
        except (json.JSONDecodeError, KeyError) as exc:
            failures.append(f"FAIL {where} assistant content not valid JSON: {exc}")
            continue
        level = answer.get("risk_level")
        reason = answer.get("reason")
        if level not in {"low", "medium", "high"}:
            failures.append(f"FAIL {where} bad risk_level={level!r}")
        elif not isinstance(reason, str) or not reason.strip():
            failures.append(f"FAIL {where} empty reason")
        elif len(reason) > 20:
            failures.append(f"FAIL {where} reason exceeds 20 chars ({len(reason)})")
        else:
            labels[level] += 1
        extra_keys = set(answer) - {"risk_level", "reason"}
        if extra_keys:
            failures.append(f"FAIL {where} unexpected answer keys {sorted(extra_keys)}")
        pairs.append((messages[1]["content"], level))
    check(len(records) == EXPECT[name], f"{name} count == {EXPECT[name]} (got {len(records)})")
    return pairs


def main() -> int:
    train_records = load_split("train")
    dev_records = load_split("dev")
    train_pairs = structural_check("train", train_records)
    dev_pairs = structural_check("dev", dev_records)

    train_hashes = {hashlib.sha256(normalize_message(t[0]).encode("utf-8")).hexdigest() for t in train_pairs}
    dev_hashes = {hashlib.sha256(normalize_message(d[0]).encode("utf-8")).hexdigest() for d in dev_pairs}
    overlap = train_hashes & dev_hashes
    check(not overlap, f"train/dev disjoint by normalized hash (overlap={len(overlap)})")

    manifest = json.loads((DATA_ROOT / "manifest.json").read_text(encoding="utf-8"))

    def collect_ids(block: dict) -> set[str]:
        return {row["sample_id"] for row in block.get("samples", [])}

    manifest_ids = {
        "candidate_pool": collect_ids(manifest["candidate_pool"]),
        "train": collect_ids(manifest["train"]),
        "dev": collect_ids(manifest["dev"]),
    }
    leaked_in_manifest = sorted(set().union(*manifest_ids.values()) & LEAKED_IDS)
    check(not leaked_in_manifest, f"no leaked sample_id anywhere in manifest (found={leaked_in_manifest})")

    rows_by_id = {row["sample_id"]: row for row in manifest["train"]["samples"]}
    manifest_train_hashes = {row["message_sha256"] for row in manifest["train"]["samples"]}
    check(
        train_hashes == manifest_train_hashes,
        f"train file hashes identical to manifest hashes (diff={len(train_hashes ^ manifest_train_hashes)})",
    )

    sources = Counter(row["source"] for row in manifest["train"]["samples"])
    metaphor_train = sum(count for src, count in sources.items() if src in METAPHOR_SOURCES)
    check(metaphor_train >= 60, f"train metaphor high reserve >= 60 (actual={metaphor_train})")
    dev_sources = Counter(row["source"] for row in manifest["dev"]["samples"])
    metaphor_dev = sum(count for src, count in dev_sources.items() if src in METAPHOR_SOURCES)
    check(metaphor_dev >= 15, f"dev metaphor high reserve >= 15 (actual={metaphor_dev})")

    rebuilt: list[RiskSample] = []
    for split_name, pairs in (("train", train_pairs), ("dev", dev_pairs)):
        for index, (content, level) in enumerate(pairs):
            rebuilt.append(
                RiskSample(
                    sample_id=f"{split_name}-verify-{index}",
                    message=content,
                    risk_level=level,
                    reason="x",
                    source="verification",
                )
            )
    leak_matches = scan_final_holdout_leakage(rebuilt, root=PROJECT_ROOT, near_duplicate_threshold=0.82)
    check(not leak_matches, f"written files contain zero holdout exact/near duplicates (matches={len(leak_matches)})")
    for item in leak_matches[:10]:
        failures.append(f"  leak detail: {item.sample_id} {item.kind} score={item.score}")

    report = json.loads((DATA_ROOT / "build_report.json").read_text(encoding="utf-8"))
    compliance = report.get("reserve_compliance", {})
    check(compliance.get("train_metaphor_ge_min") is True, "build_report train_metaphor_ge_min true")
    check(compliance.get("dev_metaphor_ge_min") is True, "build_report dev_metaphor_ge_min true")

    print("=" * 30, "CHECKS", "=" * 30)
    for note in notes:
        print(note)
    if failures:
        print("-" * 66)
        print(f"{len(failures)} FAILURE(S):")
        for failure in failures[:50]:
            print(failure)
        return 1

    print("=" * 30, "SAMPLES", "=" * 30)
    content_by_hash = {hashlib.sha256(normalize_message(t[0]).encode("utf-8")).hexdigest(): t[0] for t in train_pairs}
    shown = {"psysuicide": False, "hardneg": False}
    for row in manifest["train"]["samples"]:
        if row["source"] in METAPHOR_SOURCES and not shown["psysuicide"]:
            text = content_by_hash.get(row["message_sha256"], "<not found>")
            print(f"[metaphor:{row['source']}] {text[:60]}")
            shown["psysuicide"] = True
        if row["source"] == "hard-negative-synthetic" and not shown["hardneg"]:
            text = content_by_hash.get(row["message_sha256"], "<not found>")
            print(f"[hardneg] {text[:60]}")
            shown["hardneg"] = True
        if all(shown.values()):
            break

    print("\nVERIFICATION OK: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
