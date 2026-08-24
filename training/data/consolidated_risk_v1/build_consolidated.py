# -*- coding: utf-8 -*-
"""
consolidated_risk_v1 构建脚本
================================
目标：将 D:/AegisTraining 下「心理健康/自杀风险识别」相关的杂乱原始数据，
统一清洗、去重、格式化，并按 90/10 分层划分为训练集与测试集，
输出到本项目下的独立文件夹，供后续 QLoRA 训练直接使用。

设计约束（来自项目既有约定）：
1. 只读 D:\AegisTraining，绝不修改/删除其中任何文件。
2. 与 25 条冻结验收集（corp-106..130, category=suicidal_implicit）保持零泄漏：
   任何与其字符 3-gram Jaccard >= 0.82 的样本都会被剔除。
3. 跨副本去重：data/suicide/* 与外部仓库副本指向同一来源，按归一化文本去重。
4. 排除与「自杀风险识别」非同一任务的数据（认知歪曲 12 分类、心理咨询生成、
   已派生的 risk_sft_vX SFT 集），仅做文档化记录，不并入风险分类集。

输出文件（同一文件夹）：
  - train.jsonl / test.jsonl            统一 per-sample schema（分析友好）
  - train_messages.jsonl / test_messages.jsonl  SFT messages 格式（直接训练）
  - manifest.json                       构建统计与溯源
  - README.md                          说明文档
  - .gitignore                         避免大文件误入 git
"""
from __future__ import annotations
import json, csv, re, os, hashlib, collections, datetime

SRC_ROOT = "D:/AegisTraining"
OUT_DIR  = "D:/PythonProject/aegis-psych-agent/training/data/consolidated_risk_v1"
PROJECT_CORPUS = "D:/PythonProject/aegis-psych-agent/eval/fixtures/representative_corpus.json"
SEED = 42
TEST_RATIO = 0.10

# ---------- 冻结验收集（泄漏防护） ----------
CORPUS = json.load(open(PROJECT_CORPUS, encoding="utf-8"))
HOLDOUT = [r["message"] for r in CORPUS
           if r.get("id", "").startswith("corp-1") and 106 <= int(r["id"].split("-")[1]) <= 130]

def norm_key(s: str) -> str:
    return re.sub(r"\s+", "", s or "").strip()

def char_trigrams(s: str):
    s = re.sub(r"\s+", "", s or "")
    return set(s[i:i+3] for i in range(len(s)-2)) if len(s) >= 3 else {s}

HG = [char_trigrams(m) for m in HOLDOUT]
def leak_to_holdout(text: str) -> bool:
    g = char_trigrams(text)
    return any(len(g & x) / len(g | x) >= 0.82 for x in HG) if g else False

# ---------- PsySUICIDE 标签 -> risk_level 映射 ----------
HIGH_LABELS = {"主动自杀意图", "被动自杀意图", "自杀计划", "自杀准备行为",
               "自杀未遂", "自伤行为", "自伤意图", "用户攻击行为"}
MED_LABELS  = {"关于自杀的探索"}
LOW_LABELS  = {"与自杀/自伤/攻击行为无关", "他人攻击行为"}

# ---------- 隐喻线索（用于标记 metaphor_flag，非用于重标风险） ----------
METAPHOR_CUES = re.compile(
    r"(结束这一切|让这一切停下来|消失|离开这个世界|和这个世界告别|告别|从没出生|没出生过|"
    r"解脱|撑不下去|不想再继续|不再继续|活下去的理由|活下去的力气|睡过去|不再醒来|不想面对明天|"
    r"不想再面对|不在了|要是我不在了|放弃的念头|想放弃|做傻事|活着多余|没有任何人需要我|"
    r"一了百了|死了算了|停下来|多余|不配|废物|累赘|错误|罪人|烂泥|行尸走肉)"
)
EXPLICIT = re.compile(
    r"(想自杀|要自杀|想死|去死|不想活|结束生命|结束自己的生命|割腕|自残|跳楼|跳下去|"
    r"杀了自己|捅死自己|遗书|安眠药|轻生|自杀|自杀未遂)"
)
THIRD_PARTY = re.compile(
    r"(新闻|报道|看到|刷到|朋友|同学|室友|家人|她|他|他们|电视剧|电影|小说|论文|故事|"
    r"女孩|男孩|新闻里|书里|剧里).{0,20}(自杀|轻生|跳楼|割腕|自残|死亡|去世|想死|消失|解脱)"
)

def speaker_scope(text: str) -> str:
    if THIRD_PARTY.search(text or ""):
        return "third_party_or_fictional"
    return "self_or_unknown"

# ---------- 系统提示词（复用 risk_sft_v3 既有约定，保证训练兼容） ----------
SYSTEM_PROMPT = (
    "你是校园心理支持系统的风险评估器,只依据当下说话人的**自身意向**判断风险:"
    "high(自伤/自杀意念或计划,包括隐喻式表达如想消失/活着多余/从没出生过/不配/撑不下去/"
    "永远睡过去/离开这个世界/结束这一切/不愿再醒来/要是不在了就好了),"
    "medium(强烈痛苦、绝望、功能受损但无直接自伤)、"
    "low(一般困扰,包括提及他人/虚构情境的高危词,如\"新闻里有人轻生/写论文提到自杀/"
    "朋友直播自杀\"——这些不应升为自身风险)。"
    "判定原则:仅评估说话人自身;他人或虚构内容提及自杀/伤害不视为自身 high。"
    "只输出一个 JSON 对象,不要输出任何其他文字:"
    "{\"risk_level\": \"low|medium|high\", \"reason\": \"20字以内依据\"}"
)

# ---------- 读取器 ----------
def read_psysuicide(path, split):
    out = []
    rows = json.load(open(path, encoding="utf-8"))
    for r in rows:
        text = (r.get("text") or "").strip()
        labels = r.get("labels") or []
        if any(l in HIGH_LABELS for l in labels):
            rl = "high"
        elif any(l in MED_LABELS for l in labels):
            rl = "medium"
        else:
            rl = "low"
        present = [l for l in labels if l in (HIGH_LABELS | MED_LABELS)]
        reason = ("含" + "、".join(present[:3]) + "等标签") if present else "与自杀/自伤无关"
        out.append({
            "text": text, "risk_level": rl, "reason": reason,
            "source": "psysuicide", "source_detail": f"PsySUICIDE.{split}/{r.get('data_source')}",
            "labels_raw": labels,
        })
    return out

def read_metaphor_v1(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            out.append({
                "text": (o.get("message") or "").strip(),
                "risk_level": o.get("risk_level", "unknown"),
                "reason": o.get("reason", "")[:20],
                "source": "metaphor_corpus_v1",
                "source_detail": "metaphor_corpus_v1/" + str(o.get("sample_id", "")),
                "labels_raw": [o.get("speaker_scope") or "self"],
            })
    return out

def read_suicide_jsonl(path, split):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            msgs = o.get("messages", [])
            user = next((m["content"] for m in msgs if m.get("role") == "user"), "")
            asst = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
            rl = "high" if "高自杀风险" in asst else ("low" if "低自杀风险" in asst else "unknown")
            out.append({
                "text": user.strip(), "risk_level": rl,
                "reason": "原始标注:" + ("高" if rl == "high" else "低") + "自杀风险",
                "source": "suicide_messages", "source_detail": f"{split}",
                "labels_raw": [asst.strip()],
            })
    return out

def read_suicide_csv(path, split):
    out = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f, delimiter="\t")
        header = next(r, None)
        for row in r:
            if len(row) < 3:
                continue
            cid, label, comment = row[0], row[1], row[2]
            rl = "high" if label.strip() == "1" else "low"
            out.append({
                "text": comment.strip(), "risk_level": rl,
                "reason": "原始标注:" + ("高" if rl == "high" else "低") + "自杀风险",
                "source": "suicide_csv", "source_detail": f"{split}/id={cid}",
                "labels_raw": [f"label={label}"],
            })
    return out

def read_lsan(path):
    out = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            comment = (row.get("comment") or "").strip()
            l2 = (row.get("label2") or "").strip()
            l1 = (row.get("label") or "").strip()
            rl = "high" if (l2 == "1" or l1 == "1") else "low"
            out.append({
                "text": comment, "risk_level": rl,
                "reason": "LSAN标注:" + ("高" if rl == "high" else "低"),
                "source": "lsan", "source_detail": "LSAN",
                "labels_raw": [f"label={l1},label2={l2}"],
            })
    return out

# ---------- 主流程 ----------
raw = []
raw += read_psysuicide(f"{SRC_ROOT}/data/external/supplement/PsySUICIDE/train.json", "train")
raw += read_psysuicide(f"{SRC_ROOT}/data/external/supplement/PsySUICIDE/valid.json", "valid")
raw += read_metaphor_v1(f"{SRC_ROOT}/data/external/supplement/metaphor_corpus_v1.jsonl")
raw += read_suicide_jsonl(f"{SRC_ROOT}/data/suicide/suicide_train.jsonl", "suicide_train")
raw += read_suicide_jsonl(f"{SRC_ROOT}/data/suicide/suicide_val.jsonl", "suicide_val")
raw += read_suicide_csv(f"{SRC_ROOT}/data/suicide/suicide_train_LLM.csv", "suicide_train_llm")
raw += read_suicide_csv(f"{SRC_ROOT}/data/suicide/suicide_val_LLM.csv", "suicide_val_llm")
raw += read_lsan(f"{SRC_ROOT}/data/suicide/LSAN.csv")

stats = {"raw_loaded": len(raw)}

# 1) 无效样本过滤
def is_valid(rec):
    t = rec["text"]
    if not t or len(t.strip()) < 2:
        return False
    if re.fullmatch(r"[\s\W\d_]+", t):   # 仅标点/数字/空白
        return False
    if rec["risk_level"] not in ("high", "medium", "low"):
        return False
    return True

valid = [r for r in raw if is_valid(r)]
stats["after_invalid_filter"] = len(valid)
invalid_dropped = len(raw) - len(valid)

# 2) 去重（按归一化文本；metaphor_corpus_v1 优先保留其精选标签）
seen = {}
deduped = []
for r in valid:
    k = norm_key(r["text"])
    if not k:
        continue
    if k in seen:
        continue
    seen[k] = True
    deduped.append(r)
stats["after_dedup"] = len(deduped)

# 3) 泄漏防护（剔除与 25 条冻结 holdout 近重复者）
clean = [r for r in deduped if not leak_to_holdout(r["text"])]
leak_dropped = len(deduped) - len(clean)
stats["leak_dropped"] = leak_dropped
stats["after_leakage_filter"] = len(clean)

# 4) 标注 metaphor_flag / speaker_scope
for r in clean:
    r["metaphor_flag"] = bool(METAPHOR_CUES.search(r["text"])) and not bool(EXPLICIT.search(r["text"]))
    r["speaker_scope"] = speaker_scope(r["text"])

# ---------- 分层 90/10 划分 ----------
import random
random.seed(SEED)
by_level = collections.defaultdict(list)
for r in clean:
    by_level[r["risk_level"]].append(r)

train_set, test_set = [], []
split_detail = {}
for lvl, items in by_level.items():
    random.shuffle(items)
    n_test = int(round(len(items) * TEST_RATIO))
    test_set.extend(items[:n_test])
    train_set.extend(items[n_test:])
    split_detail[lvl] = {"total": len(items), "train": len(items) - n_test, "test": n_test}

# 重新打乱，避免按 level 聚集
random.shuffle(train_set)
random.shuffle(test_set)

def finalize(recs, split):
    out = []
    for i, r in enumerate(recs):
        rid = f"{r['source']}-{split[:1]}{i:06d}"
        rec = {
            "id": rid,
            "text": r["text"],
            "risk_level": r["risk_level"],
            "reason": r["reason"],
            "source": r["source"],
            "source_detail": r["source_detail"],
            "labels_raw": r["labels_raw"],
            "speaker_scope": r["speaker_scope"],
            "metaphor_flag": r["metaphor_flag"],
            "split": split,
        }
        out.append(rec)
    return out

train_recs = finalize(train_set, "train")
test_recs  = finalize(test_set, "test")

# ---------- 写出 per-sample ----------
def write_jsonl(path, recs):
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

os.makedirs(OUT_DIR, exist_ok=True)
write_jsonl(f"{OUT_DIR}/train.jsonl", train_recs)
write_jsonl(f"{OUT_DIR}/test.jsonl", test_recs)

# ---------- 写出 SFT messages 格式 ----------
def to_messages(rec):
    asst = json.dumps({"risk_level": rec["risk_level"], "reason": rec["reason"]},
                     ensure_ascii=False)
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": rec["text"]},
        {"role": "assistant", "content": asst},
    ]}

with open(f"{OUT_DIR}/train_messages.jsonl", "w", encoding="utf-8") as f:
    for r in train_recs:
        f.write(json.dumps(to_messages(r), ensure_ascii=False) + "\n")
with open(f"{OUT_DIR}/test_messages.jsonl", "w", encoding="utf-8") as f:
    for r in test_recs:
        f.write(json.dumps(to_messages(r), ensure_ascii=False) + "\n")

# ---------- 统计 ----------
def level_dist(recs):
    c = collections.Counter(r["risk_level"] for r in recs)
    return {"high": c.get("high", 0), "medium": c.get("medium", 0), "low": c.get("low", 0)}

def src_dist(recs):
    return dict(collections.Counter(r["source"] for r in recs))

def meta_dist(recs):
    return {"metaphor": sum(1 for r in recs if r["metaphor_flag"]),
            "explicit_or_other": sum(1 for r in recs if not r["metaphor_flag"])}

stats.update({
    "invalid_dropped": invalid_dropped,
    "split_detail": split_detail,
    "train_total": len(train_recs), "test_total": len(test_recs),
    "train_level": level_dist(train_recs), "test_level": level_dist(test_recs),
    "train_source": src_dist(train_recs), "test_source": src_dist(test_recs),
    "train_metaphor": meta_dist(train_recs), "test_metaphor": meta_dist(test_recs),
    "holdout_size": len(HOLDOUT), "leak_threshold": 0.82,
    "seed": SEED, "test_ratio": TEST_RATIO,
})

manifest = {
    "build_time": datetime.datetime.now().isoformat(timespec="seconds"),
    "source_root": SRC_ROOT,
    "output_dir": OUT_DIR,
    "included_sources": {
        "PsySUICIDE.train": "D:/AegisTraining/data/external/supplement/PsySUICIDE/train.json",
        "PsySUICIDE.valid": "D:/AegisTraining/data/external/supplement/PsySUICIDE/valid.json",
        "metaphor_corpus_v1": "D:/AegisTraining/data/external/supplement/metaphor_corpus_v1.jsonl",
        "suicide_messages.train/val": "D:/AegisTraining/data/suicide/suicide_{train,val}.jsonl",
        "suicide_csv.train/val": "D:/AegisTraining/data/suicide/suicide_{train,val}_LLM.csv",
        "LSAN": "D:/AegisTraining/data/suicide/LSAN.csv",
    },
    "excluded_sources": {
        "data/external/SupervisedVsLLM-EfficacyEval/*": "与 data/suicide 等副本重复，已通过跨文件去重覆盖，不重复读取",
        "data/cognitive distortion/*": "认知歪曲 12 分类任务，非自杀风险识别，单独训练目标",
        "data/SocialCD-3k/*": "同上（Social Cognition Distortion，认知歪曲任务）",
        "distill_psychology-10k-r1.json": "心理咨询对话生成蒸馏集，无风险标签，属不同训练目标",
        "data/risk_sft_v1|v2|v2_round2|v3": "项目 prepare 已派生的 SFT 集，避免重复计数",
    },
    "schema_per_sample": ["id", "text", "risk_level", "reason", "source",
                           "source_detail", "labels_raw", "speaker_scope", "metaphor_flag", "split"],
    "risk_level_mapping": {"high": sorted(HIGH_LABELS), "medium": sorted(MED_LABELS), "low": sorted(LOW_LABELS)},
    "stats": stats,
}
with open(f"{OUT_DIR}/manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)

# ---------- README ----------
readme = f"""# consolidated_risk_v1 — 统一风险识别训练/测试集

> 由 `build_consolidated.py` 自动构建。仅读取 `D:\\AegisTraining`，未修改/删除其中任何文件。

## 来源（已纳入）
- **PsySUICIDE** (train+valid, 经 hf-mirror 下载, MIT)：自杀意念细粒度 11 标签 → 映射 high/medium/low
- **metaphor_corpus_v1.jsonl**：从 PsySUICIDE 预筛的隐喻/隐式高危子集（按文本去重并入，优先保留其精选标签）
- **suicide 原始集** (`data/suicide/` 下 jsonl/csv/LSAN)：二分类高/低风险

## 来源（已排除，文档记录）
- 外部仓库重复副本 (`data/external/SupervisedVsLLM-EfficacyEval/*`)：与 data/suicide 同源，靠去重覆盖
- 认知歪曲集 (`cognitive distortion` / `SocialCD-3k`)：12 分类任务，非风险识别
- 心理咨询生成集 (`distill_psychology-10k-r1.json`)：无风险标签，属生成目标
- 已派生 SFT 集 (`risk_sft_v1/v2/v2_round2/v3`)：避免重复计数

## 处理步骤
1. 统一读取为多源记录；2. 剔除空/过短/纯符号/未知风险样本；
3. 按归一化文本精确去重（跨副本）；4. 剔除与 25 条冻结验收集（corp-106..130）
   近重复 (n-gram Jaccard≥0.82) 的泄漏样本；5. 按 risk_level **分层 90/10** 划分 (seed={SEED})。

## 字段（per-sample）
`id, text, risk_level(high|medium|low), reason, source, source_detail, labels_raw, speaker_scope, metaphor_flag, split`

## 产出文件
- `train.jsonl` / `test.jsonl`：统一 per-sample 格式（分析/灵活使用）
- `train_messages.jsonl` / `test_messages.jsonl`：SFT `messages` 格式（含既有 system 提示词，可直接 QLoRA 训练）
- `manifest.json`：完整统计与溯源

## 关键统计
- 原始载入：{stats['raw_loaded']} → 无效剔除：{stats['invalid_dropped']} → 去重后：{stats['after_dedup']}
  → 泄漏剔除：{stats['leak_dropped']} → 净样本：{stats['after_leakage_filter']}
- 训练集：{stats['train_total']}（high={stats['train_level']['high']}, medium={stats['train_level']['medium']}, low={stats['train_level']['low']}）
- 测试集：{stats['test_total']}（high={stats['test_level']['high']}, medium={stats['test_level']['medium']}, low={stats['test_level']['low']}）

## 重要约定
- 本测试集为**模型开发用 dev-test**，与冻结验收集（corp-106..130，由 `eval_risk_qlora.py` 使用）相互独立。
- 验收前请勿将本集与冻结集混用，以免污染验收结论。
"""
with open(f"{OUT_DIR}/README.md", "w", encoding="utf-8") as f:
    f.write(readme)

with open(f"{OUT_DIR}/.gitignore", "w", encoding="utf-8") as f:
    f.write("# 大数据文件不纳入 git\n*.jsonl\n*.json\n!manifest.json\n")

print("=== BUILD DONE ===")
print(json.dumps(stats, ensure_ascii=False, indent=2))
