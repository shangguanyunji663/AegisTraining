import json, os, collections, re, statistics

PSY = "D:/AegisTraining/data/external/supplement/PsySUICIDE"
def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)

# ---------- PsySUICIDE deep ----------
allrecs = {}
for split in ["train","valid","test"]:
    recs = load_json(os.path.join(PSY, f"{split}.json"))
    allrecs[split] = recs

# taxonomy: map label name -> vector index via majority vote
name_to_idx = collections.defaultdict(collections.Counter)
idx_present = collections.Counter()
texts_seen = collections.Counter()
src_counter = collections.Counter()
multi = 0
label_combo = collections.Counter()
empty_text = 0
len_stats = []
idx_overlap = collections.defaultdict(set)
for split, recs in allrecs.items():
    for r in recs:
        labs = r.get("labels") or []
        vec = r.get("label") or []
        for nm in labs:
            for i,v in enumerate(vec):
                if v == 1.0:
                    name_to_idx[nm][i] += 1
        src_counter[r.get("data_source")] += 1
        if len(labs) > 1: multi += 1
        label_combo[tuple(sorted(labs))] += 1
        t = r.get("text") or ""
        if not t.strip(): empty_text += 1
        len_stats.append(len(t))
        texts_seen[t] += 1
        idx_overlap[split].add(r.get("idx"))

print("### PsySUICIDE taxonomy (label name -> vector index) ###")
for nm, c in sorted(name_to_idx.items(), key=lambda x:-sum(x[1].values())):
    print(f"  {nm!r}: idx_candidates={dict(c)}")
print()
print(f"records total = {sum(len(v) for v in allrecs.values())}")
print(f"multi-label records = {multi}")
print(f"empty text = {empty_text}")
print(f"data_source distribution = {dict(src_counter)}")
print(f"text length: min={min(len_stats)} max={max(len_stats)} mean={statistics.mean(len_stats):.1f} median={statistics.median(len_stats)}")
# duplicates by exact text
dup = {t:c for t,c in texts_seen.items() if c>1}
print(f"exact-duplicate texts (across all splits) = {len(dup)} (total dup occurrences={sum(dup.values())-len(dup)})")
print(f"top label combos:")
for combo, c in label_combo.most_common(15):
    print(f"  {combo}: {c}")
print(f"idx overlap train&valid={len(idx_overlap['train'] & idx_overlap['valid'])} train&test={len(idx_overlap['train'] & idx_overlap['test'])} valid&test={len(idx_overlap['valid'] & idx_overlap['test'])}")
print()

# ---------- EXISTING SFT messages ----------
def load_jsonl(p):
    out=[]
    with open(p, encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line: out.append(json.loads(line))
    return out

print("### EXISTING risk_sft_v2/train.jsonl messages structure ###")
sft = load_jsonl("D:/AegisTraining/data/risk_sft_v2/train.jsonl")
role_counter = collections.Counter()
assistant_levels = collections.Counter()
samples_full = []
for rec in sft:
    msgs = rec.get("messages", [])
    for m in msgs:
        role_counter[m.get("role")] += 1
    # try to extract risk level from assistant
    asst = [m for m in msgs if m.get("role")=="assistant"]
    if asst:
        content = asst[0].get("content","")
        m = re.search(r'"(risk_level|level)"\s*:\s*"(high|medium|low)"', content) or re.search(r'\b(high|medium|low)\b', content, re.I)
        if m: assistant_levels[m.group(1).lower() if m.lastindex and m.group(1) in ('high','medium','low') else m.group(0).lower()] += 1
    if len(samples_full) < 2:
        samples_full.append(rec)
print(f"  role_counts = {dict(role_counter)}")
print(f"  assistant risk-level tokens = {dict(assistant_levels)}")
print(f"  sample[0] full messages:")
for m in samples_full[0]["messages"]:
    print(f"    [{m['role']}] {m['content'][:400]}")
print()

# ---------- round2 ----------
print("### EXISTING risk_sft_v2_round2/synthetic_implicit_high.jsonl ###")
r2 = load_jsonl("D:/AegisTraining/data/risk_sft_v2_round2/synthetic_implicit_high.jsonl")
rc = collections.Counter()
sc = collections.Counter()
for r in r2:
    rc[r.get("risk_level")] += 1
    sc[r.get("speaker_scope")] += 1
print(f"  risk_level dist = {dict(rc)}")
print(f"  speaker_scope dist = {dict(sc)}")
print(f"  sample message: {r2[0]['message']}")
print()

# ---------- overlap text between PsySUICIDE and round2 / sft user text ----------
psy_texts = set()
for recs in allrecs.values():
    for r in recs:
        psy_texts.add((r.get("text") or "").strip())
print(f"PsySUICIDE unique texts = {len(psy_texts)}")
r2_texts = set((r.get("message") or "").strip() for r in r2)
print(f"round2 unique messages = {len(r2_texts)}")
print(f"exact text overlap PsySUICIDE ∩ round2 = {len(psy_texts & r2_texts)}")
# sft user texts
sft_user_texts=set()
for rec in sft:
    for m in rec.get("messages",[]):
        if m.get("role")=="user":
            sft_user_texts.add(m["content"].strip())
print(f"SFT user texts = {len(sft_user_texts)}")
print(f"exact text overlap PsySUICIDE ∩ SFT-user = {len(psy_texts & sft_user_texts)}")
