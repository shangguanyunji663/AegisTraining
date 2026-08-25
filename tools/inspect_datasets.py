import json, os, collections, sys

def jprint(o, n=600):
    s = json.dumps(o, ensure_ascii=False)
    return s[:n] + ("..." if len(s) > n else "")

def inspect_json_array(path, name):
    print("=" * 70)
    print(f"[NEW] {name}  ({path})")
    print(f"  size = {os.path.getsize(path)} bytes")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"  top-level type = {type(data).__name__}")
    if isinstance(data, list):
        print(f"  num records = {len(data)}")
        if data:
            rec = data[0]
            print(f"  record[0] type = {type(rec).__name__}")
            if isinstance(rec, dict):
                for k, v in rec.items():
                    print(f"    field '{k}': type={type(v).__name__} sample={jprint(v, 120)}")
    elif isinstance(data, dict):
        print(f"  top-level keys = {list(data.keys())[:20]}")
        for k, v in list(data.items())[:5]:
            print(f"    key '{k}': type={type(v).__name__} sample={jprint(v, 120)}")
    print()
    return data

def inspect_jsonl(path, name):
    print("=" * 70)
    print(f"[EXIST] {name}  ({path})")
    print(f"  size = {os.path.getsize(path)} bytes")
    n = 0
    keys_counter = collections.Counter()
    sample = None
    field_types = collections.defaultdict(collections.Counter)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n += 1
            try:
                rec = json.loads(line)
            except Exception as e:
                print(f"  !! JSON parse error on line {n}: {e}")
                continue
            if sample is None:
                sample = rec
            if isinstance(rec, dict):
                for k, v in rec.items():
                    keys_counter[k] += 1
                    field_types[k][type(v).__name__] += 1
    print(f"  num records = {n}")
    print(f"  fields (count across records):")
    for k in keys_counter:
        print(f"    '{k}': present_in={keys_counter[k]} types={dict(field_types[k])} ")
    if sample is not None:
        print(f"  sample record[0]:")
        for k, v in sample.items():
            print(f"    '{k}': {jprint(v, 200)}")
    print()
    return None

base = os.environ.get("AEGIS_TRAINING_ROOT", "D:/AegisTraining") + "/data"
external_base = os.environ.get("AEGIS_TRAINING_ROOT", "D:/AegisTraining") + "/external-data"
# NEW dataset
for split in ["train", "valid", "test"]:
    p = os.path.join(external_base, "supplement/PsySUICIDE", f"{split}.json")
    if os.path.exists(p):
        inspect_json_array(p, f"PsySUICIDE/{split}.json")

# EXISTING datasets
for d in ["risk_sft_v1", "risk_sft_v2", "risk_sft_v2_round2"]:
    dp = os.path.join(base, "archive", d)
    if os.path.isdir(dp):
        for fn in sorted(os.listdir(dp)):
            if fn.endswith(".jsonl"):
                inspect_jsonl(os.path.join(dp, fn), f"{d}/{fn}")
