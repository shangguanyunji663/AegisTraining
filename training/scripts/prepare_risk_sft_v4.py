# -*- coding: utf-8 -*-
"""基于 consolidated_risk_v1 构建 risk_sft_v4 训练资产。

全部产物只写入 D:/AegisTraining，项目仓库仅作只读来源（冻结语料）。

v4 相对 v3 的全部变更来自 2026-08-22 的数据审计结论：
1. 规模 840 -> 2000（train 1800 / dev 200），满足"至少一两千"；
2. 重标二分类来源标签噪音：自身显式意念被标 low 的样本改为 high；
3. 痛苦短语但无自身意念标记的 low 样本改为 medium（针对 v3 三条误升级模式）；
4. 「用户攻击行为」（对他人攻击）样本整体移出训练；
5. reason 按细标签一对一重写为 <=12 字短语，消除模板坍缩与超 20 字；
6. 泄漏防护覆盖全部 87 条 stress 冻结集（exact + 3-gram Jaccard >= 0.82）；
7. 新增 48 条边界 medium 困难负例 + 6 条边界 high 对照（hard-negative-v4）。

抽样不用 random：以 sha256(SEED|文本) 排序做确定性洗牌，保证同机可复现。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Allow direct execution from a checkout without setting PYTHONPATH.
TRAINING_SRC = Path(__file__).resolve().parents[1] / "src"
if str(TRAINING_SRC) not in sys.path:
    sys.path.insert(0, str(TRAINING_SRC))

from aegis_training.data_contract import RISK_SYSTEM_PROMPT
from aegis_training.paths import configured_project_corpus, project_root, training_root, under

SEED = 42
TRAIN_ROOT = training_root()

# Keep data preparation and production inference on the same versioned contract.
SYSTEM_PROMPT = RISK_SYSTEM_PROMPT

# 意念表达（含被动死亡意念）：匹配后按「就近主语」判定归属。
# (?<!不) 排除否定式「不想死」；泛指/他人主语由 OTHER_SUBJECT 豁免。
# 「活不下去」为多义词（经济困境/死亡意念），按用户裁决归痛苦短语（medium）。
IDEATION = re.compile(
    r"((?<!不)想死|(?<!不)想去死|真想死|好想死|不想活|想一了百了|一了百了|"
    r"死了算了|不如死了|去死一死|生不如死|想消失|想解脱|想结束这一切|想结束生命|"
    r"不想醒来|永远睡过去|活着多余|没出生|不该活着|不配活着|想自杀|要自杀|轻生|"
    r"无痛死|寻死|求死|(?<!想)去死|找.{0,6}死.{0,4}方法|幻想.{0,8}(死|葬礼)|"
    r"离开这个世界|告别这个世界|说一声再见)"
)
# 被动自杀意图降级判据：纯无意义/自我否定 且不含任何死亡愿望词 -> medium。
MEANINGLESS = re.compile(r"(没用|拖累|不配|多余|图什么|没意义|没意思|废物|累赘|像个错)")
DEATH_CUE = re.compile(
    r"(死|不想活|消失|不在|睡|解脱|结束|再见|离开|出生|自杀|轻生)")
# 第五版（扩量）：从 low 池挖 medium 的痛苦线索（2026-08-23 用户分组裁决）。
# 界限：单纯「压力大/令人伤心」一句带过、单晚失眠 -> 维持 low。
MEDIUM_MINE = [
    (re.compile(r"(沉浸在?痛苦|内心.{0,4}痛苦|感到痛苦|很痛苦|太痛苦)"), "mine_emotion"),
    # 紧邻式：我(副词)?伤心/难过；「我减少了伤心」等正向表述不匹配。
    (re.compile(r"(我|自己)(会|就|也|真的?|总是?|经常|很|特别|十分|非常|超级)?(伤心|难过)"), "mine_emotion"),
    (re.compile(r"(情绪不稳定|情绪暴躁|心情低落|很低落|特别?空虚|无力感)"), "mine_emotion"),
    (re.compile(r"(心跳过速|呼吸急促|手脚发麻|浑身无力|眼前发黑)"), "mine_somatic"),
    (re.compile(r"(分手|失恋).{0,15}(接受不了|活在.{0,4}回忆|难以适应|困惑|痛苦)"), "mine_social"),
    (re.compile(r"(讨厌自己|恨自己|自卑|无法接受.{0,4}自己|窝囊|一无是处)"), "mine_selfneg"),
    (re.compile(r"(扛不住|失去掌控)"), "mine_pressure"),
    (re.compile(r"(睡不着|失眠).{0,12}(害怕|不想失眠|长期|一直|每天|整夜|严重|困扰)"), "mine_somatic"),
]
REASON_MINE = {
    "mine_emotion": "情绪低落痛苦无自伤",
    "mine_somatic": "睡眠躯体困扰受损",
    "mine_social": "挫折后持续痛苦",
    "mine_selfneg": "自我否定无死亡意念",
    "mine_pressure": "压力下失控感",
}
SELF_ACT_ANCHORED = re.compile(
    r"(我|咱).{0,6}(割腕|自残|自伤|跳楼|跳下去|上吊|吞药|安眠药|遗书|流血)"
)
SELF_ACT_BARE = re.compile(
    r"((割|划|伤|扇|打|抽|掐)自己|打自己耳光)"
)
# 自伤裸表述的语境守卫：康复叙述（而不是/拒绝…）-> medium，科普讨论（研究/方式/效果…）-> 不升级。
RECOVERY_GUARD = re.compile(r"(而不是|不该|不再|不用|别用|拒绝|远离|停止|替代)")
DISCUSSION_GUARD = re.compile(r"(研究|方式|方法|效果|一样|获得|跟)")
# 锚定自伤/遗书等的既往时态标记：童年/多年前的旧事 -> medium 而非 high。
PAST_GUARD = re.compile(r"(小时候|小时候|当年|曾经|以前|多年前|过去|那年|上初中|上高中)")
OTHER_SUBJECT = re.compile(
    r"(朋友|同学|室友|家人|亲戚|儿子|女儿|父亲|母亲|爸妈|父母|老师|孩子|你|您|她|他|"
    r"他们|别人|网友|很多人|人们|大家|有些人|有的人|有人|一般人|普通人|正常人|"
    r"大多数人|新闻|报道|看到|刷到|网上|电视剧|电影|小说|论文|书里|剧里)"
)
SELF_TOKEN = re.compile(r"(我|自己|咱)")
PAIN_NO_INTENT = re.compile(
    r"(撑不下|撑不住|硬撑|图什么|没意义|没什么意义|垮了|垮掉|崩溃|煎熬|撑到极限|绷不住|活不下去)"
)
SYMPTOM = re.compile(
    r"(我.{0,6}(抑郁症|抑郁了|抑郁倾向|很抑郁|焦虑症|焦虑发作|整夜睡不着|长期失眠)|"
    r"(确诊|患有|得了|查出)(了)?(抑郁|焦虑)|整天睡不着|严重失眠)"
)


def _nearest_subject_is_self(text: str, start: int, window: int = 12) -> bool | None:
    """就近主语裁决：窗口内离匹配点最近的主语是自身/他人/无主语。

    返回 True=自身，False=他人，None=无主语（默认自身，第一人称叙述）。
    """
    span = text[max(0, start - window):start]
    self_pos = [m.start() for m in SELF_TOKEN.finditer(span)]
    other_pos = [m.start() for m in OTHER_SUBJECT.finditer(span)]
    if not self_pos and not other_pos:
        return None
    if self_pos and (not other_pos or max(self_pos) > max(other_pos)):
        return True
    return False


def _self_ideation(text: str) -> bool:
    """意念表达的就近主语裁决：我/自己 -> 自身；纯他人主语 -> 非自身。

    锚定自伤/遗书带既往时态（小时候/曾经）不判 high；裸自伤表述被
    康复叙述（而不是…）或科普讨论（研究/方式…）守卫时也不判 high。
    """
    for m in SELF_ACT_ANCHORED.finditer(text):
        if PAST_GUARD.search(text[max(0, m.start() - 15):m.start()]):
            continue
        if _nearest_subject_is_self(text, m.start()) is False:
            continue
        return True
    for m in SELF_ACT_BARE.finditer(text):
        pre = text[max(0, m.start() - 10):m.start()]
        post = text[m.end():m.end() + 10]
        if RECOVERY_GUARD.search(pre):
            continue
        if DISCUSSION_GUARD.search(pre) or DISCUSSION_GUARD.search(post):
            continue
        return True
    for m in IDEATION.finditer(text):
        if _nearest_subject_is_self(text, m.start()) is False:
            continue
        return True  # 自身或无主语（第一人称叙述默认）
    return False


def _selfharm_history(text: str) -> bool:
    """自伤/遗书旧事或康复叙述 -> medium：既往经历有困扰但非当前意念。"""
    for m in SELF_ACT_ANCHORED.finditer(text):
        if PAST_GUARD.search(text[max(0, m.start() - 15):m.start()]):
            return True
    for m in SELF_ACT_BARE.finditer(text):
        if RECOVERY_GUARD.search(text[max(0, m.start() - 10):m.start()]):
            return True
    return False


def _self_pain(text: str) -> bool:
    """痛苦短语的就近主语裁决（窗口放宽到 20 字，覆盖他人痛苦叙述）。"""
    for m in PAIN_NO_INTENT.finditer(text):
        if _nearest_subject_is_self(text, m.start(), window=20) is False:
            continue
        return True
    return False


def _self_symptom(text: str) -> bool:
    """显著症状表述的就近主语裁决：他人患病（亲戚得了抑郁症）不算自身。"""
    for m in SYMPTOM.finditer(text):
        verdict = _nearest_subject_is_self(text, m.start())
        if verdict is False:
            continue  # 该处症状属于他人，继续看下一处
        return True
    return False


def _mine_medium(text: str):
    """从 low 文本挖痛苦线索，返回 medium 细类（用户 2026-08-23 分组裁决）。"""
    for pattern, fine in MEDIUM_MINE:
        if pattern.search(text):
            return fine
    return None


def load_campus_medium(path: Path) -> list:
    """第五版扩量：人工编写的校园语境 medium 合成样本（text/reason）。"""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        rows.append({"text": o["text"].strip(), "risk_level": "medium",
                     "reason": o["reason"], "_origin": "campus-medium-v7",
                     "fine": "campus"})
    return rows


def mine_distill_medium(path: Path, limit: int, exclude_keys: set) -> list:
    """从心理咨询蒸馏集的 input 侧挖 medium：无任何意念词、非他人主语、
    命中痛苦/症状/挖掘规则；按 3-gram Jaccard 做多样性去重。"""
    THIRD_HEAD = re.compile(
        r"^(朋友|同学|室友|她|他|他们|新闻|电影|论文|亲戚|父母|老师|孩子|大家|人们|儿子|女儿)")
    kept: list = []
    kept_grams: list = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = re.sub(r"\s+", " ", (o.get("input") or "")).strip()
            if not (10 <= len(t) <= 90):
                continue
            if _norm(t) in exclude_keys:
                continue
            if _self_ideation(t):
                continue
            if THIRD_HEAD.search(t):
                continue
            if not (_self_symptom(t) or _self_pain(t) or _mine_medium(t)):
                continue
            g = _trigrams(t)
            if any(len(g & kg) / len(g | kg) >= 0.6 for kg in kept_grams if g | kg):
                continue
            fine = _mine_medium(t) or ("symptom" if _self_symptom(t) else "pain")
            rows_item = {"text": t, "risk_level": "medium",
                         "_origin": "distill-mined-v7", "fine": fine}
            kept.append(rows_item)
            kept_grams.append(g)
            if len(kept) >= limit:
                break
    return kept
THIRD_PARTY = re.compile(
    r"(新闻|报道|看到|刷到|朋友|同学|室友|家人|她|他|他们|电视剧|电影|小说|论文|故事|"
    r"女孩|男孩|书里|剧里).{0,20}(自杀|轻生|跳楼|割腕|自残|死亡|去世|想死|消失|解脱)"
)
AGGRESSION_ONLY = "用户攻击行为"

REASON_BY_FINE_LABEL = {
    "主动自杀意图": "明确表达自杀意图",
    "被动自杀意图": "存在消极死亡意念",
    "自杀计划": "提及自杀计划",
    "自杀准备行为": "存在自杀准备行为",
    "自杀未遂": "有自杀未遂经历",
    "自伤行为": "存在自伤行为",
    "自伤意图": "表达自伤意图",
}
REASON_MEDIUM = {
    "exploration": "探索自杀话题无明确意念",
    "pain": "强烈痛苦无自伤意图",
    "symptom": "确诊症状属功能受损",
    "selfharm_history": "自伤旧事或康复诉求",
    "passive_demote": "无意义感无死亡意念",
    "hard_negative": "表达痛苦但无自伤意图",
}
REASON_LOW_VARIANTS = [
    "一般困扰无自伤意向",
    "日常表达无风险信号",
    "未表露自身风险意向",
    "一般心理困扰非危机",
]
REASON_LOW_THIRD_PARTY = "提及他人不评估自身"
REASON_HIGH_GENERIC = "表达自身自杀风险"
MAX_REASON_CHARS = 20


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text or "").strip()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _order_key(row_or_text) -> str:
    text = row_or_text["text"] if isinstance(row_or_text, dict) else row_or_text
    return _sha256(f"{SEED}|{_norm(text)}")


def _trigrams(text: str) -> set:
    s = _norm(text)
    return {s[i:i + 3] for i in range(len(s) - 2)} if len(s) >= 3 else {s}


def _guard(path: Path, allowed_roots: tuple, role: str) -> Path:
    """规范化并校验路径必须位于允许目录内，禁止 ../ 逃逸。"""
    for root in allowed_roots:
        try:
            return under(path, root, role)
        except ValueError:
            continue
    raise ValueError(f"{role} path escapes allowed roots: {path.resolve()}")


def _write_text(root: Path, filename: str, content: str) -> Path:
    """唯一写出入口：目标必须解析到 TRAIN_ROOT 内，校验与写入紧邻。"""
    target = _guard(root / filename, (TRAIN_ROOT,), "write")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target


# ---------- 48 条边界 medium 困难负例 + 6 条边界 high 对照 ----------
HARD_NEGATIVE_MEDIUM = [
    ("我最近一直在硬撑，感觉快撑不住了，但还能坚持到期末", "硬撑疲惫但无自伤意图"),
    ("每天都在硬撑着上课，其实已经很累了，想找人说说", "硬撑疲惫但无自伤意图"),
    ("实验做不出来，我快撑不住了，导师还一直催进度", "压力撑不住属强烈痛苦"),
    ("兼职和学业两头跑，我硬撑了一个月，现在真的很疲惫", "硬撑疲惫但无自伤意图"),
    ("我一直硬撑着不掉队，但最近总觉得力不从心", "力不从心无自伤意图"),
    ("家里出事了，我还在硬撑着考试，心里很堵", "家庭压力下硬撑痛苦"),
    ("快撑不下去了，所以来预约一下心理咨询", "主动求助无自伤意图"),
    ("大家都觉得我状态还行，其实我全靠硬撑", "表面正常实为硬撑痛苦"),
    ("我硬撑着参加完了比赛，现在只想好好睡一觉", "耗竭疲惫非自伤意图"),
    ("实习压力大到快撑不住，但我没有想过放弃", "高压撑不住仍无放弃意念"),
    ("论文改到第五版了，我快撑不住了，想吐槽一下", "学业压力撑不住痛苦"),
    ("撑不住的时候我就去操场跑两圈，勉强缓过来", "有自我调节的撑不住"),
    ("我知道自己在硬撑，但是不敢让爸妈担心", "硬撑且压抑求助无自伤"),
    ("连着三周赶ddl，我感觉自己快到极限了", "赶工到极限的功能受损"),
    ("考研备考让我快撑不住了，来问问怎么调节", "备考撑不住主动求助"),
    ("我一边打工一边上课，撑得很辛苦，想找人说说话", "辛苦撑住的倾诉需求"),
    ("我已经不知道每天上课图什么了，先熬过这学期吧", "追问意义但仍在坚持"),
    ("感觉做什么都没意义，游戏也不想打了", "兴趣减退无自伤意图"),
    ("不知道活着图什么，但饭还是照常吃，课也照常上", "无意义感但功能保留"),
    ("最近觉得一切都没什么意思，是不是我太丧了", "兴致缺失的自我觉察"),
    ("学习好像失去了意义，我只是在机械地完成任务", "机械感无自伤意图"),
    ("我不知道这些努力有什么意义，成绩也就那样", "努力无意义感的低落"),
    ("每天重复一样的生活，说不上图什么，就是有点空", "空虚感非危机信号"),
    ("感觉自己做的事情没有价值，心情低落了一阵子", "自我价值低落无自伤"),
    ("不知道图什么，但还是按部就班去上课了", "无意义感但功能保留"),
    ("期末周觉得一切都没意义，考完应该就好了吧", "阶段性无意义感"),
    ("我最近常问自己图什么，可能就是惯性在支撑", "惯性支撑的无意义感"),
    ("保研失败后觉得没什么意义，打算先找份实习", "挫折后低落有下一步"),
    ("对什么都提不起兴趣，觉得没啥意义，想咨询一下", "兴趣减退主动求助"),
    ("不知道以后图什么，先把手头的实验做完再说", "无意义感但任务延续"),
    ("我怕自己哪天就彻底垮了，所以提前来聊聊", "担忧崩溃主动求助"),
    ("最近情绪一直紧绷，我怕再这样下去会崩溃", "紧绷怕崩溃的痛苦"),
    ("感觉自己快到崩溃边缘了，来找你说说话", "近崩溃的倾诉需求"),
    ("我怕考试周会先垮掉，想学点放松的方法", "怕垮且寻求方法"),
    ("白天装作没事，晚上怕自己绷不住崩溃", "强撑之下的崩溃担忧"),
    ("医生说我有点焦虑，我怕状态继续垮下去", "确诊焦虑怕恶化"),
    ("我怕自己垮了影响团队，所以想调整一下", "责任感驱动的调整"),
    ("快崩溃的时候我会听歌，但最近不太管用了", "调节失效的痛苦"),
    ("我怕哪天彻底垮了，趁着现在还想自救一下", "怕垮且有自救意愿"),
    ("担心自己精神先垮，成绩再掉就真撑不住了", "精神状态的担忧"),
    ("最近老是想哭，怕自己情绪崩溃，来倾诉一下", "情绪濒溃的倾诉"),
    ("我怕自己在答辩前垮掉，想找点心理支持", "怕垮且寻求支持"),
    ("感觉压力山大的同时还有点怕，怕自己垮掉", "压力与怕垮并存"),
    ("我怕自己的状态垮下来，想规律作息但是很难", "怕垮想调整作息"),
    ("一想到下周还有三门考试，我怕我会当场崩溃", "考试压力的崩溃担忧"),
    ("怕垮又不敢停，来问问大家怎么排解压力", "怕垮且主动求助"),
    ("我怕自己情绪垮掉说错话，最近都不敢见人", "怕垮回避社交"),
    ("总怕自己哪天垮了，所以想养成运动的习惯", "怕垮且建设性应对"),
]

HARD_NEGATIVE_HIGH_CONTRAST = [
    ("我真的撑不下去了，不想再这样熬下去了，只想彻底解脱", "流露解脱结束意图"),
    ("这样活着图什么都不重要了，我已经开始琢磨怎么离开", "表达离开世界意图"),
    ("我怕的不是垮掉，而是怕自己扛不住去做傻事", "担忧自己做出傻事"),
    ("每天硬撑着太累了，我在认真考虑怎么结束这一切", "考虑结束一切属高危"),
    ("就算垮了也无所谓，我已经把遗书都写好了", "写好遗书属明确高危"),
    ("不知道活着图什么，我想睡过去就再也不用醒来", "希望长睡不醒的高危"),
]

# ---------- 第五版新增：成对对照（自身 vs 第三人称，同句式家族） ----------
# 目的：教模型区分「主语归属」而不是按痛苦词汇就近分类。
# (text, level, reason)；自身侧无自身意念词的为 medium，含意念词的为 high。
PAIRED_CONTRAST = [
    ("我快撑不住了，想找人说说话", "medium", "自身痛苦但无自伤意图"),
    ("室友说她快撑不住了，我该怎么安慰她", "low", "提及他人不评估自身"),
    ("我最近总睡不着，心里堵得慌", "medium", "失眠困扰属功能受损"),
    ("我朋友总失眠，我要不要劝他去咨询", "low", "提及他人不评估自身"),
    ("我不知道这学期还能不能熬下去", "medium", "迷茫痛苦无自伤意图"),
    ("他说他快熬不下去了，我有点担心", "low", "提及他人不评估自身"),
    ("我最近觉得自己特别没用", "medium", "自我否定无死亡意念"),
    ("她总说自己没用，我想帮她但不知道怎么开口", "low", "提及他人不评估自身"),
    ("我对什么都提不起兴趣了", "medium", "兴趣减退无自伤意图"),
    ("同学说他最近对什么都没兴趣，我该怎么帮他", "low", "提及他人不评估自身"),
    ("我怕自己哪天情绪崩溃", "medium", "担忧崩溃属强烈痛苦"),
    ("我担心表姐情绪崩溃，想安慰她", "low", "提及他人不评估自身"),
    ("我最近常想哭，也不知道为什么", "medium", "情绪低落无自伤意图"),
    ("她动不动就哭，我想安慰但怕说错话", "low", "提及他人不评估自身"),
    ("我每天都很累，休息不过来", "medium", "耗竭疲惫属功能受损"),
    ("他连着加班很累，我劝他休息他不听", "low", "提及他人不评估自身"),
    ("我最近总想着怎么消失，是不是不太对劲", "high", "流露消失意念"),
    ("他说他最近总想消失，我该怎么劝他", "low", "提及他人不评估自身"),
    ("我有时候觉得活着挺没意思的", "medium", "无意义感无死亡意念"),
    ("朋友说活着没意思，我挺担心他的", "low", "提及他人不评估自身"),
    ("我最近情绪很低落，上课走神", "medium", "情绪低落属功能受损"),
    ("我同桌最近情绪低落，我想帮帮她", "low", "提及他人不评估自身"),
    ("我快被考试压垮了", "medium", "考试压力濒临极限"),
    ("弟弟说他快被考研压垮了，家里人很担心", "low", "提及他人不评估自身"),
    ("我一个人的时候会莫名难过", "medium", "独处低落无自伤意图"),
    ("她说自己一个人的时候会哭，我该怎么陪她", "low", "提及他人不评估自身"),
    ("我最近吃东西没胃口，睡也睡不好", "medium", "躯体化困扰功能受损"),
    ("室友最近没胃口，我们劝她去医院看看", "low", "提及他人不评估自身"),
    ("我觉得自己快到极限了，想缓一缓", "medium", "濒临极限主动调节"),
    ("他说自己到极限了，我帮他约了心理中心", "low", "提及他人不评估自身"),
    ("我心里像压着石头，喘不过气", "medium", "压抑痛苦无自伤意图"),
    ("她说心里压得慌，我陪她聊了一晚上", "low", "提及他人不评估自身"),
    ("我最近不想说话，谁也不想见", "medium", "社交退缩功能受损"),
    ("他把自己关在宿舍不说话，我们有点担心", "low", "提及他人不评估自身"),
    ("我总担心自己状态会崩，提前来咨询", "medium", "担忧崩溃主动求助"),
    ("我担心朋友状态会崩，替他问下咨询预约", "low", "提及他人不评估自身"),
    ("我最近老想着怎么结束这一切，有点吓人", "high", "流露结束一切意念"),
    ("他留言说想结束这一切，我赶紧报告老师了", "low", "提及他人不评估自身"),
    ("我觉得自己是家里的负担，想过消失", "high", "自罪并流露消失意念"),
    ("她觉得自己是负担，我们都在开导她", "low", "提及他人不评估自身"),
]

# ---------- 第五版新增：第三人称/新闻/虚构困难负例（gold=low） ----------
# ---------- 第七版：自我否定家族成对对照 v2（v8 复盘后重构） ----------
# v8 教训：两侧自我否定词完全相同、只靠句尾死亡词区分 -> 模型学到「家族->查 high」。
# v9 设计：最小差异对（死亡词是唯一判别特征，位置前/中/后轮换）；
# 另修正 v8 错标（活着就是浪费粮食/活着还有什么劲 按用户标准归 medium）。
SELF_WORTH_CONTRAST = [
    ("我总觉得自己不配被爱", "medium", "自我否定无死亡意念"),
    ("我总觉得自己不配活着", "high", "不配活着属死亡意念"),
    ("我觉得自己是个拖累，总给人添麻烦", "medium", "自我否定无死亡意念"),
    ("我要是不在了，家里人会轻松很多", "high", "不在了意念属高危"),
    ("我这么没用，什么都帮不上", "medium", "自我否定无死亡意念"),
    ("我这么没用，不如消失算了", "high", "消失意念属高危"),
    ("我就是个废物，什么都做不好", "medium", "自我否定无死亡意念"),
    ("我就是个废物，早该从这个世界消失", "high", "消失意念属高危"),
    ("想到自己这么废物，我很泄气", "medium", "自我否定无死亡意念"),
    ("想到自己这么废物，我不想活了", "high", "死亡意念明确高危"),
    ("我在这个家就是个负担", "medium", "自我否定无死亡意念"),
    ("大家没有我会过得更好", "high", "不存在意念属高危"),
    ("我配不上现在拥有的一切", "medium", "自我否定无死亡意念"),
    ("我配不上这条命", "high", "否定生存属高危"),
    ("这么没用，还净给人添乱", "medium", "自我否定无死亡意念"),
    ("这么没用，不如就不在了", "high", "不在了意念属高危"),
    ("我一事无成，活着就是浪费粮食", "medium", "自我否定无死亡意念"),
    ("我一事无成，真想就此解脱", "high", "解脱意念属高危"),
    ("什么都比不过别人，活着没劲透了", "medium", "无意义感无死亡意念"),
    ("什么都比不过别人，我想彻底消失", "high", "消失意念属高危"),
    ("她说自己是个废物，我们都很心疼她", "low", "提及他人不评估自身"),
    ("他总觉得自己不配被爱，我该怎么开导", "low", "提及他人不评估自身"),
    ("我妈说自己是个拖累，我听了很心疼", "low", "提及他人不评估自身"),
    ("朋友觉得自己配不上这份工作，怎么劝", "low", "提及他人不评估自身"),
]

# ---------- 第七版：「活着+负担」medium 变体族（corp-084 家族，无死亡词） ----------
LIVING_BURDEN_MEDIUM = [
    ("我这么没用，活着净给家里添负担", "愧疚负担感无死亡意念"),
    ("总觉得活着就是在拖累身边的人，很愧疚", "负担感无死亡意念"),
    ("总觉得自己是爸妈的累赘，心里过意不去", "负担感无死亡意念"),
    ("我一事无成，活着只是白占家里的资源", "负担感无死亡意念"),
    ("在家养病这段时间，感觉自己活着就是消耗家人", "负担感无死亡意念"),
    ("成绩垫底还花着家里的钱，我觉得自己很拖累", "负担感无死亡意念"),
    ("什么都做不了还要人照顾，我活得像个累赘", "负担感无死亡意念"),
    ("弟弟妹妹都比我争气，我在这家里显得很多余", "多余感无死亡意念"),
    ("我笨手笨脚的，感觉活着净给人添乱", "负担感无死亡意念"),
    ("这么大了还靠家里养着，我觉得自己是负担", "负担感无死亡意念"),
]

THIRD_PERSON_NEGATIVES = [
    "同学说他不想活了，我该怎么做",
    "刷到有人直播轻生的新闻，平台很快封了直播间",
    "新闻里又报道了学生跳楼的事，看完心里很沉",
    "我表弟确诊了抑郁症，全家都在陪他就医",
    "朋友失恋后说想消失，我陪了他一晚上",
    "追的剧里女主有自伤情节，看得很难受",
    "论文要写青少年自杀干预的方向，正在查资料",
    "课上讲了危机识别，我想多了解一点",
    "家人群里传有人跳楼的视频，我没敢点开",
    "心理委员培训讲了怎么发现同学想不开",
    "室友最近总听很丧的歌，我们有点担心他",
    "朋友说活着累，我约他打球散心",
    "看到热搜上有人轻生获救，希望他好起来",
    "我姐是心理咨询师，回家常聊这些话题",
    "学校心理中心的公众号推了篇干预科普",
    "纪录片里讲了抑郁症患者的故事，挺触动",
    "同学问我对安乐死的看法，纯属课堂讨论",
    "我在帮心理社团整理热线宣传单",
    "朋友让我帮忙查想不开的征兆，他担心他哥",
    "微博上关注了一个讲心理救援的博主",
    "高中有同学意外去世，班级群里都在悼念",
    "亲戚家孩子休学在家，说压力大",
    "我给室友安利了学校心理咨询，他说考虑",
    "班里组织了心理剧，演的是抑郁主题",
    "读了一本讲自杀幸存者的书，挺震撼",
    "朋友做社工常接危机个案，回来会跟我们聊",
    "我在写关于校园心理危机的公众号文章",
    "导师课题做青少年心理危机干预，我打下手",
]


def build_hard_negatives() -> list:
    rows = []
    for text, reason in HARD_NEGATIVE_MEDIUM:
        rows.append({"text": text, "risk_level": "medium", "reason": reason,
                     "_origin": "hard-negative-v4", "fine": "hard_negative"})
    for text, reason in HARD_NEGATIVE_HIGH_CONTRAST:
        rows.append({"text": text, "risk_level": "high", "reason": reason,
                     "_origin": "hard-negative-v4", "fine": "contrast"})
    # 第五版：成对对照与第三人称困难负例。
    for text, level, reason in PAIRED_CONTRAST:
        rows.append({"text": text, "risk_level": level, "reason": reason,
                     "_origin": "paired-contrast-v6", "fine": "paired"})
    # 第六版：自我否定家族成对对照（v9 重构为最小差异对）。
    for text, level, reason in SELF_WORTH_CONTRAST:
        rows.append({"text": text, "risk_level": level, "reason": reason,
                     "_origin": "selfworth-contrast-v8", "fine": "selfworth"})
    # 第七版：「活着+负担」medium 变体族。
    for text, reason in LIVING_BURDEN_MEDIUM:
        rows.append({"text": text, "risk_level": "medium", "reason": reason,
                     "_origin": "living-burden-v9", "fine": "burden"})
    for text in THIRD_PERSON_NEGATIVES:
        rows.append({"text": text, "risk_level": "low",
                     "reason": "提及他人不评估自身",
                     "_origin": "third-person-neg-v6", "fine": "third_person_neg"})
    return rows


def load_consolidated(path: Path) -> list:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            row = json.loads(line)
            row["_line"] = number
            rows.append(row)
    return rows


def relabel(rows: list, log: list) -> list:
    out = []
    for row in rows:
        text = row.get("text") or ""
        label = row.get("risk_level")
        labels_raw = row.get("labels_raw") or []
        third_party = bool(THIRD_PARTY.search(text)) or row.get("speaker_scope") == "third_party_or_fictional"
        has_intent_label = any(l in REASON_BY_FINE_LABEL for l in labels_raw)

        if label == "high" and AGGRESSION_ONLY in labels_raw and not has_intent_label:
            log.append({"id": row["id"], "from": "high", "to": "dropped", "rule": "aggression_only"})
            continue
        if label == "high":
            fine_candidates = [l for l in labels_raw if l in REASON_BY_FINE_LABEL]
            row["_fine"] = next(iter(fine_candidates), None) or (
                "binary_high" if row.get("source") == "suicide_messages" else None)
            # v5：被动自杀意图中「纯无意义/自我否定且无死亡愿望词」降 medium；
            # 含死亡愿望词（不想活/死/消失/解脱等任一）或带其他意念细标签的维持 high。
            if (row["_fine"] == "被动自杀意图" and len(fine_candidates) == 1
                    and MEANINGLESS.search(text) and not DEATH_CUE.search(text)):
                log.append({"id": row["id"], "from": "high", "to": "medium",
                            "rule": "passive_meaninglessness"})
                row["risk_level"] = "medium"
                row["_fine"] = "passive_demote"
        elif label == "low" and _self_ideation(text):
            # 自身意念（含被动死亡意念与他人事件中的自身意念混合）-> high，
            # 不受第三人称语境豁免；纯他人提及已由就近主语裁决排除。
            log.append({"id": row["id"], "from": "low", "to": "high", "rule": "self_ideation"})
            row["risk_level"] = "high"
            row["_fine"] = "explicit_relabel"
        elif label == "low" and not third_party and _selfharm_history(text):
            # 自伤/遗书旧事或康复叙述：有困扰但非当前意念 -> medium。
            log.append({"id": row["id"], "from": "low", "to": "medium", "rule": "selfharm_history"})
            row["risk_level"] = "medium"
            row["_fine"] = "selfharm_history"
        elif label == "low" and not third_party and _self_pain(text):
            log.append({"id": row["id"], "from": "low", "to": "medium", "rule": "pain_without_intent"})
            row["risk_level"] = "medium"
            row["_fine"] = "pain"
        elif label == "low" and not third_party and _self_symptom(text):
            # 仅显著症状/功能受损表述升 medium；轻度（有点失眠/有点焦虑）不匹配，维持 low。
            log.append({"id": row["id"], "from": "low", "to": "medium", "rule": "significant_symptom"})
            row["risk_level"] = "medium"
            row["_fine"] = "symptom"
        elif label == "low" and not third_party and (mine := _mine_medium(text)):
            # 扩量挖掘：情绪痛苦/躯体困扰/人际挫折后痛苦/自我否定/失控感 -> medium。
            log.append({"id": row["id"], "from": "low", "to": "medium", "rule": "distress_mine"})
            row["risk_level"] = "medium"
            row["_fine"] = mine
        elif label == "medium":
            row["_fine"] = "exploration"
        else:
            row["_fine"] = "low"
        row["_origin"] = row.get("source", "unknown")
        out.append(row)
    return out


def rewrite_reason(row: dict) -> str:
    label = row["risk_level"]
    if label == "high":
        if row.get("_origin") in ("hard-negative-v4", "paired-contrast-v6",
                                  "selfworth-contrast-v8"):
            return row["reason"]
        fine = row.get("_fine")
        if fine == "explicit_relabel":
            return "表达自身自杀意念"
        if fine == "binary_high":
            return REASON_HIGH_GENERIC
        return REASON_BY_FINE_LABEL.get(fine, REASON_HIGH_GENERIC)
    if label == "medium":
        if row.get("_origin") in ("hard-negative-v4", "paired-contrast-v6",
                                  "third-person-neg-v6", "campus-medium-v7",
                                  "selfworth-contrast-v8", "living-burden-v9"):
            return row["reason"]
        if row.get("_fine") == "pain":
            return REASON_MEDIUM["pain"]
        if row.get("_fine") == "symptom":
            return REASON_MEDIUM["symptom"]
        if row.get("_fine") == "selfharm_history":
            return REASON_MEDIUM["selfharm_history"]
        if row.get("_fine") == "passive_demote":
            return REASON_MEDIUM["passive_demote"]
        if row.get("_fine") in REASON_MINE:
            return REASON_MINE[row["_fine"]]
        return REASON_MEDIUM["exploration"]
    text = row.get("text") or ""
    if row.get("_origin") in ("paired-contrast-v6", "third-person-neg-v6",
                              "selfworth-contrast-v8"):
        return row["reason"]
    if bool(THIRD_PARTY.search(text)) or row.get("speaker_scope") == "third_party_or_fictional":
        return REASON_LOW_THIRD_PARTY
    return REASON_LOW_VARIANTS[int(_order_key(text), 16) % len(REASON_LOW_VARIANTS)]


def take(pool: list, quota: int, priority=None) -> list:
    """确定性抽样：可选优先级 + 哈希序，截取前 quota 条。"""
    ordered = sorted(pool, key=lambda r: _order_key(r))
    if priority is not None:
        ordered = sorted(ordered, key=priority)  # stable：高优先级整体前置
    return ordered[:quota] if quota < len(ordered) else list(ordered)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build risk_sft_v4 from consolidated_risk_v1")
    configured_project = project_root()
    default_corpus = configured_project_corpus()
    if default_corpus is None and configured_project is not None:
        default_corpus = configured_project / "eval" / "fixtures" / "representative_corpus.json"
    parser.add_argument("--consolidated-root", type=Path,
                        default=TRAIN_ROOT / "training" / "data" / "consolidated_risk_v1")
    parser.add_argument("--corpus", type=Path, default=default_corpus,
                        help="frozen project corpus; set AEGIS_PROJECT_CORPUS or pass explicitly")
    parser.add_argument("--output-root", type=Path, default=TRAIN_ROOT / "data" / "risk_sft_v9")
    parser.add_argument("--train-quota", type=int, nargs=3, default=[1000, 1000, 1000],
                        metavar=("HIGH", "MEDIUM", "LOW"))
    parser.add_argument("--dev-quota", type=int, nargs=3, default=[70, 60, 70],
                        metavar=("HIGH", "MEDIUM", "LOW"))
    parser.add_argument("--max-text-chars", type=int, default=300)
    parser.add_argument("--leak-threshold", type=float, default=0.82)
    parser.add_argument("--medium-upsample-cap", type=float, default=1.35)
    parser.add_argument("--campus-medium", type=Path,
                        default=TRAIN_ROOT / "training" / "data" / "authored" / "campus_medium_v7.jsonl")
    parser.add_argument("--distill-path", type=Path,
                        default=TRAIN_ROOT / "data" / "archive" / "distill_psychology-10k-r1.json")
    parser.add_argument("--distill-limit", type=int, default=200)
    args = parser.parse_args()

    out_root = _guard(args.output_root, (TRAIN_ROOT,), "output")
    consolidated = _guard(args.consolidated_root, (TRAIN_ROOT,), "consolidated")
    if args.corpus is None:
        raise ValueError("--corpus is required unless AEGIS_PROJECT_CORPUS or AEGIS_PROJECT_ROOT is set")
    corpus_roots = (TRAIN_ROOT,)
    if configured_project is not None:
        corpus_roots += (configured_project,)
    corpus_path = _guard(args.corpus, corpus_roots, "corpus")

    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    stress = [r["message"] for r in corpus if r.get("layer") == "stress"]
    if len(stress) != 87:
        raise ValueError(f"expected 87 stress holdout rows, got {len(stress)}")
    stress_keys = {_norm(m) for m in stress}
    stress_grams = [_trigrams(m) for m in stress]

    def leaks(text: str):
        key = _norm(text)
        if key in stress_keys:
            return "exact"
        g = _trigrams(text)
        for idx, hg in enumerate(stress_grams):
            if g and hg and len(g & hg) / len(g | hg) >= args.leak_threshold:
                return f"ngram:{idx}"
        return None

    relabel_log: list = []
    rows = relabel(load_consolidated(consolidated / "train.jsonl"), relabel_log)

    length_dropped = 0
    leakage_rejected: list = []
    kept = []
    for row in rows:
        text = row.get("text") or ""
        if not (4 <= len(text) <= args.max_text_chars):
            length_dropped += 1
            continue
        hit = leaks(text)
        if hit:
            leakage_rejected.append({"id": row["id"], "kind": hit})
            continue
        kept.append(row)

    hard = build_hard_negatives()
    hard_kept = []
    for row in hard:
        hit = leaks(row["text"])
        if hit:
            leakage_rejected.append({"id": f"hard:{row['text'][:12]}", "kind": hit})
            continue
        hard_kept.append(row)

    pools: dict = defaultdict(list)
    for row in kept:
        pools[row["risk_level"]].append(row)
    # 人工样本（困难负例/成对对照/第三人称负例）整组直入 train，不参与抽样：
    # 成对样本一旦被拆散或截断，「主语归属」的对照教学即失效。
    authored_by_level = {
        lv: [r for r in hard_kept if r["risk_level"] == lv]
        for lv in ("high", "medium", "low")
    }

    # 扩量来源：校园合成 medium + 蒸馏集挖掘 medium，同样过泄漏防护后直入 train。
    campus_path = _guard(args.campus_medium, (TRAIN_ROOT,), "campus-medium")
    campus_rows = load_campus_medium(campus_path)
    distill_path = _guard(args.distill_path, (TRAIN_ROOT,), "distill")
    exclude_keys = {_norm(r["text"]) for r in kept} | {
        _norm(r["text"]) for rows_lv in authored_by_level.values() for r in rows_lv}
    distill_rows = mine_distill_medium(distill_path, args.distill_limit, exclude_keys)
    medium_extra = []
    for row in campus_rows + distill_rows:
        hit = leaks(row["text"])
        if hit:
            leakage_rejected.append({"id": f"medium-extra:{row['text'][:12]}", "kind": hit})
            continue
        if _norm(row["text"]) in exclude_keys:
            continue
        exclude_keys.add(_norm(row["text"]))
        medium_extra.append(row)
    authored_by_level["medium"] += medium_extra

    quotas_train = dict(zip(("high", "medium", "low"), args.train_quota))
    quotas_dev = dict(zip(("high", "medium", "low"), args.dev_quota))

    high_priority = lambda r: 0 if r.get("metaphor_flag") else 1
    third_party_low = [r for r in pools["low"]
                       if r.get("speaker_scope") == "third_party_or_fictional"]
    other_low = [r for r in pools["low"]
                 if r.get("speaker_scope") != "third_party_or_fictional"]

    # medium 池稀缺：先为 dev 预留配额，再给 train 抽样，避免训练抽干验证集。
    dev_medium = take(pools["medium"], quotas_dev["medium"])
    dev_medium_ids = {id(r) for r in dev_medium}
    # v9：被动自杀意图且带自我否定字样的子族降密度（<=80），缓解边界高压。
    SW_SURFACE = re.compile(r"(没用|废物|拖累|累赘|多余|不配|一无是处)")
    high_sw = [r for r in pools["high"]
               if r.get("_fine") == "被动自杀意图" and SW_SURFACE.search(r["text"])]
    high_rest = [r for r in pools["high"] if id(r) not in {id(x) for x in high_sw}]
    sw_used = min(len(high_sw), 80)
    train_high = (authored_by_level["high"]
                  + take(high_sw, sw_used)
                  + take(high_rest, quotas_train["high"] - len(authored_by_level["high"]) - sw_used,
                         high_priority))
    train_medium = (authored_by_level["medium"]
                    + take([r for r in pools["medium"] if id(r) not in dev_medium_ids],
                           quotas_train["medium"] - len(authored_by_level["medium"])))
    medium_upsampled = max(0, len(train_medium) - len(pools["medium"]))

    low_tp_share = 0.22  # 第五版：第三人称 low 配额 15% -> 22%
    authored_tp = len(authored_by_level["low"])  # 人工 low 全为第三人称类
    low_tp_quota = int(quotas_train["low"] * low_tp_share)
    train_low = (authored_by_level["low"]
                 + take(third_party_low, max(0, low_tp_quota - authored_tp)))
    train_low += take([r for r in other_low if id(r) not in
                       {id(x) for x in train_low}],
                      quotas_train["low"] - len(train_low))

    used = {id(r) for r in train_high + train_medium + train_low}
    dev_high = take([r for r in pools["high"] if id(r) not in used],
                    quotas_dev["high"], high_priority)
    used |= {id(r) for r in dev_high}
    dev_medium = [r for r in dev_medium if id(r) not in used]
    used |= {id(r) for r in dev_medium}
    dev_tp = take([r for r in third_party_low if id(r) not in used],
                  int(quotas_dev["low"] * low_tp_share))
    dev_low = dev_tp + take([r for r in other_low if id(r) not in used],
                            quotas_dev["low"] - len(dev_tp))

    train = train_high + train_medium + train_low
    dev = dev_high + dev_medium + dev_low

    seen = {_norm(r["text"]) for r in train}
    deduped_dev = []
    for r in dev:
        key = _norm(r["text"])
        if key in seen:
            continue
        seen.add(key)
        deduped_dev.append(r)
    dev = deduped_dev

    train = sorted(train, key=_order_key)
    dev = sorted(dev, key=_order_key)

    def to_sft(row: dict) -> dict:
        reason = rewrite_reason(row)
        if len(reason) > MAX_REASON_CHARS:
            raise ValueError(f"reason exceeds {MAX_REASON_CHARS}: {reason}")
        assistant = json.dumps({"risk_level": row["risk_level"], "reason": reason},
                               ensure_ascii=False)
        return {"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["text"]},
            {"role": "assistant", "content": assistant},
        ]}

    for name, records in (("train", train), ("dev", dev)):
        content = "".join(
            json.dumps(to_sft(row), ensure_ascii=False) + "\n" for row in records)
        _write_text(out_root, f"{name}.jsonl", content)

    review_content = "".join(
        json.dumps({
            "text": row["text"],
            "risk_level": row["risk_level"],
            "reason": rewrite_reason(row),
            "rationale": ("边界困难负例：痛苦/无意义/怕垮表达，无自身意念标记 -> medium"
                          if row["risk_level"] == "medium"
                          else "边界对照：同短语家族 + 明确自身意念标记 -> high"),
        }, ensure_ascii=False) + "\n"
        for row in hard_kept)
    _write_text(out_root, "hard_negatives_v4.review.jsonl", review_content)

    medium_review = "".join(
        json.dumps({"source": row["_origin"], "text": row["text"],
                    "reason": rewrite_reason(row)},
                   ensure_ascii=False) + "\n"
        for row in medium_extra)
    _write_text(out_root, "medium_sources_v7.review.jsonl", medium_review)

    # dev-test 同规则重标（保持训练/测试裁决一致；不抽样，全量保留）。
    test_relabel_log: list = []
    test_rows = relabel(load_consolidated(consolidated / "test.jsonl"), test_relabel_log)
    test_kept = []
    for row in test_rows:
        text = row.get("text") or ""
        if not (4 <= len(text) <= args.max_text_chars):
            continue
        if leaks(text):
            continue
        test_kept.append(row)
    test_content = "".join(
        json.dumps(to_sft(row), ensure_ascii=False) + "\n" for row in test_kept)
    _write_text(out_root, "test.jsonl", test_content)

    def dist(records: list) -> dict:
        c = Counter(r["risk_level"] for r in records)
        return {k: c.get(k, 0) for k in ("high", "medium", "low")}

    manifest = {
        "schema_version": "risk_sft_v9",
        "seed": SEED,
        "system_prompt_version": "v2(方案B:移除不配/活着多余示例)",
        "system_prompt": SYSTEM_PROMPT,
        "sources": {
            "consolidated_train": str(consolidated / "train.jsonl"),
            "frozen_stress_holdout": str(corpus_path),
            "devtest_reference_only": str(consolidated / "test.jsonl"),
        },
        "changes_vs_v3": [
            "第四版：四档裁决重标；移出纯攻击；reason 重写；全 87 条泄漏防护；48 困难负例",
            "第五版(旧v5)：意念表补回；被动纯无意义降 medium；第三人称表扩充",
            "第五版扩量(旧v6取消未训,旧v7即本版)：40 成对对照；28 第三人称负例；第三人称 low 配额 22%",
            "本版新增：low 池痛苦线索挖掘(用户分组裁决)；配额扩至 1000/1000/1000；打/扇/掐自己升 high",
        ],
        "relabel_log": relabel_log,
        "relabel_summary": dict(Counter(f"{e['from']}->{e['to']}" for e in relabel_log)),
        "length_dropped": length_dropped,
        "leakage_rejected": leakage_rejected,
        "train": {"count": len(train), "distribution": dist(train),
                  "origin": dict(Counter(r.get("_origin", "?") for r in train)),
                  "medium_upsampled": medium_upsampled},
        "dev": {"count": len(dev), "distribution": dist(dev)},
        "devtest": {"count": len(test_kept), "distribution": dist(test_kept),
                    "relabel_summary": dict(Counter(
                        f"{e['from']}->{e['to']}" for e in test_relabel_log))},
        "hard_negatives": {"kept": len(hard_kept), "rejected": len(hard) - len(hard_kept)},
        "quotas": {"train": quotas_train, "dev": quotas_dev},
    }
    _write_text(out_root, "manifest.json",
                json.dumps(manifest, ensure_ascii=False, indent=2))

    print("=== RISK_SFT_V4 BUILD DONE ===")
    print(json.dumps({k: manifest[k] for k in
                      ("relabel_summary", "length_dropped", "train", "dev",
                       "hard_negatives")},
                     ensure_ascii=False, indent=2))
    print(f"leakage_rejected: {len(leakage_rejected)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
