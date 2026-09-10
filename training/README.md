# Aegis 风险 QLoRA 训练主手册

> 本文件是 AegisTraining 的 QLoRA 操作主文档，也是本项目的**学习手册**：覆盖背景知识、术语表、环境、基座 gate、数据准备、参数、训练、合并、评测、推理服务、主项目接入、发布、回滚和故障排查，并为每章配备原理讲解、易错点提醒、可直接运行的示例和练习。
> 训练仓库与生产应用必须保持隔离：训练代码和 GPU 依赖留在 AegisTraining，生产 FastAPI 进程只通过受保护的 HTTP/JSON 契约调用已验收模型服务。
>
> 相关文档：
> - 数据格式与标签规则：[training/data/README.md](data/README.md)
> - 数据补充计划：[training/docs/DATA-SUPPLEMENT-PLAN.md](docs/DATA-SUPPLEMENT-PLAN.md)
> - 训练谱系：[reports/TRAINING-HISTORY-INDEX.md](../reports/TRAINING-HISTORY-INDEX.md)
> - 当前 v9 验收：[reports/V9-TRAINING-EVAL-SUMMARY.md](../reports/V9-TRAINING-EVAL-SUMMARY.md)
> - 模型发布模板：[docs/MODEL-RELEASES.md](../docs/MODEL-RELEASES.md)
> - v9 发布记录（当前 release-candidate）：[docs/V9-RELEASE-RECORD.md](../docs/V9-RELEASE-RECORD.md)
> - 生产项目接入说明：[aegis-psych-agent/docs/qlora-finetuning.md](https://github.com/shangguanyunji663/aegis-psych-agent/blob/main/docs/qlora-finetuning.md)

## 目录

- [手册使用指南](#手册使用指南)
- [第 0 章 背景知识：读懂本手册需要的最小知识集](#第-0-章-背景知识读懂本手册需要的最小知识集)
- [术语表（速查）](#术语表速查)
- [核心数据结构速查](#核心数据结构速查)
- [第 1 章 工程边界和安全原则](#第-1-章-工程边界和安全原则)
- [第 2 章 当前推荐版本](#第-2-章-当前推荐版本)
- [第 3 章 目录和路径约定](#第-3-章-目录和路径约定)
- [第 4 章 隔离环境](#第-4-章-隔离环境)
- [第 5 章 基座模型 gate](#第-5-章-基座模型-gate)
- [第 6 章 数据契约和数据构建](#第-6-章-数据契约和数据构建)
- [第 7 章 QLoRA 参数](#第-7-章-qlora-参数)
- [第 8 章 训练流程](#第-8-章-训练流程)
- [第 9 章 v9 验收门槛和结果](#第-9-章-v9-验收门槛和结果)
- [第 10 章 主项目接入契约](#第-10-章-主项目接入契约)
- [第 11 章 发布、校验和回滚](#第-11-章-发布校验和回滚)
- [第 12 章 故障排查](#第-12-章-故障排查)
- [第 13 章 历史版本和不可复现范围](#第-13-章-历史版本和不可复现范围)
- [第 14 章 可复现清单](#第-14-章-可复现清单)
- [第 15 章 入口速查](#第-15-章-入口速查)
- [附录 A 常见问题解答（FAQ）](#附录-a-常见问题解答faq)
- [附录 B 易错点汇总清单](#附录-b-易错点汇总清单)
- [附录 C 分级学习路径与自测清单](#附录-c-分级学习路径与自测清单)
- [附录 D 从零复刻检查表（代码导向）](#附录-d-从零复刻检查表代码导向)

---

## 手册使用指南

### 本手册的四个特性怎么落地

| 特性 | 在本手册中的落地方式 |
| --- | --- |
| 详细 | 第 0 章补齐零基础读者需要的全部背景知识；术语表覆盖全书出现的专业名词；每章不跳步，命令逐条解释输入与预期输出 |
| 深入 | 每个关键操作都配有"为什么这样设计"的原理讲解：不只告诉读者"跑哪条命令"，还解释该命令背后的机制、参数的数学含义与设计取舍 |
| 易上手 | 全书按"由浅入深"组织：先建立全景（第 0 章），再逐章实操（第 1-15 章）；每章末尾配可直接复制运行的示例与练习（练习附参考答案）；不同水平读者按下方学习路径取阅 |
| 易理解 | 章节结构固定为"是什么 → 怎么做 → 为什么 → 易错点 → 练习"；全书术语严格一致（见下方术语约定）；每章有编号、术语表可检索，方便随时回查 |

### 全书通用记号

- 所有命令均为 **Windows `cmd.exe` 语法**：`set` 设置环境变量，`^` 是行尾续行符。PowerShell 用户需把 `^` 换成反引号 `` ` ``、把 `set X=Y` 换成 `$env:X="Y"`。
- `D:\AegisTraining` 是本文档示例采用的训练根目录；请按你的实际 checkout 位置替换。
- `> **易错点**：` 开头的引用块是历史踩坑记录，动手前务必阅读。
- 练习的参考答案折叠在 `<details>` 块中，先自己动手再看答案。
- `§X.Y` 是"第 X.Y 节"的简写，两者等价；源码引用形如 `文件名.py:行号`，行号随重构可能漂移，以函数名为准。

### 三类读者与学习路径

| 读者画像 | 学习目标 | 推荐路径 | 预计用时 |
| --- | --- | --- | --- |
| 零基础：第一次接触大模型微调 | 看懂每一步在做什么，能独立照做并解释原因 | 第 0 章 → 术语表 → 第 1-8 章按顺序（做完全部练习）→ 第 9-15 章通读 → 附录 A/B | 2-3 天 |
| 有机器学习/工程经验 | 快速建立全貌并复现 v9 | 第 0.6-0.8 节 → 第 3-10 章 → 第 14 章可复现清单 → 附录 C 自测 | 0.5-1 天 |
| 运维/复核/审计：不训练模型 | 只关心验收、发布、回滚与证据链 | 第 1、2、9-15 章 + [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) + [SERVICE-RUNBOOK.md](docs/SERVICE-RUNBOOK.md) + 附录 B | 2-3 小时 |

只想"最快跑通一次训练"的快速通道：第 3 章（环境变量）→ 第 4 章（环境自检）→ 第 6.6 节（确认数据存在）→ 第 8.1-8.2 节（dry-run 后训练）。但出问题时请回到对应章节的"易错点"和第 12 章排障表。

### 章节依赖地图

```text
第 0 章 背景知识 ──────────────┐
                              ▼
第 1 章 工程边界 ─► 第 3 章 路径 ─► 第 4 章 环境 ─► 第 5 章 基座 gate
                                                      │
第 2 章 版本速览（随时查阅）                            ▼
                                          第 6 章 数据契约与构建
                                                      │
                                                      ▼
                                          第 7 章 QLoRA 参数 ─► 第 8 章 训练流程
                                                                      │
                        第 10 章 主项目接入 ◄─ 第 9 章 验收门槛 ◄───────┘
                                │                  │
                                ▼                  ▼
                        第 11 章 发布与回滚 ◄─ 第 12 章 故障排查（随时查阅）
                                           第 13/14/15 章（审计与速查，随时查阅）
```

### 术语约定

全书严格遵守以下一致性，检索（Ctrl+F）任意一处即可定位所有相关内容：

- **基座模型**：被微调的原始模型（Qwen3.5-2B-Base），不与"底模""母模型"混用。
- **adapter**：LoRA 训练产出的小权重文件，不翻译为"适配器"。
- **合并（merge）**：把 adapter 并入基座生成完整模型文件的动作。
- **冻结 holdout（stress 87）**：永久不参与训练的最终验收集，与 dev/devtest 严格区分。
- **验收门槛**：上线前必须全部满足的量化指标；"评测"指跑出指标，"验收"指对照门槛下结论。
- **泄漏（leakage）**：评测数据以任何形式进入训练数据；**回退（fallback）**：模型通道失败时退回规则通道。
- **显存**：GPU 上的显存（VRAM），与内存（RAM）区分。

---

## 第 0 章 背景知识：读懂本手册需要的最小知识集

> 本章面向零基础读者，建立理解后续所有章节所需的概念框架。目标：读完本章后，你看到"QLoRA 训练风险分类器"这句话时，能说清每个词的含义和它们为什么被组合在一起。已熟悉 LoRA/QLoRA 的读者可跳到 0.7、0.8 两节（它们解释本项目特有的设计）。

### 0.1 项目全景：两套系统，一条边界

本项目要解决的问题：校园心理支持系统每天会收到大量用户消息，其中极少数可能隐藏自伤/自杀风险。规则引擎（人工编写的关键词和逻辑）能抓住一部分，但对**隐喻式表达**（如"想消失""要是没出生过就好了"）容易漏报。本项目用一个微调过的大语言模型作为规则的**增强器**：模型读入一条消息，输出一个 JSON 风险等级，与规则结果融合后交给业务链路。

整套系统由两个仓库组成，边界必须保持：

```text
┌─────────────────────────────────────────┐      ┌──────────────────────────────────┐
│ AegisTraining（训练仓库，本手册）          │      │ aegis-psych-agent（生产仓库）      │
│                                         │      │                                  │
│  经审查的数据 ─► 数据契约校验 ─► 泄漏检查   │      │  FastAPI 应用 / Agent / RAG       │
│        ─► 4-bit NF4 QLoRA 训练           │ HTTP │  工具治理 / 审批 / 回复生成         │
│        ─► adapter 合并 ─► 冻结集评测      │ ───► │  只通过 POST /assess 调用模型服务   │
│        ─► 版本化发布或回滚                │ JSON │  规则通道永久保留，异常时回退        │
└─────────────────────────────────────────┘      └──────────────────────────────────┘
```

**为什么这样设计**：训练依赖（PyTorch、CUDA、Transformers、bitsandbytes）体积巨大且版本敏感，生产 FastAPI 进程一旦混入这些依赖，部署体积、启动时间和故障面都会失控；更关键的是，训练中的模型是**未经验收的实验资产**，绝不能与生产服务共存于一个进程。隔离边界保证：生产永远只调用"已通过验收的模型"暴露的稳定 HTTP 契约。

### 0.2 大语言模型是怎么"说话"的

- **token（词元）**：模型处理文本的最小单位，可能是半个词、一个词或一个标点。"我想消失"可能被切成 4-6 个 token。模型的世界里只有 token 序列和它们的编号。
- **下一个词预测**：大语言模型的本质是一个超大的"接龙"函数：给定前面的 token，预测下一个 token 的概率分布。"今天天气真"后面，"好"的概率很高。
- **因果语言模型（causal LM）**：只允许每个位置关注它**之前**的 token（不能偷看后文）的语言模型。Qwen3.5 就是这类模型，这也是训练配置里 `task_type: CAUSAL_LM` 的含义。
- **基座模型（base model）**：只做过"海量文本接龙预训练"的原始模型，比如本项目的 `Qwen/Qwen3.5-2B-Base`（约 18.9 亿参数、24 层、隐藏维度 2048）。它知识渊博，但**没被教过任何具体任务的输出格式**——你问它"这条消息风险多高"，它可能续写出一篇讨论风险的文章，而不是你想要的 JSON。
- **微调（fine-tuning）**：在基座上用少量任务数据继续训练，把"什么都会一点"变成"精确做好一件事"。

**为什么选 2B 而不是更大的模型**：本项目显卡是 RTX 4060 Laptop（约 8GB 显存）。模型越大，量化后占的显存越多；2B 是 8GB 显存 + 512 token 上下文下能稳定完成训练-评测闭环的规模（实测峰值约 4.9GB，见第 8 章）。

### 0.3 监督微调（SFT）：用例子教会模型任务

**监督微调（Supervised Fine-Tuning, SFT）**：给模型看大量"(输入，标准答案)"对，让它学会"看到这类输入就产生这类输出"。

本项目的每条 SFT 样本是一个三段对话（来自 `data_contract.py` 的 `to_sft_record`）：

```json
{
  "messages": [
    {"role": "system", "content": "你是校园心理支持系统的风险评估器,……只输出一个 JSON 对象……"},
    {"role": "user", "content": "我最近觉得自己很多余。"},
    {"role": "assistant", "content": "{\"risk_level\":\"medium\",\"reason\":\"明显痛苦但无自伤意向\"}"}
  ]
}
```

三个角色各司其职：**system** 是任务说明书（永远固定，即第 6.3 节的提示词契约）；**user** 是被评估的消息（训练数据里的 `message` 字段）；**assistant** 是标准答案（风险等级 JSON）。模型在训练中学到的核心能力是：在 system 说明书约束下，把任意 user 消息压缩成一个符合格式的 JSON 判断。

### 0.4 全参数微调的困境与 LoRA 的思路

**困境**：最朴素的微调是更新模型全部 18.9 亿参数。但这意味着显存里除了模型本身，还要为每个参数存放梯度（1 份）和优化器状态（AdamW 需要一阶、二阶动量，2 份）。2B 模型全参数微调在 8GB 显卡上必然显存溢出（OOM）。此外，全参数微调会大幅改写基座权重，容易把预训练学到的通用能力"洗掉"（灾难性遗忘）。

**LoRA（Low-Rank Adaptation）**的思路：微调对权重的改动量 ΔW 其实是"低秩"的——教一个模型做风险分类，并不需要改动它 18.9 亿个参数中的每一个，需要的"新知识"维度远小于模型本身的维度。于是：

1. **冻结**原有权重 W（不训练、不存梯度）；
2. 在需要改动的层旁挂一对小矩阵：ΔW = B×A，其中 A 是 r×d、B 是 d×r，秩 r 远小于模型维度 d（本项目 r=8）；
3. 只训练 A 和 B。

**数字对比**：本项目可训练参数 8,409,600，占总量 1,890,234,688 的 **0.4449%**（第 8 章 dry-run 会实测这两个数）。训练显存从"放不下"降到"峰值约 4.9GB"，而且 adapter 文件只有几十 MB，便于版本管理和审计。

三个 LoRA 超参数的直觉（第 7 章有完整解释）：

- **r（秩）**：小矩阵的"容量"。r 越大表达能力越强，但也越容易过拟合、越占显存。
- **alpha（缩放）**：LoRA 分支输出会乘以 alpha/r。本项目 alpha=16、r=8，等效把新学到的改动放大 2 倍。
- **dropout**：训练时随机丢弃 LoRA 分支的一部分神经元，抑制过拟合（本项目 0.05）。

### 0.5 量化：用更少的比特保存权重

**量化（quantization）**：用更少的比特表示每个权重数值。比特数减半，存储近似减半：

| 精度 | 每参数比特数 | 2B 模型权重体积（约） | 特点 |
| --- | ---: | ---: | --- |
| FP32 | 32 | 7.6 GB | 训练原始精度，本机用不到 |
| BF16 / FP16 | 16 | 3.8 GB | 合并与推理采用（BF16，见 4.5 节） |
| INT8 | 8 | 1.9 GB | Ollama 本地模型常用 |
| **NF4** | **4** | **约 1 GB** | **QLoRA 训练基座的存储精度** |

**NF4（NormalFloat4）**是 QLoRA 论文提出的 4-bit 数据类型。它的关键观察：神经网络权重分布近似正态分布（大部分值集中在 0 附近），NF4 按正态分布的分位数设计 4-bit 的取值格子，让量化误差在统计上最小——比普通 INT4 更适合权重。

**二重量化（double quantization）**：4-bit 量化每组需要一个缩放常数，这些常数本身也是 32-bit 存的，体积可观。把常数再量化一次（用 8-bit 存），能再省下约 0.4 bit/参数的额外开销。对应配置 `bnb_4bit_use_double_quant: true`。

**存储精度 ≠ 计算精度**：4-bit 只决定权重**怎么存**；真正做矩阵乘法时，权重先反量化回 BF16 再计算（配置 `bnb_4bit_compute_dtype: bfloat16`）。所以 QLoRA 是"4-bit 存、16-bit 算"，两者不矛盾。

### 0.6 QLoRA：把 0.4 和 0.5 拼起来

**QLoRA = 4-bit 量化冻结基座 + 在其上训练 BF16 的 LoRA adapter**。组合后的显存账（量级估算，非精确值）：

```text
基座 18.9 亿参数 × 4 bit          ≈ 1.0 GB   （NF4 + 二重量化）
LoRA 8.4M 参数 BF16 权重+梯度+状态  ≈ 0.1 GB
优化器分页/激活/上下文/碎片         ≈ 3-4 GB
─────────────────────────────────────────
实测训练峰值 CUDA 显存             ≈ 4.9 GB   （8GB 显卡稳定完成）
```

**为什么这样设计**：单独 LoRA（不量化基座）时，仅基座 BF16 权重就要 3.8GB，加上激活与碎片，8GB 显卡几乎没有余量；单独量化（不做 LoRA）没有解决"训练时优化器状态爆炸"的问题。QLoRA 让"冻结的大模型用 4-bit 省","训练的小模块用 16-bit 稳"，两者各取所长。配合**梯度检查点**（用重算换显存，7.4 节）和**分页优化器**（优化器状态显存不足时自动搬到内存，7.4 节），2B 模型在笔记本显卡上完成训练成为可能。

### 0.7 任务设计：为什么输出是一个 JSON

模型被训练成只输出 `{"risk_level": "low|medium|high", "reason": "20字以内依据"}`，这是有意为之的工程决策：

1. **机器可解析**：纯文本判断（"我觉得这条消息挺危险的"）无法被程序消费；JSON 可以直接进入业务逻辑。
2. **可校验、可量化**：格式固定才能计算"JSON 有效率""合法标签率"这类验收指标（第 9 章）；reason 限 20 字防止模型输出长篇解释拖垮延迟。
3. **可回退**：解析失败时服务返回 `risk_level: null`，调用方明确地回退规则通道——"失败要显式，不要伪装成成功"（第 10 章）。
4. **约束即安全**：只允许三个离散标签，杜绝模型自由发挥给出模糊表述；融合层用 `max(规则, 模型)` 保证模型**只能升高风险、不能降低风险**（第 10.2 节）。

### 0.8 数据划分与"泛化"：四个集合各司其职

**泛化（generalization）**：模型在**没见过**的数据上依然做对的能力。衡量泛化的唯一方法是留出一部分数据不参与训练，只用它们考试。本项目有四类数据集合，用考试打个比方：

| 集合 | 规模 | 比喻 | 参与训练？ | 用途 |
| --- | ---: | --- | --- | --- |
| train | 2867 | 平时上课的教材与习题 | ✅ | 拟合 |
| dev | 200 | 模拟考 | ❌ | 每个 epoch 算 eval_loss，选最优 checkpoint、决定早停 |
| devtest（test.jsonl） | 1414 | 随堂大测 | ❌ | 开发期参考评测（第 8.6 节），不作为上线门槛 |
| 冻结 holdout（stress 87） | 87 | 高考密封卷 | ❌ 永久 | 最终验收门槛（第 9 章） |

**为什么 stress 必须永久冻结**：如果它的任何一条样本混进训练数据，模型等于"背过原题"，评测分数会虚高而真实能力不变——这叫**数据泄漏（data leakage）**。所以第 6.4 节的泄漏防护在构建数据时就会把与 stress 精确或近重复（相似度 ≥ 0.82）的候选全部拒绝。

**为什么 devtest 通过了还不能上线**：devtest 与训练数据出自同一构建流程（同规则重标），分布接近训练集；stress 87 条则包含隐喻、第三人称干扰等**专门设计的困难场景**。随堂测考得好，不代表密封卷考得好——第 13 章的历史教训（第二、三、五、六版都倒在 stress 的某个门槛上）反复验证了这一点。

### 0.9 本章练习

**练习 0.1** 用自己的话解释：为什么本项目不用全参数微调？（至少两个理由）

**练习 0.2** 已知可训练参数 8,409,600、总参数 1,890,234,688，不查文档估算可训练占比。

**练习 0.3** 一条用户消息"新闻里报道了一起轻生事件"应被评估为 high 吗？结合 0.7 节和 0.8 节说明判断依据应来自哪里。

<details>
<summary>参考答案</summary>

0.1 ①显存：全参数微调需为 18.9 亿参数额外存梯度和优化器状态，8GB 显卡放不下；②稳定性：全参数改写易灾难性遗忘，LoRA 只学 0.44% 的增量，基座能力保留；③工程：adapter 文件小，便于版本化、审计和回滚。

0.2 8,409,600 ÷ 1,890,234,688 ≈ 0.004449，即约 0.4449%（与第 2 章表格一致）。

0.3 不应。判定原则是"只评估**说话人自身**"：新闻里的轻生是第三人称/他人语境，按提示词契约应判 low。判断依据应来自提示词契约（第 6.3 节）与训练数据的标签原则（第 6.1.4 节），而不是"出现了高危词就升 high"。
</details>

---

## 术语表（速查）

按主题分组，全书用法与下表严格一致。遇到陌生名词先查这里。

### 模型与训练

| 术语 | 含义 |
| --- | --- |
| 基座模型（base model） | 被微调的原始模型；本项目固定为官方 Qwen3.5-2B-Base 的指定 revision 快照 |
| 因果语言模型（causal LM） | 只依据前文预测下一个 token 的模型结构 |
| token / 词元 | 模型处理文本的最小单位 |
| SFT（监督微调） | 用 (输入, 标准答案) 对训练模型执行特定任务 |
| LoRA | 冻结基座、只训练低秩增量矩阵 ΔW=BA 的微调方法 |
| QLoRA | 基座以 4-bit NF4 量化加载，其上训练 BF16 LoRA adapter 的方法 |
| NF4（NormalFloat4） | 按正态分位数设计的 4-bit 量化类型，对权重分布误差最小 |
| 二重量化 | 对量化缩放常数再做一次量化以进一步省显存 |
| compute dtype | 实际矩阵计算精度；本项目 BF16（不支持时回退 FP16） |
| rank（r）/ alpha | LoRA 低秩矩阵的秩 / 缩放系数（等效增益 alpha/r） |
| adapter | LoRA 训练产出的小权重文件，仅几 MB，可独立保存与审计 |
| 合并（merge） | 将 adapter 并入基座，导出完整 BF16 safetensors 模型 |
| 梯度检查点（gradient checkpointing） | 前向不保存全部激活、反向时重算，用约 20-30% 时间换大幅显存 |
| 分页优化器（paged_adamw_8bit） | 8-bit AdamW，状态显存不足时自动在内存与显存间分页 |
| warmup | 训练初期用较小学习率"热身"，随后按 cosine 曲线衰减 |
| epoch | 全部训练数据完整过一遍；本项目 3 个 epoch，每个 epoch 评估并保存 |
| eval_loss | 模型在 dev 集上的损失；越低越好，用于选最优 checkpoint |
| 过拟合 | 训练集表现继续变好但 dev 表现变差；表现为 eval_loss 回升 |
| 早停（early stopping） | eval_loss 连续 2 个 epoch 无改善则停止训练 |
| OOM | 显存溢出（Out Of Memory） |
| BF16 / FP16 | 两种 16-bit 浮点：BF16 指数位多、动态范围大；FP16 尾数多、范围小需 loss scaling |

### 数据与评测

| 术语 | 含义 |
| --- | --- |
| 数据契约 | 原始 JSONL 必须满足的字段与取值约束（第 6.1 节） |
| 提示词契约（prompt contract v2） | system prompt 的唯一合法原文；训练/评测/服务使用同一文本（第 6.3 节） |
| train / dev / devtest / 冻结 holdout | 2867 训练 / 200 模型选择 / 1414 开发参考 / 87 最终验收（第 0.8 节） |
| stress 87 | 冻结 holdout 的别称，来自项目 fixture 的 stress 层，永久不入训练 |
| 泄漏（leakage） | 评测数据进入训练数据，导致评测虚高；用精确哈希 + 3-gram 相似度防护 |
| 3-gram Jaccard | 相似度 = 字符三元组集合交集/并集；≥0.82 视为近重复 |
| hard negative | 人工编写的困难负例：长得像高危但实为 medium/low 的样本 |
| third_person / fictional | 第三人称（谈论他人）与虚构（影视/论文）语境；高危词不升自身风险 |
| JSON 有效率 | 模型输出能被解析为合法 JSON 的比例；门槛 ≥98% |
| 合法标签率 | risk_level 落在 low/medium/high 的比例；门槛 ≥99% |
| accuracy / macro-F1 | 总体正确率 / 三个类别 F1 的平均（不偏袒大类） |
| recall（召回） | 某类真实样本中被正确找出的比例；high recall 是安全生命线 |
| FPR（误升级率） | non-high 样本被判为 high 的比例；增幅门槛 ≤2pp |
| P95 延迟 | 95% 的请求快于此延迟；门槛 ≤8s |
| 混淆矩阵（confusion） | devtest 报告中 gold×pred 的计数表，用于定位误判模式 |
| rules ∪ QLoRA | 融合结果：取规则与模型预测中较高的那个风险等级（第 10.2 节） |

### 工程与发布

| 术语 | 含义 |
| --- | --- |
| gate | 训练/合并前对本地基座快照的结构与来源检查（第 5 章） |
| revision | Hugging Face 仓库某次提交的固定标识；钉死以杜绝"同名不同内容" |
| safetensors | 安全、可校验的权重文件格式；训练只接受此格式 |
| GGUF | Ollama 使用的推理工件格式；**不可**作为训练基座 |
| manifest | 记录一次构建/训练/导出全部元数据的 JSON；审计依据 |
| SHA-256 | 文件内容哈希；用于锁定配置、数据与权重的精确版本 |
| 回退（fallback） | 模型通道失败时自动退回规则通道；fail-safe 设计 |
| release-candidate | 通过全部验收门槛、待审批发布的模型状态 |
| research-only | 未通过验收的模型状态；禁止接入生产 |
| 回滚（rollback） | 关闭 QLoRA 开关并恢复上一版已批准模型或纯规则通道 |
| 路径守卫 | 所有脚本强制输入/输出位于训练根目录内的安全机制（第 3.2 节） |
| AEGIS_TRAINING_ROOT | 训练根目录环境变量；路径守卫的判定基准 |

---

## 核心数据结构速查

四个 dataclass 贯穿全流程，三类 manifest 是三大审计锚点。行号以当前代码为准。

### RiskSample（`data_contract.py:31`）——原始契约样本

| 成员 | 说明 |
| --- | --- |
| `sample_id/message/risk_level/reason/source` | 必填五件套（约束见 §6.1.2） |
| `source_version/label_method/review_status/annotator/speaker_scope/adjudication_note` | 溯源元数据（如实记录，见 §6.1.3） |
| `normalized_message` / `message_hash`（property） | NFKC+去空白+小写规范化文本 / 其 SHA-256——泄漏检测的统一键 |
| `to_sft_record()` | 转成三段 messages（system=v2 契约） |

### Prediction（`metrics.py:13`）——评测的单条预测

| 成员 | 说明 |
| --- | --- |
| `sample_id/expected` | 样本与金标 |
| `predicted` | 解析后的标签；`None` = JSON 解析失败（指标层的"无答案"） |
| `raw_output/json_valid/reason/latency_ms` | 模型原文、格式有效性、依据、延迟 |
| `category/layer` | 分层口径（`suicidal_implicit`/`suicidal_explicit`/`third_person`；`base`/`stress`） |

### GateReport（`base_model_gate.py:28`）——基座资格报告

11 个字段：`repo_id/revision/model_type/architecture/hidden_size/text_layers/trainable_safetensors/family_scale_match/ollama_conversion_provenance/status/notes`；`status` 两级语义见 §5.2。它会原样进入 training-manifest 的 `gate` 字段。

### LeakageMatch（`leakage_guard.py:16`）——泄漏命中

`sample_id/source_path/kind(exact|near_duplicate)/score/preview`——每条被拒样本的"判决书"。

### SFT messages（`data_contract.to_sft_record` / prepare `to_sft`，L793）

```json
{"messages": [
  {"role": "system", "content": "<v2 契约全文>"},
  {"role": "user", "content": "<message>"},
  {"role": "assistant", "content": "{\"risk_level\":\"…\",\"reason\":\"…\"}"}
]}
```

assistant 段即训练目标——§7.5 的 `-100` 掩码只对这一段计算 loss。

### 三类 manifest 对照：三大审计锚点

| manifest | 产出者 | 关键字段 | 回答的审计问题 |
| --- | --- | --- | --- |
| `data/risk_sft_v9/manifest.json` | prepare v4（L848，字段表见 §6.7.8） | seed/system_prompt/sources/relabel_*/leakage_rejected/train.origin/quotas | 用了什么数据、怎么裁决 |
| `checkpoints/.../training-manifest.json` | train 脚本（L269） | gate(=GateReport)/cuda_device/compute_dtype/target_modules/trainable_*/train_rows/peak_cuda_memory_bytes | 什么环境、什么结构、训了多少 |
| `exports/.../aegis-export-manifest.json` | merge 脚本（L69） | official_base/adapter_dir/output_dir/dtype/merged_model_class/max_shard_size | 从哪个 adapter 合出、什么精度 |

---
## 第 1 章 工程边界和安全原则

### 1.1 训练生命周期总览

AegisTraining 只负责风险识别模型的训练生命周期：

```text
经审查的数据
  -> 数据契约校验与重标
  -> 最终 holdout 泄漏检查
  -> 4-bit NF4 QLoRA 训练
  -> adapter 合并
  -> 冻结集和外部集评测
  -> 版本化发布或回滚
```

生产应用 `aegis-psych-agent` 负责 Agent、RAG、工具治理、审批和回复生成，不在 FastAPI 进程中加载 Transformers、PEFT、bitsandbytes 或训练 adapter。

### 1.2 为什么流程必须这样串联

上面七个环节的顺序不是任意的，每一环都是前一环的质量前提：

1. **先契约校验、再训练**——格式错误的样本会在训练中途崩溃，或更糟：以错误格式被静默学进去。契约校验（第 6 章）把失败拦在最便宜的阶段。
2. **先泄漏检查、再训练**——泄漏一旦发生，训练已不可撤销，评测结果全部作废；把检查放在数据构建阶段，泄漏样本根本进不了 train.jsonl。
3. **先训练、再合并**——adapter 是"可审计的增量"：保留它就能追溯"这次到底改了什么"，也能在合并前反复评测。直接训练出全量模型再想审计就难了。
4. **先评测、再发布**——第 9 章的八项门槛全部通过之前，模型在制度上只是 research-only 资产。

### 1.3 六条红线逐条解读

必须遵守：

- **不把训练依赖追加到生产项目的 `requirements.txt`。**
  为什么：生产镜像会被 torch/CUDA 拖大数 GB，且训练依赖更新频繁，会把生产部署拖入"为训练改动而重新部署"的耦合。
- **不把模型权重、HF cache、checkpoint、merged 权重、GGUF、日志或原始敏感数据提交到 Git。**
  为什么：单个基座约 4.5GB，Git 不是文件存储；心理健康原始数据入 Git 等于不可撤销地外泄（历史无法干净删除）。
- **不把 Ollama 的 Q8 GGUF 推理工件当作 Transformers/PEFT 的训练基座。**
  为什么：GGUF 是为推理重新打包的工件（可能量化、可能转换工具不同），无法证明与官方 revision 同源；用它训练等于在未知底座上盖楼。详见第 5.3 节。
- **规则风险通道永久保留；QLoRA 只能升级风险，不能降低规则风险。**
  为什么：模型可能宕机、超时、输出非法。规则通道是"最后一道保险丝"；融合采用 `max(规则, 模型)`（第 10.2 节），保证任何异常都不会把风险等级**降**下来——漏报高危的代价远高于误报。
- **未通过冻结验收、外部审阅和发布清单的模型只能作为研究资产。**
  为什么：评测分数好 ≠ 可以对真实用户使用。第 9 章的门槛只是工程底线，上线还需外部专家审阅（第 9、13 章）。
- **心理健康和自杀风险数据必须完成授权、脱敏、许可证、用途和公开范围审查。**
  为什么：法律与伦理双重约束；未审查数据一旦扩散不可撤回。

> **易错点**：最常见的越界冲动是"临时调试方便"——把 `torch` 加进生产 requirements、把 checkpoint 拷进 Git LFS、把本机 8301 服务写进生产 `.env`。三者都违反红线。临时产物请放在训练根目录的被忽略目录（`checkpoints/`、`exports/`、`hf-cache/`），本机服务只做 smoke test（第 8.7 节）。

### 1.4 练习

**练习 1.1** 判断下列做法各违反哪条红线：(a) 为了让同事复现，把 merged 模型上传到仓库 Release 之外的代码分支；(b) 生产项目里 `pip install bitsandbytes` 以便"顺便跑一下评测"；(c) 看到 QLoRA 把某条规则判 high 的消息改判 medium，决定直接采信模型。
<details>
<summary>参考答案</summary>
(a) 违反"权重不入 Git/仓库"，发布必须走第 11 章的受控渠道并补全元数据；(b) 违反"训练依赖不入生产项目"；(c) 违反"QLoRA 只能升级风险"，融合层必须取 max，规则 high 不能被任何模型输出压低。
</details>

---

## 第 2 章 当前推荐版本

### 2.1 版本速览

| 项目 | 当前值 |
| --- | --- |
| 模型候选 | `aegis-risk-qwen3.5-2b-v9` |
| 数据集 | `risk_sft_v9` |
| 提示词契约 | `v2` |
| 基座 | `Qwen/Qwen3.5-2B-Base` |
| 固定 revision | `b1485b2fa6dfa1287294f269f5fb618e03d52d7c` |
| GPU 对照 | RTX 4060 Laptop，约 8GB 显存 |
| 训练方式 | 纯文本 4-bit NF4 QLoRA |
| 实际训练规模 | 2867 train / 200 dev / 1414 devtest |
| 实测训练 | 540 steps，约 2h54m，峰值显存约 4.9GB |
| 最优 checkpoint | epoch 2，step 360，按 `eval_loss` 选择 |
| 可训练参数 | 8,409,600 / 1,890,234,688，约 0.4449% |
| 冻结最终 holdout | `representative_corpus.json` 的 stress 87 条 |
| v9 结论 | 八项冻结验收门槛全部通过，可作为 release candidate；生产接入仍需显式审批 |

### 2.2 表格逐行解读

- **模型候选 / 数据集 / 提示词契约**：三者是一个整体——"第七版（v9）模型 = risk_sft_v9 数据 × 提示词 v2 × 当前参数"。换任何一个都必须重训并重新验收，旧数字不能拼给新组合（第 13 章）。
- **固定 revision**：模型的"版本号中的版本号"。Hugging Face 仓库内容会更新，revision 锁定到一次具体提交，保证你下载的 4.5GB 文件和验收时的逐字节一致（哈希核对见第 11 章）。
- **实际训练规模**：2867/200/1414 的来历与三个集合的分工见第 0.8 节和第 6.2 节。
- **实测训练**：RTX 4060 Laptop 8GB 上的真实记录。如果你在同档显卡上训练时间或显存显著偏离（如 >5h 或 OOM），先查第 12 章排障表，不要带病继续。
- **最优 checkpoint epoch 2 / step 360**：3 个 epoch 的 eval_loss 是 0.0818 → 0.0638 → 0.0654，第 3 个 epoch 回升（过拟合迹象），脚本按 `metric_for_best_model: eval_loss` 自动回选 epoch 2（机制见第 7.4、8.3 节）。
- **v9 结论**：`release-candidate` ≠ 已上线。生产接入需要显式审批并完成第 10、11 章的接入与发布动作。

### 2.3 版本编号遗留问题

版本编号有历史遗留：磁盘目录可能出现 `v3`、`v5`、`v7`、`v8` 等旧编号，正式谱系以 [TRAINING-HISTORY-INDEX.md](../reports/TRAINING-HISTORY-INDEX.md) 的"第一版～第七版"映射为准（旧 v1=第一版、旧 v3=第二版、旧 v4=第三版、旧 v5=第四版、旧 v7=第五版、旧 v8=第六版、旧 v9=第七版；"v2"编号作废，历史上并无独立 v2 训练）。

> **易错点**：看到磁盘目录名 `checkpoints/aegis-risk-qwen3.5-2b-v5` 就当成"第五版"是错的——它是**第四版**。引用历史结论时一律以谱系索引的版次为准，不要以目录名推断。

---

## 第 3 章 目录和路径约定

### 3.1 目录树

```text
AegisTraining/
├── training/
│   ├── configs/risk_qlora_4060.yaml       # 当前 4060/8GB QLoRA 参数
│   ├── scripts/                           # prepare/train/merge/eval/serve 入口
│   ├── src/aegis_training/                 # 数据契约、gate、泄漏、指标、路径工具
│   ├── data/consolidated_risk_v1/          # 经审查的当前数据来源
│   ├── data/authored/                      # 受控人工困难样本
│   └── requirements-qlora.txt              # 独立 GPU 训练依赖
├── models/Qwen3.5-2B-Base/                # 本地官方 safetensors 基座，不进 Git
├── data/risk_sft_v9/                       # 生成的 train/dev/test，不进 Git
├── checkpoints/...                         # PEFT adapter 和 epoch checkpoint，不进 Git
├── exports/...                             # 合并后 safetensors，不进 Git
├── reports/                                # 轻量验收摘要和谱系记录
├── docs/MODEL-RELEASES.md                  # 发布元数据模板
└── envs/qlora-qwen35/                      # 隔离 Python 环境
```

三类目录的性质不同：`training/` 下的代码与配置进 Git（可审计）；`models/ data/ checkpoints/ exports/ envs/` 是本地工件（被 .gitignore 忽略，体积大或敏感）；`reports/ docs/` 只保留轻量的、经审查的结论性文档。

### 3.2 路径守卫：脚本为什么"拒绝越界"

脚本通过 `AEGIS_TRAINING_ROOT` 解析训练根目录，默认是当前 checkout；模型、数据、checkpoint 和导出文件都必须位于训练根目录允许的路径下。路径检查会拒绝越界路径，避免把输出写到未授权目录。

**原理**（`training/src/aegis_training/paths.py`）：每个脚本在读写任何路径前都会调用 `under(path, root, role)`——它把路径解析成绝对路径后，用 `relative_to(root)` 判断是否落在根目录之内，不在就抛出 `path escapes allowed root`。这是一层**防呆+安全**设计：

- 防呆：手滑把 `--output-root` 写成别的项目目录时，脚本立即报错而不是把半成品写到别处；
- 安全：防止配置或参数被注入后把敏感输出（如带训练数据的中间文件）写到仓库之外。

**可直接运行的示例**（体验守卫的放行与拒绝，不依赖 GPU）：

```bat
D:\AegisTraining\envs\qlora-qwen35\Scripts\python.exe -c "import sys; sys.path.insert(0, r'D:\AegisTraining\training\src'); from pathlib import Path; from aegis_training.paths import under; print(under(Path(r'D:\AegisTraining\data\x.json'), Path(r'D:\AegisTraining'), 'demo'))"

D:\AegisTraining\envs\qlora-qwen35\Scripts\python.exe -c "import sys; sys.path.insert(0, r'D:\AegisTraining\training\src'); from pathlib import Path; from aegis_training.paths import under; under(Path(r'C:\Temp\x.json'), Path(r'D:\AegisTraining'), 'demo')"
```

第一条打印放行后的路径；第二条应看到 `ValueError: demo path escapes allowed root: ...`——这就是守卫生效的样子。

### 3.3 环境变量设置

Windows `cmd.exe` 示例：

```bat
set AEGIS_TRAINING_ROOT=D:\AegisTraining
set AEGIS_PROJECT_ROOT=D:\PythonProject\aegis-psych-agent
set AEGIS_PROJECT_CORPUS=%AEGIS_PROJECT_ROOT%\eval\fixtures\representative_corpus.json
set HF_HOME=%AEGIS_TRAINING_ROOT%\hf-cache
```

四个变量各管一件事：训练根（路径守卫基准）、生产仓库（只读其已提交的 fixture）、冻结语料的显式路径、HF 下载缓存（把几 GB 的缓存留在训练根内，不污染 C 盘用户目录）。

如果两个仓库不在同一台机器上，应把已审查、已版本化的 fixture 作为明确输入提供给训练流程，不要把本机绝对路径硬编码进数据文件或 manifest。

> **易错点**：`set` 只对当前 cmd 窗口有效，新开窗口要重设；忘设 `AEGIS_PROJECT_CORPUS` 时构建脚本会直接报 `--corpus is required`，这是设计行为（第 6.6.2 节），不要为绕过它指向工作区里未提交的语料副本。

### 3.4 模块地图与调用链：读代码前先看这张图

后续章节会不断落到具体源码文件。这一节给出"谁依赖谁"的全景地图：进入任何文件前，先在这里确认它站在哪里、被谁调用。

#### 3.4.1 模块依赖图（以当前代码 import 关系为准）

```text
training/src/aegis_training/（公共库，被各脚本复用）
├── paths.py            路径守卫 training_root()/under()          （无内部依赖，所有脚本都用）
├── data_contract.py    RISK_SYSTEM_PROMPT 权威定义 + RiskSample/
│                       parse_sample()/normalize_message()/text_hash（无内部依赖）
├── base_model_gate.py  基座资格检查 verify_snapshot()/GateReport  （无内部依赖）
├── metrics.py          Prediction/classification_report()        （无内部依赖，纯标准库）
├── leakage_guard.py    最终 holdout 泄漏防护（倒排索引+双相似度）  ──依赖→ data_contract
└── source_ingest.py    外部/项目语料弱映射读取                    ──依赖→ data_contract

training/scripts/
├── prepare_risk_sft_v4.py  v9 数据构建   ──依赖→ data_contract(仅 RISK_SYSTEM_PROMPT)、paths
│                                         （泄漏检查/标签裁决为脚本内联实现，见 6.4.2 与 6.7）
├── prepare_risk_sft.py     legacy v1 构建 ──依赖→ data_contract(全量校验)、leakage_guard、
│                                          source_ingest、paths
├── train_risk_qlora.py     训练          ──依赖→ paths（顶层）、base_model_gate（main 内导入）
├── merge_risk_qlora.py     合并          ──依赖→ base_model_gate、paths
├── eval_risk_qlora.py      冻结评测      ──依赖→ data_contract、metrics、paths
│                                         + 生产仓库 app.assessment（运行时导入，见 8.5）
├── eval_risk_devtest.py    devtest 评测  ──依赖→ paths + eval_risk_qlora 的推理器/解析器（跨脚本导入）
└── serve_risk_qlora.py     推理服务      ──依赖→ data_contract、paths
```

#### 3.4.2 从依赖图读出的四个设计事实

1. **v9 管线刻意"轻依赖"**：`prepare_risk_sft_v4.py` 只从公共库取 prompt 与路径守卫，泄漏检查与标签裁决都是内联实现（第 6.4.2 节、第 6.7 节）。原因：它消费的是 consolidated 候选格式（`id/text/risk_level/reason/labels_raw/speaker_scope/...`），与 `data_contract` 面向的原始契约格式（`sample_id/message/...`，见第 6.1 节）是两套数据形态；其裁决逻辑也远比 legacy 弱映射复杂（正则引擎 + 六族人工数据资产）。因此 `parse_sample()` 的全量校验主要保护**原始契约入口**（legacy 构建与人工编写样本），而 v9 构建以自己的方式执行同类约束（长度、泄漏、reason ≤20 字断言）。
2. **`leakage_guard.py` 与 `source_ingest.py` 属于 legacy 管线**：只被 `prepare_risk_sft.py`（第一版流程）使用。它们仍是理解本项目"数据安全"与"弱监督映射"两个核心设计的最佳教学材料，因此保留在 src 中并在本节讲清要点。
3. **评测跨仓库拿规则基线**：`eval_risk_qlora.py` 在运行时把 `AEGIS_PROJECT_ROOT` 指向的生产项目插入 `sys.path` 并执行 `from app.assessment import assess_message`——规则基线用的就是**生产同款规则引擎**。这样验收对比的才真正是"接入后的系统行为"。代价：评测必须设置 `AEGIS_PROJECT_ROOT`（第 8.5 节）。
4. **devtest 评测复用冻结评测的推理器**：`eval_risk_devtest.py` 直接 `from eval_risk_qlora import _build_transformers_runner, _parse_risk`。跨脚本导入保证两个评测**同口径**（同一 generate 参数、同一 JSON 解析逻辑）——口径一致性靠代码复用保证，而不是靠两处配置"记得对齐"。

#### 3.4.3 legacy 管线走读要点：source_ingest.py

`source_ingest.py`（272 行）演示了"外部数据集如何被安全地映射进风险三分类"，是第 6.1.4 节标签原则的另一个代码化身。

| 行号 | 函数/常量 | 职责 |
| --- | --- | --- |
| L13/L17 | `SELF_HIGH_PATTERN`/`THIRD_PARTY_CONTEXT_PATTERN` | 弱映射两个正则：自身意念关键词；"主语(新闻/朋友/他…).{0,24}高危词"的他人语境 |
| L44 | `_distortion_scope_and_risk` | 认知歪曲/SocialCD 的弱映射：命中自身意念→high（他人语境则 low），否则 medium/low |
| L54 | `_speaker_scope_and_risk_from_suicide` | 二分类源标签 × 主体检查合并：**他人语境优先于源标签**（源标签=1 的"他人事件"仍降 low） |
| L177 | `_read_committed_json` | 用 `git show HEAD:路径` 读生产仓库 fixture——**只读已提交版本** |
| L224 | `load_project_candidate_pool` | 只取 fixture 的 `layer=base` 层；`stress` 层永不进入候选 |
| L245 | `dedupe_samples` | 同一规范化文本重复时**保留风险更高的那条**（宁高勿漏） |

两处值得记住的设计：

```python
# L177：只读已提交数据——工作区未提交的改动永远进不了训练集
completed = subprocess.run(["git", "-C", str(project_root), "show", f"HEAD:{relative_path}"], ...)
```

```python
# L54：他人语境覆盖源标签——外部数据集把"朋友说要自杀"标成 1（高危）也不采信
if _is_third_party_context(text):
    return "third_party", "low", "rule_mapped"
if source_label == "1":
    return "self", "high", "source_label"
```

前者把"数据来源必须版本化"从口号变成机制；后者把"他人语境不升自身风险"的原则强制应用于**外部弱标签之上**——这正是第 6.1.4 节"仅凭关键词不升 high"在摄取层的执行。

### 3.5 练习

**练习 3.1** 新开一个 cmd 窗口，运行 `set AEGIS` 查看哪些变量还在；再运行第 3.2 节的两条示例，确认守卫行为。
**练习 3.2** 说出 `checkpoints/` 与 `training/scripts/` 在 Git 策略上的区别及原因。
<details>
<summary>参考答案</summary>
3.1 `set` 设置的变量随窗口关闭失效，新窗口应全部为空（或只剩系统级定义）；示例一输出放行路径，示例二抛 `path escapes allowed root`。3.2 `checkpoints/` 是本地工件（GB 级权重与训练状态），被 .gitignore 忽略、绝不提交；`training/scripts/` 是训练逻辑，必须进 Git 才能被审计与复现。
</details>

---

## 第 4 章 隔离环境

### 4.1 为什么要独立虚拟环境

训练依赖（Transformers 5.x、PEFT、bitsandbytes）版本激进、更新频繁，与任何其他项目（包括生产项目）共存一个 Python 环境时，一次 `pip install` 就可能互相升/降级造成隐性破坏。独立环境（本项目为 `envs/qlora-qwen35/`）保证：训练环境坏了重建即可，其他项目完全不受影响。该环境本身也在训练根目录内、被 Git 忽略。

推荐 Python 3.11、与显卡驱动匹配的 CUDA PyTorch。先安装 GPU 版 PyTorch，再安装训练依赖：

```bat
D:\AegisTraining\envs\qlora-qwen35\Scripts\python.exe -m pip install --upgrade pip
D:\AegisTraining\envs\qlora-qwen35\Scripts\python.exe -m pip install -r D:\AegisTraining\training\requirements-qlora.txt
```

### 4.2 依赖包逐个解释

当前训练依赖最低约束（`requirements-qlora.txt` 使用**下限**版本，不是精确锁定）：

| 包 | 版本要求 | 它是什么 | 为什么必须 |
| --- | --- | --- | --- |
| `transformers` | `>=5.2.0` | Hugging Face 模型库 | 提供 Qwen3.5 架构实现、`apply_chat_template`、Trainer 训练循环 |
| `peft` | `>=0.18.0` | 参数高效微调库 | 提供 LoRA：注入低秩模块、保存/加载 adapter、`merge_and_unload` |
| `bitsandbytes` | `>=0.49.0` | 量化计算库 | 提供 4-bit NF4 量化线性层与 8-bit/分页优化器，QLoRA 的"Q" |
| `accelerate` | `>=1.12.0` | 设备调度库 | 处理 `device_map`、模型在 GPU 上的放置与加速 |
| `datasets` | `>=4.4.0` | 数据集库 | 把 JSONL 高效包装成可训练 Dataset |
| `safetensors` | `>=0.6.2` | 权重序列化库 | 安全读写权重文件（不执行任意代码、可哈希校验） |
| `PyYAML` | `>=6.0.2` | YAML 解析 | 读取唯一参数源 `risk_qlora_4060.yaml` |
| `scikit-learn` | `>=1.6.0` | 传统 ML 库 | 遗留/离线分析辅助；当前训练主链路脚本未直接导入，随 requirements 一并安装以兼容旧工具 |

> **易错点**：`>=` 语义意味着"明天重装环境可能装到更新的版本"。真正复现实验必须保留 `pip freeze` 输出（见 [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) 第 3 节），requirements 文件本身不构成完整环境证据。

### 4.3 PyTorch 与 CUDA 的关系

`pip install torch` 默认装的可能是 **CPU 版**——能 `import torch`，但永远用不了 GPU。CUDA 训练需要：NVIDIA 驱动 → CUDA 运行时 → GPU 版 PyTorch wheel 三者匹配。PyTorch/CUDA wheel 必须根据本机驱动从官方渠道单独安装；不能因为 `pip install -r` 成功就认为 CUDA 训练可用（requirements 里不含 GPU wheel 的正确版本选择）。

### 4.4 BF16 与 FP16：训练脚本会选哪个

两者都是 16-bit 浮点，区别在内部比特分配：**BF16** 与 FP32 共享 8 个指数位，动态范围大，几乎不会数值上溢/下溢，现代 NVIDIA 卡（Ampere 及以后，含 RTX 4060）原生支持；**FP16** 尾数位更多但范围窄，需要 loss scaling 等额外保护。

本项目的规则（`train_risk_qlora.py`）：`bnb_4bit_compute_dtype: bfloat16` 是首选；只有当 GPU 不支持 BF16 且 `fp16_fallback: true` 时，才自动降级为 FP16——"降级是显式配置允许的，不是静默发生的"。**必须同时确认** `torch.cuda.is_available()` 为 `True`、显存足够，并且 BF16 支持情况与配置一致。若 GPU 不支持 BF16，训练脚本只有在 `fp16_fallback: true` 时才允许自动降为 FP16。

### 4.5 环境自检（训练前必做）

```bat
D:\AegisTraining\envs\qlora-qwen35\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

预期输出（示例）：

```text
2.x.x+cu12x
True
NVIDIA GeForce RTX 4060 Laptop GPU
```

逐行解读：第一行是 PyTorch 版本，**必须带 `+cu12x` 后缀**（CPU 版显示 `+cpu`，训练会直接失败）；第二行必须是 `True`；第三行是你显卡的名字，`no cuda` 说明驱动/wheel 不匹配。任何一行不符合预期，先解决环境再继续——第 8 章的训练脚本会在入口处再次强制检查 CUDA，但提前发现能省一次失败启动。

### 4.6 易错点与练习

> **易错点**：
> 1. 用错解释器。装了依赖却装到系统 Python：务必始终使用 `envs\qlora-qwen35\Scripts\python.exe` 的完整路径运行脚本。
> 2. `torch.cuda.is_available()` 返回 False 的三大原因依次排查：CPU 版 wheel（版本号无 `+cu` 后缀）→ 驱动过旧（`nvidia-smi` 看驱动与 CUDA 支持版本）→ 显卡不是 NVIDIA（本流程不支持）。

**练习 4.1** 运行 4.5 节自检，把三行输出记入你的实验记录（这就是 REPRODUCIBILITY 要求的环境快照之一）。
**练习 4.2** 运行自检命令并观察版本号后缀：如果是 `+cpu`，说明装的是哪种 wheel？应如何修复？
<details>
<summary>参考答案</summary>
4.2 `+cpu` 表示 CPU 版 PyTorch wheel，`cuda.is_available()` 恒为 False。修复：从 PyTorch 官方渠道按本机驱动选择对应 CUDA 版本的 wheel 重新安装（覆盖现有 torch），再运行自检确认出现 `+cu12x` 与 `True`。
</details>

---
## 第 5 章 基座模型 gate

### 5.1 为什么训练前必须过 gate

训练的一切都建立在"基座是什么"这个前提上。如果基座被替换成同名但不同内容的文件（HF 仓库更新、镜像不完整、甚至被换成 GGUF），训练会照常跑完，产出一个**无法追溯、无法对比、无法复现**的模型。gate 是在花 3 小时训练之前，花 3 秒确认地基无误。

训练只能使用固定 revision 的官方 Qwen3.5-2B-Base safetensors 快照：

```text
repo:       Qwen/Qwen3.5-2B-Base
revision:   b1485b2fa6dfa1287294f269f5fb618e03d52d7c
model_type: qwen3_5
hidden:     2048
text layers: 24
```

快照至少应包含 `config.json` 和 `model.safetensors.index.json`，并且权重清单非空。执行 gate：

```bat
D:\AegisTraining\envs\qlora-qwen35\Scripts\python.exe ^
  D:\AegisTraining\training\src\aegis_training\base_model_gate.py ^
  --snapshot-dir "D:\AegisTraining\models\Qwen3.5-2B-Base"
```

### 5.2 gate 检查项逐条解读

gate 会检查：

- **repo、revision、model type、架构、hidden size、层数和权重清单**：与钉死的官方值逐一比对（`qwen3_5` / `Qwen3_5ForCausalLM` 系 / 2048 / 24 层）。任何一项对不上都说明本地文件不是验收时的那个模型。
- **是否为可训练 safetensors，而不是 Ollama GGUF**：缺 `model.safetensors.index.json` 通常意味着拿到的是推理工件或不完整下载。
- **模型是否暴露视觉模块**：模块名含 vision/visual/image/video 的结构会被拒绝。训练脚本只接受文本 causal-LM 路径——在多模态权重上跑纯文本训练，等于训练了一个从未被计划评测过的模型。
- **可选的 Ollama 转换 provenance**：本机同时存在 Ollama 的 `qwen3.5:2b`（Q8 GGUF）。gate 不禁止它的存在，但要求：没有精确转换证明时，状态只能是 `pass_same_family_with_caveat`——"同族、同规模"，**不能声称**与官方 revision 已证明同源。严格的 `--require-exact-ollama-provenance` 只用于已有完整转换证明的场景；当前本机 `qwen3.5:2b` 没有这样的证明，因此不应强行通过严格模式。

gate 通过状态有两级，语义要与代码对齐（`base_model_gate.py:111`）：`pass_same_family_with_caveat` 是**常规通过**——结构、架构、规模全部匹配官方快照，但 Ollama 转换 provenance 未验证；`pass_exact` **仅在提供了通过验证的 Ollama 转换证明记录时出现**。当前 v9 管线不传 provenance（训练与合并都调 `verify_snapshot(..., provenance_path=None)`），因此实际产出的 training-manifest 里状态恒为 `pass_same_family_with_caveat`（v4/v5/v7 的训练 manifest 可复核）。dry-run 输出里必须能看到该状态（第 8.1 节）；不要把"没看到 `pass_exact`"当成 gate 失败。

### 5.3 深入理解：GGUF 与 safetensors 的区别

| 维度 | safetensors（训练用） | GGUF（推理工件） |
| --- | --- | --- |
| 生态 | Hugging Face / Transformers / PEFT | Ollama / llama.cpp |
| 内容 | 与官方发布逐字节对应的原始权重 | 可能经过再量化（如 Q8）、重排与打包 |
| 溯源 | revision + SHA-256 可精确锁定 | 通常只能追溯到"某个转换"，且证明难获取 |
| 能否作为 QLoRA 基座 | ✅ 唯一接受格式 | ❌ 禁止（第 1 章红线） |

**为什么红线禁止 GGUF 当基座**：训练是在基座上叠加学习的。若基座与验收评测所用的官方权重不同源，所有"基座 vs QLoRA"的对比都失去参照物，而且 PEFT/bitsandbytes 的量化路径也不保证对 GGUF 转换件数值等价。

> **易错点**：
> 1. 从非官方镜像或网盘下载"同名字"模型——gate 会因 revision/结构不符拒绝，这是保护而不是麻烦。
> 2. 以为 `pass_same_family_with_caveat` 是 gate 失败——它是合法状态，只是要求你不得宣称 Ollama 工件与官方权重同源；训练本身只依赖 safetensors 快照。

### 5.4 练习

**练习 5.1** 运行 5.1 节的 gate 命令，把输出保存下来，并逐字段核对 repo/revision/model_type/hidden/layers。
**练习 5.2** 同事提议"为了省下载时间，直接从 Ollama 模型目录导出权重当基座"。写两句话拒绝他，并给出正确做法。
<details>
<summary>参考答案</summary>
5.2 GGUF 是推理工件，无法证明与官方 revision b1485b2…同源，用它训练会让全部验收失去参照（违反第 1 章红线）；正确做法是从 Hugging Face 官方仓库下载钉死 revision 的 safetensors 快照到 `models/Qwen3.5-2B-Base/`，再跑 gate 确认状态为 `pass_same_family_with_caveat`（即第 5.2 节的"常规通过"；`pass_exact` 需要额外的 Ollama 转换证明，与本快照训练无关）。
</details>

---

## 第 6 章 数据契约和数据构建

> 本章是全书信息量最大的一章。核心思想先记住一句话：**模型的上限是数据决定的，代码只能逼近这个上限**。v9 的验收成绩，首先是数据契约、标签纪律和泄漏防护的胜利。

### 6.1 原始数据契约

#### 6.1.1 为什么先定契约

训练数据来自多个渠道（外部数据集映射、人工编写、蒸馏挖掘）。没有统一契约时，" medium 是什么意思"" reason 能多长""第三人称怎么标"每个来源一套答案，模型学到的就是一锅粥。契约的作用：让所有来源在**进入训练之前**被校验、不合格的直接报错——把数据问题拦截在最便宜的阶段。

#### 6.1.2 契约定义

原始 JSONL 每行至少包含：

```json
{
  "sample_id": "example-001",
  "message": "我最近觉得自己很多余。",
  "risk_level": "medium",
  "reason": "明显痛苦但无自伤意向",
  "source": "curated-source",
  "source_version": "source@revision",
  "label_method": "manual_reviewed",
  "speaker_scope": "self",
  "review_status": "manual_reviewed"
}
```

字段逐个解释（实现见 `training/src/aegis_training/data_contract.py`）：

| 字段 | 约束 | 为什么需要它 |
| --- | --- | --- |
| `sample_id` | 必填、非空、全局唯一 | 泄漏拒绝、误判追溯都要靠 ID 定位到单条 |
| `message` | 必填非空字符串，≤1500 字符 | 训练输入本体；限长防止极端长文撑爆上下文 |
| `risk_level` | 只能是 `low` / `medium` / `high` | 三分类标签，与提示词契约和验收指标一一对应 |
| `reason` | 必填非空，≤20 字符 | 模型输出的依据字段，训练时作为 assistant JSON 的一部分 |
| `source` | 必填非空 | 数据来源标识，审计与分来源诊断的依据 |
| `source_version` | 默认 `unspecified` | 来源的精确版本，复现时锁定到具体 revision |
| `label_method` | `source_label` / `rule_mapped` / `project_mapped` / `manual_reviewed` / `synthetic` | 如实记录这条标签怎么来的 |
| `speaker_scope` | `self` / `third_party` / `fictional` | 说话主体：自己/他人/虚构——high 只允许 self |
| `review_status` | `source_provided` / `not_reviewed` / `manual_reviewed` | 如实记录复核状态 |
| `annotator` | 默认 `automated-mapping` | 具体标注者/映射器 |
| `adjudication_note` | 可空 | 裁决备注（重标/争议记录） |

#### 6.1.3 校验规则（代码强制执行）

- `risk_level` 只能是 `low`、`medium`、`high`；
- `reason` 最多 20 个字符；
- 原始 message 最多 1500 个字符；
- `speaker_scope` 只能是 `self`、`third_party`、`fictional`；
- `high` 必须描述说话人自身风险（`speaker_scope` 必须是 `self`）；
- `sample_id` 不得重复；
- `label_method` 和 `review_status` 必须如实记录，自动映射不能伪装成人工审核。

违反任何一条，`parse_sample` 抛出 `DataContractError`，构建立即终止——**报错是契约在正常工作**。

**可直接运行的示例**（不依赖 GPU，验证一条样本是否满足契约）：

```bat
D:\AegisTraining\envs\qlora-qwen35\Scripts\python.exe -c "import sys; sys.path.insert(0, r'D:\AegisTraining\training\src'); from aegis_training.data_contract import parse_sample; s = parse_sample({'sample_id':'demo-001','message':'我最近觉得自己很多余。','risk_level':'medium','reason':'明显痛苦但无自伤意向','source':'demo','speaker_scope':'self','label_method':'manual_reviewed','review_status':'manual_reviewed'}); print('OK:', s.sample_id, s.risk_level)"
```

把 `risk_level` 改成 `"extreme"` 或把 `reason` 写成超过 20 个字的句子再运行，应看到 `DataContractError`。

#### 6.1.4 标签原则与边界案例

标签原则（训练数据的"法律条文"）：

- `high`：说话人自身明确或隐喻的自杀、自伤意向、计划或即时危险；
- `medium`：明显痛苦、绝望或功能损害，但没有足够自身自伤证据；
- `low`：一般困扰，以及新闻、论文、影视、朋友或虚构情境中的高危词；
- "撑不下去""不配""没意义"等不能脱离主体、意图和计划性，仅凭关键词升级为 `high`。

为什么这组边界如此划分：系统的输出会触发不同强度的业务响应（high 触发危机干预），漏报 high 的人身代价极高，但把所有痛苦都判 high 会造成"狼来了"、淹没真正的高危。因此边界画在"**自身 + 意向/计划**"上：痛苦（medium）与危险（high）的分界是**有没有指向自身的自伤证据**；他人/虚构语境（low）再吓人也不升自身风险。

边界案例对照（与提示词契约 v2 一致）：

| 消息示例 | 标签 | 关键理由 |
| --- | --- | --- |
| "我已经写好了遗书，打算这周末结束一切" | high | 自身 + 明确计划 |
| "有时候觉得要是没出生过就好了" | high | 自身 + 隐喻式自伤意念（提示词明确列出此类） |
| "撑不下去了，每天失眠，撑不住了" | 视语境 | 出现"撑不下去"不自动升 high：需结合主体、意图和计划性判断 |
| "最近压力很大，晚上睡不着，觉得自己很多余" | medium | 明显痛苦、功能受损，但无自身自伤证据 |
| "新闻里有人轻生了，好难过" | low | 他人语境（third_party），高危词不升自身风险 |
| "我在论文里讨论自杀干预方法" | low | 虚构/学术语境 |
| "如果我不在了，大家会过得更好吧" | high | 自身 + 自我消失的隐喻表达（提示词 v2 覆盖"要是不在了就好了"） |

> **易错点**：
> 1. 看到"死""自杀"字样就标 high——这是第二、三、五、六版反复踩过的坑（第三人称/自我否定句被误升级，见谱系索引）。标签判的是**说话人自身的意向**，不是关键词。
> 2. 把自动映射的样本 `label_method` 写成 `manual_reviewed`——契约校验也许拦不住谎言，但 manifest 审计会，谱系索引里这类记录等于学术造假。

### 6.2 当前数据来源和规模

当前主流程使用已审查的 `training/data/consolidated_risk_v1`，通过 `prepare_risk_sft_v4.py` 生成 `data/risk_sft_v9`：

| 输出 | 当前规模 | 用途 |
| --- | ---: | --- |
| `train.jsonl` | 2867 | 训练 |
| `dev.jsonl` | 200 | epoch 评估、early stopping 和最优模型选择 |
| `test.jsonl` | 1414 | devtest 参考评测，不作为最终冻结验收集 |
| 项目 `base` | 63 | 可作为开发候选 |
| 项目 `stress` | 87 | 永久冻结最终 holdout，不能进 train/dev |

四个集合的分工与"为什么"已在第 0.8 节讲清；这里补充两个数字的来历：dev 200 = dev 配额 70/60/70 之和（第 6.6.1 节表格）；devtest 1414 是第三版起引入的"同规则重标"开发测试集，用于在**不动用冻结集**的前提下做大样本诊断——它是防止"反复在 stress 上调参"这道防线的关键设计（对冻结集反复调参等于把高考卷变成练习题）。

### 6.3 提示词契约 v2

当前 system prompt 必须和训练、评测、推理服务保持一致：

```text
你是校园心理支持系统的风险评估器,只依据当下说话人的**自身意向**判断风险:high(自伤/自杀意念或计划,包括隐喻式表达如想消失/从没出生过/撑不下去/永远睡过去/离开这个世界/结束这一切/不愿再醒来/要是不在了就好了),medium(强烈痛苦、绝望、功能受损但无直接自伤)、low(一般困扰,包括提及他人/虚构情境的高危词,如"新闻里有人轻生/写论文提到自杀/朋友直播自杀"——这些不应升为自身风险)。判定原则:仅评估说话人自身;他人或虚构内容提及自杀/伤害不视为自身 high。只输出一个 JSON 对象,不要输出任何其他文字:{"risk_level": "low|medium|high", "reason": "20字以内依据"}
```

**原理：为什么 system prompt 是"契约"而不是"配置"**——模型是**针对这段文字**训练的：每条 SFT 样本的 system 段就是它。训练和推理时 prompt 不同，等于考前发的题目类型和平时练习完全不一样，性能必然漂移。当前代码中它有**两处独立的物理副本**：`training/src/aegis_training/data_contract.py` 是权威定义——数据构建脚本 `prepare_risk_sft_v4.py`、冻结评测 `eval_risk_qlora.py` 和本地服务 `serve_risk_qlora.py` 都从这里导入，自动保持一致；生产项目 `app/llm/client.py` 持有另一份独立副本，必须与 data_contract 逐字同步。`build_consolidated.py` 内嵌的是构建期旧 prompt，仅作审计产物，不参与 v9。

v1→v2 的关键变化是移除"**不配**""**活着多余**"等宽泛示例。原因见谱系索引：冻结金标里"觉得拖累家人/不配拥有"这类**自我贬低**判 medium，而"不配活着""想消失"这类**指向自身消失**的才判 high——v1 把两类示例混列，模型在这条细线上大量误判。移除歧义示例后，第五、六版卡住的 corp-084/091 在 v9 修复。

> **易错点**：改 prompt 是**重训级事件**，不是改配置。任何一处副本的"顺手微调"都会造成训练-生产漂移；历史版本评测必须绑定各自训练期的 prompt，不能用 v2 重新解释旧版数字（第 13 章）。

### 6.4 泄漏防护

#### 6.4.1 防的是什么

**数据泄漏**：与评测集相同或高度相似的文本混入训练数据，模型"背过题"，评测分数虚高而真实能力不变。本项目要保护的是**永久冻结**的 stress 87 条——它一旦被污染，八项门槛全部失去意义，且损失不可逆（你无法知道没有泄漏时的真实分数）。

#### 6.4.2 防护机制逐层拆解

`prepare_risk_sft_v4.py` 会：

1. 读取项目 fixture 中的 stress 87 条，并**确认数量准确**（不是 87 条直接报错——防止 fixture 版本漂移）；
2. 对候选文本做 NFKC、去空白、转小写规范化（同一个文本的各种"化妆写法"——全角/半角、多空格——归一成同一个样子，防止换字符绕过检测）；
3. 先用规范化文本哈希拒绝**精确复用**（逐字节相同必被拦截，代价 O(1)）；
4. 再用字符 3-gram Jaccard 检测**近重复**，默认阈值 `0.82`：把文本切成所有连续 3 字符片段的集合，两文本相似度 = 交集大小 ÷ 并集大小；满足阈值即拒绝（内联实现 `leaks()`，`prepare_risk_sft_v4.py:665`，对候选逐一比对 87 条 stress 的 3-gram 集合）；
5. 将泄漏原因写入 manifest 的 `leakage_rejected`，**逐条可审计**（拒绝不是静默丢弃）；
6. 对 train 内部和 dev 也做去重，避免验证样本被训练集吸收（dev 里有 train 原题会让 eval_loss 虚低、模型选择失真）。

**为什么阈值是 0.82 而不是 0.99 或 0.5**：隐喻类高危样本的表达高度模板化（"想消失/离开这个世界/结束这一切"），合法训练样本与 stress 样本天然存在共享片段。阈值过高（如 0.95）漏掉"改几个词的原题"；过低（如 0.5）会把大量合法的隐喻模板样本误杀，训练集失去最难最重要的部分。0.82 是在"漏检"与"误杀"之间反复权衡的经验值——**不允许为了多进几条数据而调低它绕过保护**。

**两套实现，同一个原则**：上面这份是 v9 管线的**内联实现**（精确哈希 + 3-gram Jaccard，逐条比对 87 条 stress，构建脚本内约 9 行代码）。`src/aegis_training/leakage_guard.py` 是同一原则的另一份实现，服务于 **legacy 管线** `prepare_risk_sft.py`：相似度取 Jaccard 与 SequenceMatcher 的最大值（`leakage_guard.py:36`），并先建 3-gram 倒排索引预筛——只对"与某条 stress 共享 ≥3 个三元组"的候选做精确相似度计算（`leakage_guard.py:44`），近重复文本必然共享大量 3-gram，预筛不漏检。两套实现的阈值同为 0.82、规范化同源（复用 `data_contract.normalize_message`：NFKC+去空白+小写）。**复刻时不要混用**：给 v9 管线换用 `leakage_guard` 会把相似度口径从"纯 Jaccard"变成"max(Jaccard, SequenceMatcher)"，泄漏拒绝集合随之改变，manifest 数字不可比。

#### 6.4.3 明确的禁区清单

`risk.json`、`routing.json`、`multi_turn_corpus.json`、`safety.json`、RAG fixtures、Harness、probe、政策文档、Skill 内容和测试消息**不得作为训练语料**。这些文件要么就是评测/验收资产，要么与规则引擎同源设计——混入即等于把考卷编进教材。

同时要清醒认识边界：stress 是工程冻结集，不是完全独立的临床盲测集（其中的合成表达与仓库规则/测试存在共同设计背景）；上线前仍需外部专家审核和独立测试（第 9、13 章）。

**可直接运行的示例**（直观感受 3-gram Jaccard，纯标准库）：

```bat
D:\AegisTraining\envs\qlora-qwen35\Scripts\python.exe -c "def g(t):return {t[i:i+3] for i in range(len(t)-2)}; a='有时候真的想从这个世界上消失'; b='真的想从世界上消失掉算了'; print(len(g(a)&g(b))/len(g(a)|g(b))); print(len(g(a)&g('今天天气不错心情挺好的'))/len(g(a)|g('今天天气不错心情挺好的')))"
```

第一行输出两条文本的相似度（应接近 0.4-0.6 的近重复区间，改到逐句雷同时会冲过 0.82）；第二行是完全无关文本的低相似度。动手改字符串，观察阈值两侧的行为。

### 6.5 可选：重建 consolidated 候选池

当前 v9 主流程默认使用已审查的 `training/data/consolidated_risk_v1`。只有需要重建候选池、审计来源或在受控数据环境中复现实验时，才运行 `build_consolidated.py`：

```bat
set AEGIS_CONSOLIDATED_ROOT=D:\AegisTraining\training\data\consolidated_risk_v1
python D:\AegisTraining\training\data\consolidated_risk_v1\build_consolidated.py
```

重建前必须准备并记录：

- 项目 `representative_corpus.json`，其中冻结输入应保持当前版本；
- PsySUICIDE、suicide 原始数据和 `metaphor_corpus_v1.jsonl` 的来源、许可证、获取时间和 SHA-256；
- `AEGIS_TRAINING_ROOT`、`AEGIS_PROJECT_ROOT`、`AEGIS_PROJECT_CORPUS`；
- 外部数据脱敏、授权和公开范围审查结果。

该脚本只读训练根目录，按归一化文本去重，并对历史 25 条冻结样本做 3-gram Jaccard `0.82` 泄漏过滤。它生成的 `train_messages.jsonl` 使用构建阶段的旧 prompt，是审计中间产物；当前 v9 仍必须经过下一步 `prepare_risk_sft_v4.py` 重新裁决和生成。

> **易错点**：把 `build_consolidated.py` 的输出直接当训练数据——它用的是旧 prompt、旧裁决标准。正确链路永远是 `consolidated_risk_v1 → prepare_risk_sft_v4.py → risk_sft_v9`（见 training/data/README.md）。

### 6.6 构建当前数据与 manifest 检查

#### 6.6.1 默认构建参数逐项解释

| 参数 | 默认值 | 含义 | 设计意图 |
| --- | --- | --- | --- |
| `--train-quota` | `1000 1000 1000` | high / medium / low 目标配额 | 三类均衡，防止模型偏向多数类 |
| `--dev-quota` | `70 60 70` | high / medium / low 开发集配额 | 同上，dev 合计 200 |
| `--max-text-chars` | `300` | v9 构建时文本长度上限 | 与 `cutoff_len=512` 配合，保证 assistant 目标不被截断（第 7.5 节） |
| `--leak-threshold` | `0.82` | 与 stress 的近重复阈值 | 见第 6.4.2 节 |
| `--medium-upsample-cap` | `1.35` | medium 扩量上限控制 | medium 真实分布稀疏，允许合成/蒸馏补量，但封顶 1.35 倍防止过矫 |
| `--distill-limit` | `200` | 蒸馏来源挖掘上限 | medium 扩量来源之一（校园合成 + 蒸馏挖掘），限量防稀释 |
| 随机种子 | `42` | 确定性哈希抽样 | 同一输入永远得到同一输出，配额抽样可复现 |

**抽样为什么是"确定性哈希"**（`take()` 函数）：候选按文本哈希排序后截取配额，而不是随机洗牌——不依赖随机数生成器状态，两次构建（哪怕在不同机器）得到完全相同的 train/dev，可复现性从机制上保证。

**人工样本"整组直入"但计入配额**（与代码核对：`prepare_risk_sft_v4.py:748-764`）：人工困难负例、成对对照、第三人称负例先直接进入 train，随后抽样只补足 `配额 − 人工数` 的差额——配额是**含人工样本的上限**，不是"抽样之外再叠加"。实际 train 行数（2867）低于 3000 的配额总和，因为候选池经长度/泄漏过滤与 dev 预留后供给不足，`take()` 在池小于配额时全取。为什么整组直入：成对对照样本的价值恰恰在"主语一换、标签翻转"的对照关系上（"我想消失"=high vs "我朋友想消失"=low），一旦被随机抽样拆散或截断，对照教学即失效。最终以 `manifest.json` 为准。

#### 6.6.2 构建命令

```bat
D:\AegisTraining\envs\qlora-qwen35\Scripts\python.exe ^
  D:\AegisTraining\training\scripts\prepare_risk_sft_v4.py ^
  --consolidated-root "D:\AegisTraining\training\data\consolidated_risk_v1" ^
  --corpus "%AEGIS_PROJECT_CORPUS%" ^
  --output-root "D:\AegisTraining\data\risk_sft_v9"
```

#### 6.6.3 构建后必须检查 manifest

```bat
python -c "import json; p=r'D:\AegisTraining\data\risk_sft_v9\manifest.json'; m=json.load(open(p,encoding='utf-8')); print(m['schema_version']); print(m['train']); print(m['dev']); print(m['devtest']); print('leaks=',len(m['leakage_rejected']))"
```

预期 `schema_version` 为 `risk_sft_v9`，`leakage_rejected` 应被审阅，不能只看命令是否返回 0。逐项含义：`schema_version` 确认产物代际；`train/dev/devtest` 数量与 6.2 节表格对得上；`leakage_rejected` 列出每条被拒样本的 ID 与拒绝原因（`exact` 或 `ngram:序号`）——**审阅它是确认"防护在工作"而非"防护在碍事"**：被拒样本应能看出与冻结集的真实相似，而不是莫名其妙的文本。

manifest 还会记录样本哈希、来源、映射方式、分布、重标摘要（`relabel_summary`）、hard negatives 与 medium 扩量数量——这些是第 14 章可复现清单和第 11 章发布元数据的直接输入。

> **易错点**：
> 1. 构建"成功"（退出码 0）就跳过 manifest 审阅——退出码只说明程序没崩，不说明数据质量合格。
> 2. 手工编辑 `train.jsonl` 加私货——绕过了契约、泄漏检查和 manifest 记录，等于亲手废掉整条审计链。要加数据请走 `authored/` 与构建参数。

### 6.7 构建脚本源码走读：prepare_risk_sft_v4.py

> 本节是 §6.1-6.6 的代码落地层。读完你应当能不看源码复述：一条 consolidated 候选如何变成一条 SFT 样本、每一步为什么这样设计。行号以当前代码为准（重构后会漂移，以函数名为准）。

#### 6.7.1 函数地图

| 行号 | 函数/常量 | 职责 |
| --- | --- | --- |
| L44-98 | `IDEATION`/`MEANINGLESS`/`DEATH_CUE`/`MEDIUM_MINE`/`SELF_ACT_*`/`RECOVERY_GUARD`/`DISCUSSION_GUARD`/`PAST_GUARD`/`OTHER_SUBJECT`/`SELF_TOKEN`/`PAIN_NO_INTENT`/`SYMPTOM` | 裁决正则引擎：把 §6.1.4 标签原则编码为可执行规则 |
| L101 | `_nearest_subject_is_self` | 就近主语裁决（12 字窗口） |
| L116/143/154/163/173 | `_self_ideation`/`_selfharm_history`/`_self_pain`/`_self_symptom`/`_mine_medium` | 五类判定器，全部基于就近主语 |
| L181/194 | `load_campus_medium`/`mine_distill_medium` | medium 扩量两来源 |
| L235-259 | `REASON_BY_FINE_LABEL`/`REASON_MEDIUM`/`REASON_LOW_*` | reason 短语常量表 |
| L263-278 | `_norm`/`_sha256`/`_order_key`/`_trigrams` | 规范化、确定性排序键、3-gram |
| L281-296 | `_guard`/`_write_text` | 唯一写出入口（路径守卫复用 `paths.under`） |
| L299-508 | 六族人工数据资产 + `build_hard_negatives` | 见 6.7.5 |
| L511/521/579 | `load_consolidated`/`relabel`/`rewrite_reason` | 载入、六分支裁决漏斗、reason 定稿 |
| L616 | `take` | 确定性哈希抽样 |
| L624-647 | `main` 前段 | argparse、路径守卫、corpus 解析 |
| L658-673 | stress 断言 + `leaks` | "必须 87 条" + 内联泄漏检查 |
| L675-728 | 主流程前半 | 载入→重标→长度/泄漏过滤→人工资产直入→medium 扩量 |
| L730-791 | 主流程后半 | 配额分配、dev 预留、≤80 降密度、22% 第三人称、去重与定序 |
| L793-842 | `to_sft` 与三份输出 | SFT 组装、train/dev/test 落盘、两份 review 文件 |
| L848-879 | manifest | 全量审计记录（字段表见 6.7.8） |

#### 6.7.2 裁决正则引擎：标签原则如何变成代码

核心难题：§6.1.4 说"标签判的是说话人自身的意向"，但正则只会匹配字面。脚本的解法是三件套：**意图正则 × 语境守卫 × 就近主语裁决**。

（1）就近主语裁决（`_nearest_subject_is_self`，L101）：

```python
def _nearest_subject_is_self(text, start, window=12):
    span = text[max(0, start - window):start]          # 只看匹配点前 12 字
    self_pos  = [m.start() for m in SELF_TOKEN.finditer(span)]     # 我/自己/咱
    other_pos = [m.start() for m in OTHER_SUBJECT.finditer(span)]  # 朋友/新闻/他/…
    if not self_pos and not other_pos:
        return None            # 无主语：默认自身（第一人称独白）
    if self_pos and (not other_pos or max(self_pos) > max(other_pos)):
        return True            # 离匹配点最近的主语是自身
    return False
```

为什么只看前文、取最近：中文口语的风险短语几乎都紧跟主语（"我快撑不住了"），离匹配点更近的主语就是短语的主人；无主语的独白（"撑不住了……"）默认按第一人称处理——宁可交给后续验收去发现误报，也不放过潜在高危。

（2）三重语境守卫，防三类经典误升级：

| 守卫 | 防什么 | 例子 |
| --- | --- | --- |
| `PAST_GUARD`（L85：小时候/当年/曾经/以前…） | 既往自伤旧事被当当前意念 | "我小时候割过腕，现在好了" → 自伤史（medium），不是 high |
| `RECOVERY_GUARD`（L82：而不是/拒绝/停止…） | 康复叙述中的自伤词 | "我用运动替代了自伤" → 不升级 |
| `DISCUSSION_GUARD`（L83：研究/方式/方法/效果…） | 科普/讨论语境 | "论文里比较了各种干预方式的效果" → 不升级 |

（3）判定器 `_self_ideation`（L116）按"锚定自伤（我+割腕/遗书…）→ 裸自伤（割/划/伤自己）→ 意念词（IDEATION）"顺序扫描；锚定自伤命中 `PAST_GUARD` 时**跳过该处继续扫描**而不是整句否决——同一句可能既有旧事也有新意念。注意 `IDEATION`（L44）本身就编码了边界裁决：`(?<!不)想死` 用否定后顾排除"不想死"；"活不下去"是多义词，按用户裁决归痛苦短语（medium）而不是意念。

**正反例速查**（可对照源码验证）：

| 文本 | 判定 | 触发机制 |
| --- | --- | --- |
| "我最近总想着怎么消失" | high | IDEATION"想消失"+ 无主语默认自身 |
| "他说他最近总想消失" | low | 就近主语 = 他 |
| "我不想死了" | 不命中 | `(?<!不)想死` 否定后顾 |
| "我小时候割过腕" | medium（selfharm_history） | PAST_GUARD 命中 |
| "快撑不下去了，来预约心理咨询" | medium（痛苦无意图族） | PAIN_NO_INTENT 命中、无自伤意图；同型文本也是 `hard-negative-v4` 的人工正例 |

#### 6.7.3 relabel()：六分支裁决漏斗（L521）

```text
consolidated 候选（每条带 labels_raw 细标签）
 ① high 且 labels_raw 只含「用户攻击行为」            → 丢弃（aggression_only）
 ② high → 按 labels_raw 细标签定 _fine；
       被动自杀意图 + MEANINGLESS 且无 DEATH_CUE      → 降 medium（passive_demote）
 ③ low + 自身意念                                    → 升 high（explicit_relabel，不受第三人称豁免）
 ④ low + 非第三人称 + 自伤史/康复叙述                → medium（selfharm_history）
 ⑤ low + 非第三人称 + 自身痛苦短语                   → medium（pain_without_intent）
 ⑥ low + 非第三人称 + 显著症状                       → medium（significant_symptom）
 ⑦ low + 非第三人称 + 痛苦线索挖掘（MEDIUM_MINE）    → medium（distress_mine）
 ⑧ medium → exploration；其余 → low
```

为什么顺序不可调换：①必须在②前（先清掉与风险无关的攻击样本）；③必须在④-⑦前（意念是 high 级信号——"我想死"同时命中痛苦短语，若⑤在前会把高危降级）；④在⑤前（旧事叙述往往也含痛苦词，"既往史"是更精确的归因）。每次改判写入 `relabel_log`（`{id, from, to, rule}`），manifest 的 `relabel_summary` 按"from→to"聚合计数——**重标可审计，不是黑盒**。

#### 6.7.4 reason 定稿与防模板坍缩（L579）

reason 不照抄候选字段，而按 `_fine` 从常量表映射：high 用 `REASON_BY_FINE_LABEL`（7 项细标签→短语）；medium 用 `REASON_MEDIUM`/`REASON_MINE`；low 按 `_order_key(text)` 哈希从 4 个变体中取一（"一般困扰无自伤意向"等）——避免千篇一律让模型把 reason 背成模板。人工资产（hard-negative/paired/selfworth/campus/living-burden）保留人写的 reason。`to_sft()`（L793）最后再次断言 ≤20 字，超限抛错——与 `data_contract` 校验形成双保险。

#### 6.7.5 六族人工数据资产（L299-508）：失败复盘如何变成数据

| 资产（`_origin`） | 规模 | 解决什么问题 |
| --- | --- | --- |
| `hard-negative-v4` | 48 medium + 6 high 对照 | 第二版三条误升级（corp-090 硬撑/corp-097 图什么/corp-099 怕垮）：同短语家族各配无意图 medium 正例 |
| `paired-contrast-v6` | 20 对（40 条） | 教"主语归属"：同句式换主语翻转标签（"我快撑不住了"=medium ↔ "室友说她快撑不住了"=low） |
| `selfworth-contrast-v8`（v9 重构） | 10 对（20 条）+ 4 第三人称，共 24 条 | **corp-084/091 修复的落地代码**：最小差异对——"不配被爱"medium ↔ "不配活着"high，死亡词是唯一判别特征且位置前/中/后轮换，防模型学"家族→查 high"的捷径 |
| `living-burden-v9` | 10 | corp-084 家族（"活着+负担"）的 medium 变体族，无死亡词 |
| `third-person-neg-v6` | 28 | 新闻/影视/他人语境 low 困难负例 |
| `campus-medium-v7` + `distill-mined-v7` | ≤200 | medium 扩量：人工校园样本 + 蒸馏集挖掘（10-90 字、无意念词、非他人主语开头、命中痛苦/症状规则、0.6 Jaccard 多样性去重） |

`build_hard_negatives()`（L484）统一打 `_origin`/`fine` 标，manifest 的 `train.origin` 因此能按来源统计每族实际入选多少条。第 9 章验收里"隐喻隐式新增 +6"和"corp-084/091 回归 medium"，正是 `selfworth-contrast-v8` 与 `living-burden-v9` 两族在冻结集上的回响——**数据设计、历史失败与验收结果三者在代码里闭环**。

#### 6.7.6 确定性抽样与配额分配（L616-791）

- **确定性洗牌**（`_order_key`，L271）：排序键 = `sha256("42|" + 规范化文本)`。完全不用 random——两次构建（哪怕不同机器）必得同一 train/dev，可复现从机制上保证（§6.6.1"确定性哈希抽样"就是它）。
- **`take(pool, quota, priority)`**（L616）：按 `_order_key` 排序后截前 quota 条，池小于配额时全取；priority（隐喻样本 `metaphor_flag` 优先）用稳定排序前置，不破坏确定性。
- **dev_medium 先预留**（L739-741）：medium 池稀缺，先给 dev 抽足 60 条，train 再从剩余抽——防止训练把验证集抽干。
- **自我否定子族降密度 ≤80**（L743-747）：`SW_SURFACE`（没用/废物/拖累/累赘/多余/不配/一无是处）命中的"被动自杀意图"high 子族最多取 80 条——该表面族若不设上限会在训练集中形成密度高压（第五、六版的边界病灶）。
- **第三人称 low 配额 22%**（L757）：`low_tp_share=0.22` 单独配额，保证"提及他人"类 low 的对抗强度。
- **人工样本整组直入、计入配额**：机制见 §6.6.1 的修正说明。

#### 6.7.7 devtest 全量同规则重标（L829-842）

test.jsonl 不是"从未动过的原始集"：它用**同一套 `relabel()`/`leaks()`/`rewrite_reason()`** 全量重标（不抽样），再过长度与泄漏过滤（consolidated test 1456 条 → v9 devtest 1414 条）。为什么：训练金标与测试金标必须是同一裁决标准，否则模型学的和考的不是同一套规则——"同规则重标"正是 devtest 数字可与训练分布对照解释的前提（这也再次解释了 §8.6 为什么 devtest 只能做开发诊断）。另产出两份人工复核文件：`hard_negatives_v4.review.jsonl`（每条困难负例带 rationale）与 `medium_sources_v7.review.jsonl`（扩量来源清单）——扩量样本全部可追溯、可复核。

#### 6.7.8 manifest 字段对照（L848-879）

| 字段 | 含义 | 审计用途 |
| --- | --- | --- |
| `schema_version`/`seed` | `risk_sft_v9` / 42 | 产物代际与确定性 |
| `system_prompt`/`system_prompt_version` | v2 全文与版本说明 | prompt 锁定（§6.3） |
| `sources` | 三个输入路径 | 溯源 |
| `relabel_log`/`relabel_summary` | 逐条改判记录 / "from→to"聚合 | 裁决可审计 |
| `length_dropped`/`leakage_rejected` | 长度剔除数 / 泄漏拒绝清单 | 防护在工作 |
| `train.{count,distribution,origin,medium_upsampled}` | 规模/类别分布/来源构成/扩量数 | 与 §6.2 表格对账 |
| `dev`/`devtest` | 各自 count/distribution（devtest 另有 relabel_summary） | 覆盖核对 |
| `hard_negatives.{kept,rejected}` | 人工资产通过/被拒数 | 人工质量关 |
| `quotas` | 本次生效配额 | 参数记录 |

### 6.8 练习

**练习 6.1** 构造三条样本并运行第 6.1.3 节的 parse_sample 示例验证：一条合法的 `low`（第三人称），一条非法的 `high`（`speaker_scope: "third_party"`，应报错），一条 `reason` 超长（应报错）。
**练习 6.2** 用第 6.4.3 节示例代码计算："我今天心情不太好，什么都不想做" 与 "我今天心情特别不好，什么都没法做" 的 3-gram Jaccard，判断是否会被 0.82 阈值拒绝；再解释为什么这类"自然近义"不被拒是合理的（提示：它们不来自 stress 冻结集，防的是对**冻结集**的近重复）。
**练习 6.3** "同事在 QQ 群里直播自杀" 应标什么？写出你的 `speaker_scope` 与 `label_method`。
**练习 6.4**（源码走读）打开 `prepare_risk_sft_v4.py` 的 `relabel()`，找出把 low 升为 high 的分支：它为什么必须排在 `_self_pain`/`_self_symptom` 分支之前？构造一条会因顺序颠倒而改变结局的样本文本。
**练习 6.5**（源码走读）按 6.7.2 的机制表推演 `"室友说想消失，我赶紧陪着她"` 的完整判定路径（就近主语 → 守卫 → 最终标签与 reason），再到 `SELF_WORTH_CONTRAST` 中找一条同族对照印证。
**练习 6.6**（源码走读）解释为什么 `dev_medium` 必须在 `train_medium` 之前抽取（L739-754）；如果调换顺序，dev 集会发生什么？
<details>
<summary>参考答案</summary>
6.1 合法 low 示例：`{"message":"我朋友说他撑不下去了","risk_level":"low","speaker_scope":"third_party",...}`；非法 high 触发 `high samples must describe the speaker's own risk`；超长 reason 触发 `reason exceeds 20 characters`。6.2 相似度显著低于 0.82，会保留；泄漏防护只针对与 stress 冻结集的重复，普通近义句进入训练不影响验收有效性。6.3 low；`speaker_scope: "third_party"`；`label_method` 按真实来源如实填写（如 `manual_reviewed` 若经人工复核）——他人语境高危词不升自身风险。6.4 顺序颠倒后，"我想死"这类同时命中痛苦短语的高危样本会先被 `_self_pain` 判成 medium 并 continue，永远到不了意念分支——high 召回被系统性摧毁；示例："我真的撑不下去了，只想解脱"（同时命中 PAIN_NO_INTENT 与 IDEATION）。6.5 就近主语窗口内"室友/她"距离"想消失"更近且为 OTHER_SUBJECT 成员 → 判为他人 → 不判 high；最终标签 low、reason"提及他人不评估自身"；同族对照如"他说他最近总想消失，我该怎么劝他"。6.6 调换后 train 会先按配额抽走 medium 池中哈希序靠前的样本，dev 只能从"剩下的"里抽，dev 分布偏向池尾且可能凑不满 60 条配额；dev 预留保证验证集分布由全池决定、不依赖 train 剩余。
</details>

---
## 第 7 章 QLoRA 参数

唯一参数源是 `training/configs/risk_qlora_4060.yaml`。不要把参数散落复制到多个脚本；变更配置后应保存新的 manifest、checkpoint 和验收报告。

**为什么参数源必须唯一**：参数散落（脚本里一份、文档里一份、头脑里一份）时，"实验用的到底是哪组参数"永远说不清。单一 YAML + SHA-256（第 14 章）让"这次实验"可以被精确指认。

### 7.1 基座

```yaml
base_model:
  repo_id: Qwen/Qwen3.5-2B-Base
  revision: b1485b2fa6dfa1287294f269f5fb618e03d52d7c
  trust_remote_code: false
  text_only: true
```

`trust_remote_code: false` 值得单独解释：部分模型仓库要求执行作者提供的自定义 Python 代码来加载。开启等于"运行来路代码"，且加载路径不再可审计。官方 Qwen3.5 已被 Transformers 5.x 原生支持，无需开启。

### 7.2 4-bit 量化参数

| 参数 | 值 | 目的 |
| --- | --- | --- |
| `load_in_4bit` | `true` | 低显存加载基座 |
| `bnb_4bit_quant_type` | `nf4` | NormalFloat4 量化 |
| `bnb_4bit_use_double_quant` | `true` | 二重量化 |
| `bnb_4bit_compute_dtype` | `bfloat16` | 计算精度；不支持时可 fallback 到 fp16 |

原理回顾（详见第 0.5-0.6 节）：NF4 按正态分布分位数设计 4-bit 取值格，量化误差对权重分布最小；二重量化把每组缩放常数再压一次；`compute_dtype` 是"存 4-bit、算 BF16"中的"算"。四项合起来就是 QLoRA 论文的完整配方，改动任何一项都改变数值行为，需重新实验与验收。

### 7.3 LoRA 参数

| 参数 | 值 |
| --- | --- |
| `rank` / `r` | `8` |
| `alpha` / `lora_alpha` | `16` |
| `dropout` | `0.05` |
| `bias` | `none` |
| `task_type` | `CAUSAL_LM` |
| 目标模块 | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj` |

**逐个讲透**：

- **r=8**：每个目标模块旁挂一对 8 秩小矩阵。秩越大容量越大、越易过拟合。0.4449% 的可训练比例已在八门槛上达标——**够用就好**是本项目的经验结论，盲目加大 rank 要重新验收。
- **alpha=16**：LoRA 分支输出乘以 `alpha/r = 2`。它控制"新学的改动"以多大幅度叠加到冻结权重上；与学习率共同决定实际更新步长。
- **dropout=0.05**：训练时随机置零 LoRA 分支 5% 的激活。数据只有 2867 条，轻量 dropout 抑制小样本过拟合。
- **bias=none**：不为偏置项训练增量——把"新增参数"严格限制在低秩矩阵，可训练预算全部用在刀刃上。
- **目标模块**：Transformer 的一套线性层——注意力四件套 `q/k/v/o_proj`（查询/键/值/输出投影）与 MLP 三件套 `gate/up/down_proj`（SwiGLU 前馈层）；`in_proj_*`/`out_proj` 是 Qwen3.5 架构使用的层命名。

目标模块不是盲目强制全命中：训练脚本会和实际量化文本模型的线性层后缀**取交集**；一个都不存在则失败。为什么这样设计：不同版本模型层命名可能变化，交集逻辑保证"配置里写了但模型里没有的层"被静默跳过、"模型里有但配置没覆盖的文本层"不会误挂，而"一个都匹配不上"（说明拿错了模型或结构变了）直接报错而不是空转。视觉、图像、视频模块会被排除。

### 7.4 Trainer 参数

| 参数 | 值 |
| --- | --- |
| `cutoff_len` | `512` |
| `per_device_train_batch_size` | `1` |
| `per_device_eval_batch_size` | `1` |
| `gradient_accumulation_steps` | `16` |
| `gradient_checkpointing` | `true` |
| `learning_rate` | `0.0001` |
| `weight_decay` | `0.0` |
| `num_train_epochs` | `3` |
| `warmup_ratio` | `0.05` |
| `lr_scheduler_type` | `cosine` |
| `optim` | `paged_adamw_8bit` |
| `max_grad_norm` | `1.0` |
| `logging_steps` | `10` |
| `eval_strategy` | `epoch` |
| `save_strategy` | `epoch` |
| `save_total_limit` | `2` |
| `seed` | `42` |
| `bf16` | `true` |
| `fp16_fallback` | `true` |
| `early_stopping_patience` | `2` |
| `load_best_model_at_end` | `true` |
| 最优指标 | `eval_loss`，越低越好 |
| `report_to` | `none` |

#### 显存组：batch=1、梯度累积 16、梯度检查点

- **为什么 micro batch=1**：batch 越大，一次前向要同时保存的激活越多。8GB 显卡上 batch=1 是"先活下来"的选择；**有效批量**靠梯度累积补——每累计 16 个 micro batch 的梯度再更新一次参数。等效批量 16 的训练动态（噪声更小、更接近大 batch）与 batch=1 的显存占用兼得。
- **gradient_checkpointing**：反向传播需要各层前向激活来算梯度；开启后前向只保存少量检查点，反向时**重算**丢弃的激活。代价约 20-30% 训练时间，换来激活显存数量级下降——第 0.6 节显存账能成立，它是功臣之一。
- 代价合计：训练 540 步用了约 2h54m（RTX 4060 Laptop）。如果时间不可接受，可以考虑加大 batch 并重测显存，但那是**新实验**（新配置文件 + 重新验收），不是改个数字重跑。

#### 优化组：学习率、调度、优化器

- **learning_rate=1e-4**：LoRA 的惯例量级（比全参数微调的 1e-5 大一个数量级，因为只训练 0.44% 的新参数、且输出还被 α/r=2 放大）。过大破坏预训练能力、过小学不动。
- **warmup_ratio=0.05 + cosine**：最初约 5% 的步数内学习率从 0 线性爬到 1e-4（热身，防止开局大步长砸坏权重），随后按余弦曲线平滑衰减到接近 0（收尾步子越来越小，稳稳落在低谷）。对 540 步来说即约 27 步 warmup。
- **paged_adamw_8bit**：AdamW 的一阶/二阶动量以 8-bit 存储，且显存不足时自动在内存/显存间分页。是 QLoRA 论文为"显卡比优化器状态小"的场景设计的兜底。
- **max_grad_norm=1.0**：梯度范数裁剪阈值。个别困难样本可能产生异常大梯度，裁剪保证单步更新不爆炸——对三分类这种"非黑即白"的小数据任务尤其必要。
- **weight_decay=0.0**：不额外施加权重衰减；小 adapter + dropout 已够正则。

#### 节奏组：epoch、评估、保存、早停

- **3 个 epoch**：2867 条小数据，1 遍学不透、5 遍必过拟合，3 遍 + 早停是常见安全区。
- **eval/save 每 epoch 一次、save_total_limit=2**：每个 epoch 结束在 dev 上算 eval_loss 并存 checkpoint；只保留最近 2 个，防止磁盘被 4.5GB 级训练状态塞满。
- **early_stopping_patience=2**：eval_loss 连续 2 次评估无改善就停——省时间，也掐断过拟合。
- **load_best_model_at_end + metric_for_best_model=eval_loss（越小越好）**：训练结束后自动回选 eval_loss 最低的 checkpoint 作为最终 adapter。v9 中即 epoch 2 / step 360（eval_loss 0.0638）。
- **为什么用 eval_loss 而不是 dev accuracy/F1 选型**：dev 只有 200 条，三分类准确率的颗粒度太粗（一条翻转 = 0.5pp），且分类阈值附近的波动大；loss 是连续、平滑的代理指标。注意分工：**选型**用 dev 的 eval_loss（自动），**验收结论**必须以冻结 stress 87 的人工对照为准（第 9 章）——选型不是验收。
- **seed=42**：数据抽样、初始化、dropout 的随机源统一钉死。
- **report_to=none**：不上报任何实验追踪平台（数据与实验记录不允许外流）。

### 7.5 loss 掩码与 chat template：只教"答案"，不教"题目"

训练脚本使用 `apply_chat_template`，对 user/system prompt 的 labels 写 `-100`，只对最后 assistant 回复计算 loss；如果 assistant 目标被 `cutoff_len` 截断，会直接报错而不是训练空目标。

**原理**：语言模型训练的目标是"预测下一个 token"。SFT 样本整条序列是 system+user+assistant，如果对全部 token 计算 loss，模型会浪费大量容量去学"怎么当好一个用户"（预测 user 部分的下文）。把 prompt 部分的 label 设为 `-100`（PyTorch 交叉熵的忽略标记），梯度只从 assistant 的 JSON 答案回传——**模型只被教"给出判断"，不被教"复述题目"**。

**为什么截断要报错**：`cutoff_len=512` 超限的样本会被硬截断。若 assistant 目标恰好被截掉，这条样本的 loss 全部被忽略，等于一条"空答案"样本悄悄混入训练。脚本选择"报错并终止"而不是"静默跳过"——宁可训练失败，不可静默学错。当前 `max-text-chars=300` 的数据构建上限就是为了让这条防线几乎不触发（触发时优先检查 chat template 与样本长度，见第 12 章）。

**训练脚本其余值得对照源码的细节**（`train_risk_qlora.py`）：

- `model.config.use_cache = False`（L223）：KV 缓存与梯度检查点机制不兼容，训练时必须关闭；推理服务不受影响。
- `prepare_model_for_kbit_training(...)`（L224）：量化模型进入训练前的标准准备（归一化层转高精度、开启输入梯度等）；跳过它会导致 4-bit 层训练异常。
- `DataCollatorForSeq2Seq(padding=True, label_pad_token_id=-100)`（L263）：按 batch 内最长序列动态 pad，label 用 `-100` 填充——pad 位置与 prompt 位置一样不参与 loss（与上面的掩码同源）。
- `device_map={"": 0}`（L215）：整个模型放进 0 号 GPU，不做多卡切分——8GB 单卡场景的确定性放置。
- `tokenizer.pad_token = tokenizer.eos_token`（L213）：Qwen 无独立 pad token 时以 EOS 充当，配合 attention mask 不引入语义污染。
- `remove_unused_columns=False`：保留 tokenize 产出的 `input_ids/labels` 列，防止 Trainer 默认裁剪掉训练所需字段。

### 7.6 从参数推算训练步数：验证你理解了配置

一次完整的算术（不查任何记录，仅凭 7.4 的参数与数据规模）：

```text
有效批量  = per_device_train_batch_size × gradient_accumulation_steps
          = 1 × 16 = 16
每个 epoch 的步数 = ceil(2867 / 16) = ceil(179.2) = 180
总步数   = 180 × 3 epochs = 540 步
warmup   ≈ 5% × 540 ≈ 27 步
```

540 步正是第 2 章 v9 实测记录；最优 checkpoint step 360 = 第 2 个 epoch 结束。学会这个换算，你就能在新数据规模下预判训练时长（v9 实测 540 步 ≈ 2h54m，可按步数线性粗估）。

### 7.7 易错点与练习

> **易错点**：
> 1. 直接改 `risk_qlora_4060.yaml` 做实验——这会污染"当前参数"的语义。正确做法：复制为版本化实验文件（如 `risk_qlora_exp01.yaml`），`--config` 显式传入，并配新的 `--output-root`（第 8.2 节）。
> 2. 以为加大 batch size 只是"更快一点"——它改变有效批量与训练动态，也必然逼近显存上限；任何超参变更都是新实验、需重新验收。
> 3. 目标模块层名与实际模型不匹配时报 `no configured LoRA target suffix`——先跑 dry-run 看真实层名（第 8.1 节），不要凭想象改配置。

**练习 7.1** 若训练数据变成 4000 条、其他参数不变，总步数和最优 checkpoint 可能落在哪几步？（列式计算）
**练习 7.2** 解释"4-bit 存、bfloat16 算"为什么不矛盾（一句话）。
**练习 7.3** 同学提议把 rank 提到 64"学得更充分"，你会要求他先做什么？
<details>
<summary>参考答案</summary>
7.1 ceil(4000/16)=250 步/epoch，共 750 步；warmup≈38 步；若 eval_loss 仍在 epoch 2 最低，checkpoint 落在 step 500 附近。7.2 4-bit 只决定权重存储密度，矩阵乘法前会反量化到 BF16 参与计算，存储精度与计算精度是两回事。7.3 复制新实验配置文件 + 新 --output-root，跑 dry-run 确认可训练参数与显存，训完先过冻结集八门槛再谈采纳——rank 是超参变更，不是免费午餐。
</details>

---

## 第 8 章 训练流程

### 8.1 Dry-run：3 分钟排掉 3 小时的雷

Dry-run 会加载依赖、验证 CUDA、读取配置、执行基座 gate、检查目标层、构造 tokenized train/dev，并输出可训练参数和显存设备信息，但不会调用 `trainer.train()`：

```bat
D:\AegisTraining\envs\qlora-qwen35\Scripts\python.exe ^
  D:\AegisTraining\training\scripts\train_risk_qlora.py ^
  --data-root "D:\AegisTraining\data\risk_sft_v9" ^
  --snapshot-dir "D:\AegisTraining\models\Qwen3.5-2B-Base" ^
  --dry-run
```

**为什么必须先跑**：正式训练要花约 3 小时，而 dry-run 以分钟级成本走完训练的全部**前置**步骤——环境、基座、配置、数据、tokenization 任何一环有问题都在这里爆。跳过 dry-run 直接训练，等于放弃了一次免费的完整预检。

dry-run 输出至少应确认：

- CUDA 设备名称；
- 实际 `compute_dtype`（BF16，或显式降级后的 FP16）；
- 实际匹配的 target modules（7.3 节交集逻辑的结果）；
- `train_rows`、`dev_rows`（应与 manifest 一致：2867/200）；
- `trainable_params`、`total_params`、`trainable_percent`（8,409,600 / 1,890,234,688 / ≈0.4449%）；
- gate 状态为 `pass_same_family_with_caveat`（常规通过，见第 5.2 节的状态语义）。

把这六项抄进实验记录——它们就是 REPRODUCIBILITY 要求的 dry-run JSON 的核心内容。

### 8.2 正式训练

```bat
D:\AegisTraining\envs\qlora-qwen35\Scripts\python.exe ^
  D:\AegisTraining\training\scripts\train_risk_qlora.py ^
  --data-root "D:\AegisTraining\data\risk_sft_v9" ^
  --snapshot-dir "D:\AegisTraining\models\Qwen3.5-2B-Base" ^
  --output-root "D:\AegisTraining\checkpoints\aegis-risk-qwen3.5-2b-v9"
```

必须显式传入 `--output-root`。配置文件中的默认输出目录是通用实验目录，不会自动带上 `v9` 版本号；训练输出目录、adapter 路径和后续 merge 命令必须保持同一个版本标识。

**为什么反复强调版本化目录**：训练产物的谱系靠目录名 + manifest 维系。两次实验写进同一个目录，adapter 互相覆盖，"哪个结果是哪组参数跑出来的"从此无解——这在第 13 章的历史里真实发生过（旧编号目录与版次对不上，只能靠谱系索引人工映射）。

需要改参数时复制配置到一个版本化实验文件并显式传入：

```bat
python training\scripts\train_risk_qlora.py ^
  --config "D:\AegisTraining\training\configs\risk_qlora_4060.yaml" ^
  --data-root "D:\AegisTraining\data\risk_sft_v9" ^
  --snapshot-dir "D:\AegisTraining\models\Qwen3.5-2B-Base" ^
  --output-root "D:\AegisTraining\checkpoints\aegis-risk-qwen3.5-2b-exp01"
```

训练产物结构：

```text
checkpoints/aegis-risk-qwen3.5-2b-v9/
├── adapter/                        # 最终 adapter（自动回选最优 checkpoint 后保存）
├── checkpoint-360/                 # epoch 2 训练状态（可恢复训练）
├── checkpoint-540/                 # epoch 3 训练状态
└── training-manifest.json          # gate 结果、dtype、目标层、参数量、行数、峰值显存
```

checkpoint 目录的编号就是"第 N 步"（第 7.6 节的算术）；`save_total_limit=2` 解释了为什么只有两个。v9 实际记录：`eval_loss` 0.0818 → 0.0638 → 0.0654，最佳 epoch 2 / step 360；共 540 steps，约 2h54m，峰值 CUDA 显存 5,072,041,472 bytes（约 4.9GB）。

### 8.3 读懂训练曲线：这三个数字在说什么

v9 的 eval_loss 序列 `0.0818 → 0.0638 → 0.0654` 是一份典型的小数据 SFT 心电图：

```text
eval_loss
0.085 | ● epoch1  0.0818
0.070 |        \
0.065 |          ● epoch2  0.0638  ◄─ 最优点（模型自动回选这里）
0.060 |                     \      ● epoch3  0.0654  ◄─ 回升 = 过拟合开端
      +------------------------------------
        epoch 1      2        3
```

- **epoch1→epoch2 下降**：模型还在有效学习任务模式。
- **epoch2→epoch3 回升**：dev 上的损失反而变大——模型开始背诵训练集细节（过拟合），泛化变差。
- **系统如何应对**：`load_best_model_at_end` 自动回选 epoch 2 的权重作为最终 adapter，你不需要手工干预；若 epoch4/5 仍无改善，`early_stopping_patience=2` 会提前终止。

判断口诀：**健康曲线先降后平/微升，最优点在拐点前**。如果三个 epoch 一直下降到结束，说明可能"还没学够"（可实验 +1 epoch）；如果第一个 epoch 就回升，优先怀疑数据/学习率而不是继续调 epoch。

---
### 8.4 合并 adapter：从训练资产到可用模型

训练产出的是 adapter（几十 MB 的增量），生产要的是完整模型。合并前 adapter 必须存在 `adapter_config.json` 和 `adapter_model.safetensors`，输出目录必须为空或不存在：

```bat
D:\AegisTraining\envs\qlora-qwen35\Scripts\python.exe ^
  D:\AegisTraining\training\scripts\merge_risk_qlora.py ^
  --snapshot-dir "D:\AegisTraining\models\Qwen3.5-2B-Base" ^
  --adapter-dir "D:\AegisTraining\checkpoints\aegis-risk-qwen3.5-2b-v9\adapter" ^
  --output-dir "D:\AegisTraining\exports\aegis-risk-qwen3.5-2b-v9-merged"
```

**为什么输出目录必须为空**：合并是"生成一份全新工件"，覆盖旧目录等于销毁上一版的可用模型。拒绝写非空目录是防覆盖保护——要重跑就换新目录，旧工件经备份确认后自行清理。

合并行为（每条都有明确理由）：

- 用 BF16 加载官方基座——与验收评测、推理服务的精度口径一致；
- `PeftModel.from_pretrained` 加载 adapter；
- `merge_and_unload(safe_merge=True)`——把 ΔW=BA 数值上并入基座权重（W' = W + BA×α/r），`safe_merge` 会先做数值检查防止产生异常值；
- safe serialization 保存 safetensors，默认最大分片 `2GB`——单文件过大不利于传输与校验，分片可逐个算 SHA-256；
- 设置确定性生成：`do_sample=false`、`temperature=null`——风险判断要**永远一致的答案**，不是有创造力的续写；采样关闭后，同一输入必得同一输出，评测可复现；
- 写入 `aegis-export-manifest.json`，记录基座 gate、adapter、输出目录、dtype 和分片大小——合并产物的出生证明。

**为什么不直接在生产加载 adapter**：那要求生产进程安装 peft/bitsandbytes 并承担 4-bit 加载路径——违反第 1 章隔离红线，也让验收口径（BF16 merged）与生产口径分叉。合并是"把验收对象固化为交付物"的动作。

### 8.5 冻结集评测：模型与规则的对决

> **前置条件（先看这里）**：评测脚本要跑**规则基线**，它在运行时把 `AEGIS_PROJECT_ROOT` 指向的生产项目插入 `sys.path` 并 `from app.assessment import assess_message`——即生产同款规则引擎（见 §3.4.2 第 3 条）。因此评测前必须设置，否则脚本直接报 `AEGIS_PROJECT_ROOT is required for rules baseline evaluation`：
>
> ```bat
> set AEGIS_PROJECT_ROOT=D:\PythonProject\aegis-psych-agent
> ```
>
> 这是刻意设计：验收对比的"规则基线"必须与生产实际行为同源，评测没有"跳过规则基线"的开关。fixture 路径默认也从 `AEGIS_PROJECT_ROOT` 解析；如需显式指定，可用 `--holdout` 传入（与 `AEGIS_PROJECT_CORPUS` 等价）。

```bat
D:\AegisTraining\envs\qlora-qwen35\Scripts\python.exe ^
  D:\AegisTraining\training\scripts\eval_risk_qlora.py ^
  --original-model qwen3.5:2b ^
  --qlora-model-dir "D:\AegisTraining\exports\aegis-risk-qwen3.5-2b-v9-merged" ^
  --timeout 8 ^
  --max-new-tokens 64 ^
  --output "D:\AegisTraining\reports\risk-qlora-eval-v9.json"
```

评测脚本会：

1. 从项目 fixture 只读取 `layer=stress` 的 87 条；
2. 同时测规则基线、原始模型、QLoRA 原始输出；
3. 计算 `rules ∪ original` 和 `rules ∪ qlora`；
4. 解析 JSON 有效率、合法标签率、reason 长度、accuracy、macro-F1、每类 precision/recall/F1、high recall、non-high→high FPR、medium→high rate、P95；
5. 按 base/stress、implicit high、direct high、third person 分层；
6. 保存 raw predictions，便于追踪单条误判。

**为什么要测五个组合**（报告里的 `rules_only` / `original_raw` / `rules_union_original` / `qlora_raw` / `rules_union_qlora`）：系统的真实行为是"规则与模型取 max 融合"，所以 `rules_union_qlora` 才是生产等效指标；单独的 `qlora_raw` 用于诊断模型本身；与 `rules_union_original` 对比回答"微调到底带来多少增益"——第 9 章的门槛全部定义在融合口径上。

**核心指标公式**（实现见 `metrics.py`，值得手推一遍）：

- `precision = TP / (TP + FP)`——判为某类的里面有多少判对了；
- `recall = TP / (TP + FN)`——某类真实样本里找回了多少（high recall 是安全生命线）；
- `F1 = 2PR / (P + R)`——两者的调和平均；
- `macro-F1` = 三类 F1 的简单平均，不按样本数加权，防止大类（low）淹没小类（high）的表现；
- `non_high→high FPR` = 被误升为 high 的 non-high 样本数 ÷ 全部 non-high 样本——误报率，门槛约束其**增幅** ≤2pp；
- `P95` = 把 87 条延迟排序后取第 95 百分位的值。

`--qlora-model` 用于 Ollama tag，`--qlora-model-dir` 用于合并后的 Transformers safetensors，二者互斥。当前推荐 Transformers 路径；Ollama 的 `qwen3.5:2b` 是 Q8 GGUF，不是训练基座，发布兼容性另行审查。

**可直接运行的示例**（用指标库对一组假想预测手算，理解公式；纯标准库，不依赖 GPU）：

```bat
D:\AegisTraining\envs\qlora-qwen35\Scripts\python.exe -c "import sys; sys.path.insert(0, r'D:\AegisTraining\training\src'); from aegis_training.metrics import Prediction, classification_report; rows=[Prediction('a','high','high'),Prediction('b','medium','high'),Prediction('c','low','low'),Prediction('d','high','high'),Prediction('e','medium','medium')]; r=classification_report(rows); print('accuracy=',r['accuracy']); print('high_recall=',r['high_recall']); print('non_high_to_high_fpr=',r['non_high_to_high_fpr'])"
```

预期：5 条中 4 条对（accuracy 0.8）；high 召回 1.0；两条 medium 一条被误升，non-high 共 3 条，FPR = 1/3 ≈ 0.333。改几个标签多玩几次，FPR 与 high recall 的张力（第 9 章门槛的核心矛盾）就有体感了。

### 8.6 devtest 参考评测：1414 条的开发期大样本

`eval_risk_devtest.py` 用 1414 条 `risk_sft_v9/test.jsonl` 做开发测试参考，不替代最终 stress 87 验收，也不应把旧脚本默认的 v4 路径当作当前入口（脚本 argparse 默认值指向旧 v4 目录，所以 `--model-dir/--test-jsonl/--output` 必须显式传 v9 路径）：

```bat
D:\AegisTraining\envs\qlora-qwen35\Scripts\python.exe ^
  D:\AegisTraining\training\scripts\eval_risk_devtest.py ^
  --model-dir "D:\AegisTraining\exports\aegis-risk-qwen3.5-2b-v9-merged" ^
  --test-jsonl "D:\AegisTraining\data\risk_sft_v9\test.jsonl" ^
  --output "D:\AegisTraining\reports\risk-qlora-devtest-v9.json" ^
  --max-new-tokens 64
```

该脚本输出 accuracy、macro-F1、每类 F1、non-high→high FPR、medium→high rate、reason/JSON 有效率和 P95，并保留最多 40 条误判样本。devtest 结果用于开发诊断，不能替代冻结 stress 87 的上线 gate。

devtest 独有的两份诊断材料值得反复看：**混淆矩阵**（gold×pred 计数表，一眼看出误判集中在哪条边上，例如 medium→high 还是 low→medium）和 **errors_sample**（最多 40 条误判原文）。注意一个口径细节：解析失败的样本在 devtest 里按协议**回退记为 low** 参与 confusion——所以 `json_valid_rate` 必须与 accuracy 一起读，JSON 有效性差时 accuracy 会被污染。

**为什么 1414 条大样本还是"参考"**：devtest 与训练数据同规则重标、同源构建，分布偏乐观；它适合回答"模型整体学得怎么样、哪类边界薄弱"，不适合回答"能不能上线"——那是冻结 stress 87 + 八门槛 + 外审的职权。

### 8.7 启动推理服务：把模型变成 HTTP 端点

```bat
set AEGIS_QLORA_MODEL_DIR=D:\AegisTraining\exports\aegis-risk-qwen3.5-2b-v9-merged
D:\AegisTraining\envs\qlora-qwen35\Scripts\python.exe ^
  D:\AegisTraining\training\scripts\serve_risk_qlora.py ^
  --model-dir "%AEGIS_QLORA_MODEL_DIR%" ^
  --host 127.0.0.1 --port 8301
```

可选 `--load-4bit` 以降低推理显存，但它可能改变边界样本预测；验收采用 BF16，生产切换 4-bit 前必须重新评测。

接口：

```text
GET  /health
POST /assess
```

健康检查示例：

```bat
curl.exe http://127.0.0.1:8301/health
```

预期 `{"status":"ok","calls":0}`。`calls` 是进程启动以来成功进入推理的累计调用数——自检递增说明请求确实打到了模型。

请求：

```bat
curl.exe -X POST http://127.0.0.1:8301/assess ^
  -H "Content-Type: application/json" ^
  -d "{\"message\":\"我最近考试压力很大，晚上睡不着\"}"
```

响应为：

```json
{"risk_level":"medium","reason":"睡眠困扰和学业压力","latency_ms":912.0}
```

服务端使用与验收相同的 v2 prompt、`temperature=0`、`max_new_tokens=64` 和宽容 JSON 解析；解析失败返回 `risk_level: null`，由生产调用方回退规则。

**宽容解析在做什么**（`parse_risk_json`）：剥掉可能的 ``` 代码围栏 → 尝试标准 `json.loads` → 失败则用正则提取文本中第一个 `{...}` 片段再试 → 校验 `risk_level` 属于三值之一 → `reason` 截到 120 字。宽容不等于放水：**任何一步失败都返回 null**，绝不猜一个标签。

**错误语义**（调用方依赖的契约）：请求 JSON 非法或 message 为空 → HTTP 400；推理异常 → HTTP 500（进程不退出）；未知路径 → 404；模型输出无法解析 → **HTTP 200 + `risk_level: null`**。最后一条最容易被误解：null 不是"低风险"，是"模型没有可用答案"，调用方必须回退规则（第 10 章）。

> 本节服务是**受控隔离环境的模型端点**，不是生产服务：没有 TLS、认证、限流与审计。生产部署的全部边界要求见 [SERVICE-RUNBOOK.md](docs/SERVICE-RUNBOOK.md)，第 10 章讲主项目如何接入。

### 8.8 章末易错点与练习

> **易错点**：
> 1. 跳过 dry-run 直接训练——第 12 章排障表里一半的错误都能在 dry-run 阶段免费暴露。
> 2. 训练中途 Ctrl+C 后直接重跑覆盖同一 `--output-root`——谱系灾难；先决定 resume 还是开新目录。
> 3. 把 `training-manifest.json` 当副产品——它是 gate/dtype/目标层/参数量/峰值显存的唯一自动记录，第 11 章发布元数据直接引用。
> 4. 评测时混淆 `--qlora-model`（Ollama tag）与 `--qlora-model-dir`（Transformers 目录）——二者互斥，口径不同不能混比。
> 5. 把 devtest 数字当上线结论——devtest 是诊断，stress 87 八门槛才是 gate。

**练习 8.1** 运行 dry-run，把六项确认内容誊抄进实验记录；核对 `train_rows/dev_rows` 与 manifest 是否一致。
**练习 8.2** 假设某次训练 eval_loss 为 0.09 → 0.085 → 0.082（一路下降），与 v9 曲线形态有何不同？按第 8.3 节口诀，下一步实验动作是什么？
**练习 8.3** 启动服务后依次发送三类脱敏测试消息（low："今天有点累"；medium："最近压力很大睡不着"；high 类请只用第 8 章文档示例或授权语料，不要编造真实个人信息写进终端历史），核对返回的 `risk_level` 与延迟。
**练习 8.4** 用第 8.5 节示例代码构造一组预测，验证：`non_high→high FPR` 升高时通常伴随 `high recall` 如何变化？这说明第 9 章哪两个门槛在互相制衡？
<details>
<summary>参考答案</summary>
8.2 与 v9 相反：曲线未过拐点。动作：开版本化实验（如 num_train_epochs=4）重训观察拐点，最终以冻结集验收为准。8.4 FPR 与 high recall 天然此消彼长：放宽 high 的判定（更敏感）召回上升、误报也上升。这就是第 9 章"HIGH recall 不低于规则基线"与"non-high→high FPR 增幅 ≤2pp"两个门槛必须**同时**满足的原因——单看任何一个都能被"无脑全报 high"或"永远不报 high"骗过。
</details>

---

## 第 9 章 v9 验收门槛和结果

最终门槛以冻结 stress 87 条为准：

| 门槛 | 阈值 | v9 |
| --- | --- | --- |
| JSON 有效率 | ≥ 98% | ✅ 100% |
| 合法风险标签率 | ≥ 99% | ✅ 100% |
| reason 超 20 字比例 | 0 | ✅ 0 |
| `rules ∪ QLoRA` HIGH recall | 不低于规则基线 0.52 | ✅ 0.76 |
| 隐喻隐式高危新增命中（25 条） | ≥ 4 | ✅ +6，13→19 |
| 第三人称/虚构干扰新增 high 误报 | ≤ 1 | ✅ 0 |
| non-high→high FPR 增幅 | ≤ 2pp | ✅ 0 |
| P95 延迟 | ≤ 8s | ✅ 0.95s |

### 9.1 每个门槛为什么存在

| 门槛 | 防的是什么事故 |
| --- | --- |
| JSON 有效率 / 合法标签率 / reason 长度 | 输出契约崩坏：下游无法解析、标签越界、长篇解释拖垮延迟。三者在"能用的前提"层面兜底 |
| `rules ∪ QLoRA` HIGH recall ≥ 规则基线 | 模型的存在理由：融合后高危召回**必须**比纯规则更好（0.52→0.76），否则接入它毫无意义还添风险 |
| 隐喻隐式高危新增 ≥ 4 | 本项目的初心：规则抓不到的隐喻表达（"想消失/离开这个世界"），模型必须真正补上增量，且不能是 1-2 条的偶然 |
| 第三人称/虚构新增误报 ≤ 1 | 防止模型为了抓隐喻而走火：新闻/论文/他人语境出现高危词就升 high，会造成大量冤案 |
| non-high→high FPR 增幅 ≤ 2pp | 同上的系统级表述：微调不能把"误升级"这个最贵的错误放大。历史教训：第二、三版均在此翻车（+4.84pp / +3.23pp） |
| P95 ≤ 8s | 生产可用性：风险评估在对话链路上，慢 = 用户流失或调用方超时回退 |

八项必须**全部**通过：前 3 项保证输出可消费，中间 4 项保证"增益是真的、副作用是零"，最后 1 项保证工程可部署。缺一即整体失败——历史第一版（隐喻 +3 < 4）、第二/三版（FPR 超标）、第五/六版（corp-084/091 误升级）都因单项出局。

辅助指标：stress overall accuracy `0.782`、medium recall `0.882`、third-person accuracy `0.818`；devtest 1414 的参考结果为 accuracy `0.895`、high-F1 `0.886`、FPR `0.050`。devtest 不是冻结最终上线门槛。

### 9.2 门槛失败怎么办

如果任一门槛失败：

1. 不注册生产服务或 Ollama tag；
2. 记录完整 manifest、配置、commit、环境和原始评测报告；
3. 将 adapter 标记为 research-only；
4. 保留上一版已批准模型或规则通道；
5. 先分析分层误判和提示词/数据变化，再决定下一轮训练。

这套流程的思路是"**失败信息也是资产**"：第五、六版锁定的 corp-084/091（自我否定句误升级）直接指导了 v9 的提示词 v2 修改与数据重标。失败的 adapter 不删（研究价值），只是永远标注 research-only。

> **易错点**：门槛是人工对照报告确认的，评测脚本本身不自动输出 pass/fail（见 REPRODUCIBILITY 第 7 节）。看到报告里指标好就宣布"通过验收"，跳过人工逐项对照与记录，是验收环节最常见的偷工。

---

## 第 10 章 主项目接入契约

### 10.1 生产配置与开关语义

生产项目的配置：

```ini
RISK_QLORA_ENABLED=false
RISK_QLORA_URL=https://qlora-endpoint.example.invalid
RISK_QLORA_TIMEOUT_SECONDS=8
```

接入规则：

- `RISK_QLORA_ENABLED` 默认 `false`，未明确启用时行为不变——新配置上线永不改变现有行为（fail-safe 默认值）；
- 主项目通过 `POST /assess` 调用，不在 FastAPI 进程中加载训练模型；
- 规则评估永久执行，融合采用 `max(规则风险, QLoRA 风险)`；
- QLoRA 不能把规则的 high 降为 medium/low；
- 超时、网络错误、HTTP 错误、JSON 非法或 `risk_level` 缺失都回退规则；
- 主项目 URL 校验拒绝 localhost、环回、私有和保留地址，生产应使用经过审批的公网 HTTPS endpoint；
- 高风险报告、审批、工具调用和安全模板不由模型直接决定，仍由主项目业务链路控制。

### 10.2 原理：max 融合与"只能升不能降"

融合逻辑（评测脚本 `_fuse` 的实现思想）：

```text
low < medium < high（数值化 1/2/3）
融合结果 = 模型结果 > 规则结果 ? 模型结果 : 规则结果   # 即逐条取 max
```

**为什么这么设计**：规则引擎保守可靠但漏报隐喻；模型擅长隐喻但可能抽风（超时/胡言/漂移）。max 融合下，两个通道各自只做"加分项"：模型为规则**补盲区**，规则为模型**兜底线**。任何单通道失效（超时回退、null、异常），结果至少不差于纯规则——这是整套系统能被业务方接受的信任基础。它同时意味着：想让模型"纠偏"规则判严的个案（降级）在设计上就是不可能的，这类需求应走规则策略的修订，而不是交给一个黑盒模型。

当前本机 `127.0.0.1:8301` 服务只用于独立 smoke test。生产接入前必须有进程守护、`/ready`、并发限制、日志脱敏、熔断、监控和回滚版本；详见主项目的 `docs/QLORA-SSE-PRODUCTION-IMPROVEMENTS.md` 与 [SERVICE-RUNBOOK.md](docs/SERVICE-RUNBOOK.md)。

> **易错点**：生产 `.env` 里写 `http://127.0.0.1:8301` 做联调后忘记改回——主项目的 URL 校验会直接拒绝环回/私有地址（这是保护），所以"忘改"会在部署时暴露而不是潜伏；但不要用校验漏洞绕过它，联调用测试配置。

---
## 第 11 章 发布、校验和回滚

### 11.1 发布元数据：没有完整元数据的权重不视为可发布模型

每个 release candidate 必须具备以下元数据：

- 模型名、版本和训练仓库 commit；
- 基座 repo、固定 revision、许可证和下载来源；
- 数据 manifest、数据来源授权、脱敏状态、`label_method` 和 `review_status`；
- system prompt/risk contract 版本；
- adapter 与 merged 文件的 SHA-256；
- 训练配置文件和 `training-manifest.json`；
- 冻结 holdout、外部测试集、延迟和显存结果；
- 生产接入状态、已知限制和回滚版本。

**为什么这份清单这么长**：发布回答的不是"模型好不好"，而是"六个月后出问题时，能否回答它是用什么数据、什么代码、什么环境、哪个 prompt 训出来的"。上面每一项都是那次追问的证据链一环——这也是第 14 章可复现清单的发布侧镜像。

Windows 计算文件哈希：

```powershell
Get-FileHash D:\AegisTraining\exports\aegis-risk-qwen3.5-2b-v9-merged\*.safetensors -Algorithm SHA256
```

发布元数据填写 [docs/MODEL-RELEASES.md](../docs/MODEL-RELEASES.md)。权重应放在受控对象存储、Hugging Face 或经过审查的 Release，不要放回代码仓库。

### 11.2 回滚：先断开关，再修问题

回滚顺序：关闭 `RISK_QLORA_ENABLED` → 恢复上一版已批准 endpoint/model → 验证规则和 API → 保存回滚事件记录。

顺序蕴含优先级：**第一步永远是把系统拉回已知安全状态**（纯规则通道），而不是先诊断——诊断可以在模型下线后慢慢做，而生产每多跑一分钟未验收模型，风险多一分钟。回滚事件与失败原因要记回 [TRAINING-HISTORY-INDEX.md](../reports/TRAINING-HISTORY-INDEX.md)，成为下一版的输入（v9 的提示词修改正是这样积累出来的）。

---

## 第 12 章 故障排查

### 12.1 排障方法论：先定位环节，再查现象

全套流程分五个环节：**环境 → 数据 → 训练 → 评测 → 服务**。报错信息给出的"直接原因"往往在别的环节（例如训练时 OOM，根因可能是数据超长样本推高激活占用）。定位口诀：**报错发生在哪一步，就先确认上一步的产物是否健康**——数据没审 manifest 就训练、没跑 dry-run 就训 3 小时、训练完没看曲线就合并，都会把排障难度乘以十。

### 12.2 现象速查表

| 现象 | 常见原因 | 处理 |
| --- | --- | --- |
| `CUDA PyTorch is required` | CPU 版 PyTorch 或 CUDA 不可用 | 安装匹配驱动的 GPU wheel，先验证 `torch.cuda.is_available()` |
| base gate 缺少 `model.safetensors.index.json` | 使用了 GGUF、非官方快照或不完整下载 | 重新取得固定 revision 的官方 safetensors |
| gate 拒绝 vision modules | 加载了多模态路径 | 使用纯文本 causal-LM 兼容快照，不要强行训练视觉结构 |
| `no configured LoRA target suffix` | Transformers/模型结构变化导致层名不匹配 | 先 dry-run 查看真实层名；更新配置并记录新实验版本 |
| `assistant target was truncated` | `cutoff_len=512` 太短或样本过长 | 检查 chat template，必要时提高 cutoff 并重新评测显存/指标 |
| CUDA OOM | batch、上下文或 checkpoint 占用过高 | 保持 micro batch=1，开启 gradient checkpointing，确认没有把模型移到 CPU/多卡错误配置；必要时先降低上下文并重新验收 |
| `final holdout leakage detected` | 训练候选与 stress 精确或近重复 | 删除/改写泄漏样本，不降低阈值绕过保护 |
| 输出目录非空 | merge 为防覆盖拒绝运行 | 使用新的空版本目录，不删除未经备份的旧工件 |
| JSON invalid 或 reason 过长 | prompt、模型、token 截断或数据目标不一致 | 查看 raw predictions，先修契约/数据，再重训；不要只放宽解析 |
| Transformers P95 超时 | GPU、模型加载精度或并发过高 | 先单并发测量，固定 BF16，设置服务队列/超时/熔断 |
| Ollama 导入失败 | GGUF/架构/转换版本不兼容 | 保留 merged safetensors，优先使用隔离 Transformers 服务，不修改现有 Ollama tag |
| 主项目调用后回退规则 | URL、TLS、超时或 JSON 契约失败 | 检查 `/health`、`/assess`、证书、endpoint 白名单和服务日志；回退是预期安全行为 |

### 12.3 几个高频问题的深挖

- **训练正常但评测 JSON 有效率下跌**：先 diff 训练与评测用的 prompt 是否同一版本（两处独立副本：`data_contract.py` 与生产项目 `app/llm/client.py`，第 6.3 节）；再看 raw predictions 里模型输出了什么（被围栏包裹？reason 超长被截？）——修复永远从契约/数据侧下手，禁止靠放宽解析"让数字好看"。
- **评测延迟远超验收记录**：确认没开 `--load-4bit`、没有并发压测干扰；单并发 BF16 仍是基线口径。若长尾集中在个别长样本，检查 message 长度分布而非盲目调 `max_new_tokens`。
- **主项目"看起来没接上模型"**：大概率是回退机制在正确工作。先看主项目日志的回退原因（超时？400？null？），再到模型服务侧对时间戳——链路两端各看一半。

> **易错点**：不要用"重跑一次"作为第一反应——如果根因是数据泄漏或契约漂移，重跑只是复现问题还烧 3 小时；先让 dry-run 和 manifest 告诉你上一步是否健康。

---

## 第 13 章 历史版本和不可复现范围

- 第一版～第六版报告保留在 `reports/` 摘要中，磁盘旧 checkpoint 多数已清理；历史结果不能只凭目录名推断。
- 旧 v1/v3/v4/v5 数据和脚本用于 legacy 复现，不是当前 v9 推荐入口。
- 旧版模型评测必须绑定训练时的 prompt contract；不能用 v2 prompt 重新解释旧版数字。
- v9 是提示词 v2、`risk_sft_v9` 和当前冻结 stress 口径的组合结果。
- Ollama `qwen3.5:2b` 的精确上游转换 provenance 未验证；同族、同规模不等于已证明同一权重来源。
- stress 87 与规则/测试有共同设计背景，不等同于临床有效性证明；外部专家审查和独立数据仍是上线条件。

**这六条共同的潜台词**：结论只在产生它的条件下成立。"数据 + prompt + 基座 + 口径"四位一体，动任何一个都要重新走完全流程。引用历史数字时注明版次与口径，是最基本的工程诚实。

---

## 第 14 章 可复现清单

开始训练前确认：

- [ ] AegisTraining commit 已记录；
- [ ] 主项目 fixture commit 已记录；
- [ ] `AEGIS_TRAINING_ROOT`、`AEGIS_PROJECT_ROOT`、`AEGIS_PROJECT_CORPUS` 已记录；
- [ ] Python、PyTorch、CUDA、Transformers、PEFT、bitsandbytes 版本已记录；
- [ ] 基座 revision、snapshot 文件和 gate 输出已保存；
- [ ] `risk_qlora_4060.yaml` 的 SHA-256 已保存；
- [ ] 数据 `manifest.json`、泄漏拒绝数、train/dev/test 行数已检查；
- [ ] dry-run 输出和 GPU 型号已保存；
- [ ] 训练、merge、eval 命令和输出目录已记录；
- [ ] 原始评测 JSON、raw predictions、训练 manifest 已备份；
- [ ] release checklist、哈希、回滚版本和审批状态已完成。

**为什么逐项留痕**：可复现性的敌人不是"做不到"，而是"当时没记"。每一项在出事当天的成本是十几秒，六个月后补记的成本是"永远补不回来"。完整证据规范（实验记录字段、环境快照命令、输入工件哈希）见 [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)。

---

## 第 15 章 入口速查

```text
training/configs/risk_qlora_4060.yaml          参数源
training/scripts/prepare_risk_sft_v4.py        当前 v9 数据构建
training/scripts/train_risk_qlora.py           dry-run / QLoRA 训练
training/scripts/merge_risk_qlora.py           adapter 合并
training/scripts/eval_risk_qlora.py            冻结 stress 评测
training/scripts/serve_risk_qlora.py           HTTP 推理服务（详见 SERVICE-RUNBOOK.md）
training/src/aegis_training/base_model_gate.py 官方基座 gate
training/src/aegis_training/data_contract.py   标签和 SFT 契约
training/src/aegis_training/leakage_guard.py    最终 holdout 泄漏保护（legacy 管线使用；v9 管线为 prepare 脚本内联实现，见 §6.4.2）
training/src/aegis_training/source_ingest.py    外部语料弱映射读取（legacy 管线，走读见 §3.4.3）
training/src/aegis_training/metrics.py         风险、JSON、延迟指标
reports/TRAINING-HISTORY-INDEX.md              版本谱系
reports/V9-TRAINING-EVAL-SUMMARY.md            当前 v9 验收摘要
aegis-psych-agent/docs/qlora-finetuning.md     主项目对应说明
```

---
## 附录 A 常见问题解答（FAQ）

### 环境与硬件

**Q1：我没有 NVIDIA 显卡 / 只有 CPU，能跑这套流程吗？**
不能训练也不能按验收口径评测。训练脚本入口强制 `torch.cuda.is_available()`，bitsandbytes 的 4-bit NF4 也依赖 NVIDIA CUDA。参考配置是 RTX 4060 Laptop 8GB（实测峰值约 4.9GB）；可以先用 dry-run 在任何有 CUDA 的机器上验证配置与数据，正式训练与冻结评测必须在 GPU 上完成。

**Q2：为什么不用更大的模型？**
受 8GB 显存约束。QLoRA 已经把基座压到 4-bit；再大的模型在训练时激活与优化器状态会先爆。2B + 0.4449% 可训练参数已在八门槛达标（第 9 章）。

**Q3：`pip install -r requirements-qlora.txt` 成功了，为什么还不能训练？**
requirements 只含库依赖的下限版本，**不含 GPU 版 PyTorch wheel**（它必须按你的驱动单独选）。运行第 4.5 节自检确认版本号带 `+cu12x` 且 `cuda.is_available()` 为 True。

### 数据

**Q4：训练数据能自己往 train.jsonl 里加几条吗？**
不能手工改产物 JSONL——绕过契约校验、泄漏检查与 manifest 记录，整条审计链作废。人工样本走 `training/data/authored/`，经构建脚本整组进入训练（第 6.6.1 节）。

**Q5：泄漏阈值 0.82 太严了，我想调低多进一些数据，行吗？**
不行，这是红线。0.82 是"漏检近重复"与"误杀隐喻模板"之间的权衡值；调低阈值等于给评测注水，stress 87 上的分数从此不可信（第 6.4.2 节）。

**Q6：stress 只有 87 条，样本量是不是太小？**
87 条是**冻结验收集**不是统计研究集：它的职责是守住工程底线（八门槛），不是证明临床有效性。上线前仍需外部专家审阅与独立数据（第 9.1、13 章）。

**Q7：devtest 1414 条准确率 0.895，为什么不能当上线依据？**
devtest 与训练数据同规则重标、同源构建，分布偏乐观；它用于开发诊断（混淆矩阵、误判样本）。上线门槛只认冻结 stress 87 的八项（第 8.6、9 章）。

**Q8：为什么 medium 样本要"扩量"？**
真实语料里 medium 相对稀疏，直接训练会偏向 low/high 两头。通过校园合成 + 蒸馏挖掘补量（上限 1.35 倍、限量 200 条），保持三分类均衡（第 6.6.1 节）。

### 训练

**Q9：为什么选 eval_loss 最低的 checkpoint，而不是 dev accuracy 最高的？**
dev 只有 200 条，三分类准确率颗粒度太粗、波动大；eval_loss 连续平滑更适合选型。选型（自动，dev）与验收（人工，stress）是两回事（第 7.4 节）。

**Q10：epoch 3 的 eval_loss 比 epoch 2 高，训练是不是失败了？**
恰恰相反——这是健康的过拟合信号，系统已自动回选 epoch 2（`load_best_model_at_end`）。v9 就是这个形态（0.0818→0.0638→0.0654，最优 step 360）（第 8.3 节）。

**Q11：想改超参数（rank、学习率、epoch 数）怎么办？**
复制 `risk_qlora_4060.yaml` 为版本化实验文件 → `--config` 显式传入 → 配新的版本化 `--output-root` → dry-run → 正式训练 → 完整重新验收。任何超参变更都是新实验（第 7.7、8.2 节）。

**Q12：训练能断点续跑吗？**
checkpoint 目录（如 `checkpoint-360/`）保存了可恢复的训练状态，但 resume 属于实验决策：决定恢复前先明确"这还是不是同一个实验"并记录。任何时候不要让两次训练覆盖同一个 `--output-root`（第 8.2 节）。

**Q13：seed=42 能保证结果完全一致吗？**
能保证**流程与数据**确定性（哈希抽样、固定初始化种子），但 CUDA 算子的非确定性与库版本差异仍可能带来微小数值波动。所以可复现 = "流程可复现 + 证据完整"，不承诺逐位一致（REPRODUCIBILITY 第 1 节）。

### 评测与验收

**Q14：八项门槛是脚本自动判定的吗？**
不是。评测脚本自动产出指标，**通过与否需要人工对照报告逐项确认并记录**（REPRODUCIBILITY 第 7 节）。未来若引入自动 gate，也必须保存 gate 版本与代码 commit。

**Q15：为什么验收用"融合口径"而不是模型裸分？**
生产真实行为是 `max(规则, 模型)`。验收必须评"接入后的系统"，`rules ∪ QLoRA` 的 HIGH recall 与 FPR 增幅才是对生产有效的数字（第 8.5、9.1 节）。

**Q16：模型某条判错了，能手工修吗？**
不存在"手工修单条"的通道——模型行为只能通过数据、提示词契约或超参的**版本化变更 + 重训 + 重验收**改变。单点修法（后处理规则）会破坏可审计性。

### 服务与接入

**Q17：`risk_level: null` 和 `low` 有什么区别？**
null = 模型没有给出可用答案（解析失败/输出为空），调用方必须回退规则；把它当 low 处理会漏报。low 是模型明确判断"一般困扰"（第 8.7、10 章）。

**Q18：`--load-4bit` 能省一半显存，为什么生产不建议直接开？**
4-bit 推理可能偏移边界样本预测，而验收口径是 BF16。口径一致性是验收有意义的前提；要切 4-bit，先重跑冻结评测（第 8.7 节）。

**Q19：能把 8301 服务直接给生产用吗？**
不能。本机服务无 TLS、认证、限流与审计，只用于 smoke test；生产必须经审批的 HTTPS 端点 + 反向代理 + 完整保护清单（SERVICE-RUNBOOK 第 5 节），且主项目 URL 校验本就拒绝环回/私有地址。

**Q20：想更新 system prompt 措辞，改哪里？**
两处独立副本同步修改（训练仓库 `data_contract.py` 与生产项目 `app/llm/client.py`；数据构建、评测与本地服务脚本均从 `data_contract.py` 导入，自动跟随），且这是**重训级事件**：新 prompt → 重建数据 → 重训 → 全量重验收。旧模型数字只绑定旧 prompt（第 6.3 节）。

---

## 附录 B 易错点汇总清单

按环节归集全书"易错点"，动手前过一遍。

### 环境（第 3-4 章）

- [ ] 用错解释器：始终用 `envs\qlora-qwen35\Scripts\python.exe` 完整路径。
- [ ] 把 `pip install -r` 成功当成"CUDA 可用"：必须自检 `+cu12x` 与 `is_available()`。
- [ ] `set` 的环境变量随窗口失效；新窗口要重设。
- [ ] 把 requirements 的 `>=` 下限当成实际版本；复现要靠 `pip freeze` 证据。

### 数据（第 6 章）

- [ ] 关键词触发就升 high——标签判的是**说话人自身**的意向与计划性。
- [ ] 自动映射冒充 `manual_reviewed`——manifest 审计会暴露。
- [ ] 构建退出码 0 就跳过 manifest 审阅（尤其 `leakage_rejected`）。
- [ ] 手工编辑产物 JSONL 加数据。
- [ ] 为多进数据调低泄漏阈值 0.82。
- [ ] 把 `build_consolidated.py` 的旧 prompt 中间产物直接当训练数据。
- [ ] 改 prompt 只改一处独立副本（两处必须同步：`data_contract.py` 与生产项目 `app/llm/client.py`），或对旧模型用新 prompt 解读旧数字。

### 训练（第 7-8 章）

- [ ] 跳过 dry-run 直接训 3 小时。
- [ ] 忘传 `--output-root` 或两个实验共用一个输出目录。
- [ ] 直接改 `risk_qlora_4060.yaml` 做实验而非复制版本化实验文件。
- [ ] 把"eval_loss 回升"当训练失败（那是选型信号，系统已自动回选最优点）。
- [ ] 忽视 `assistant target was truncated` 报错强行继续。
- [ ] 训练产出 manifest 不备份。

### 评测与验收（第 8-9 章）

- [ ] `--qlora-model` 与 `--qlora-model-dir` 混用。
- [ ] devtest 数字当上线结论。
- [ ] JSON 有效率低时只放宽解析不查根因。
- [ ] 指标好就宣布"通过验收"，跳过人工逐项对照门槛。
- [ ] 引用历史数字不注明版次与 prompt 口径。

### 服务与接入（第 8.7、10-11 章）

- [ ] 把 `risk_level: null` 当 low 处理。
- [ ] 切换 `--load-4bit` / 更换精度不重新评测。
- [ ] 把 8301 本机服务或环回地址写进生产配置。
- [ ] merge 输出目录复用旧版本目录（防覆盖保护会拒绝，别绕过）。
- [ ] 异常时先诊断后断开关——正确顺序是先回退规则、再排查。
- [ ] 权重、日志、敏感语料提交进 Git 或放回代码仓库。

---

## 附录 C 分级学习路径与自测清单

### C.1 零基础路径（2-3 天）

| 阶段 | 内容 | 产出 | 预计用时 |
| --- | --- | --- | --- |
| ① 建立全景 | 第 0 章全部 + 术语表通读 | 能向别人复述"QLoRA 训练风险分类器"每个词的含义 | 2h |
| ② 认识系统 | 第 1、2、3 章 + 练习（含 3.4 模块地图与调用链） | 理解隔离边界、版本谱系、跑通路径守卫示例、能画出模块依赖图 | 1.5h |
| ③ 环境实操 | 第 4 章自检、第 5 章 gate 实跑 | 环境快照记录 + gate 输出记录 | 1-2h |
| ④ 数据关 | 第 6 章逐节 + 全部练习；重点 6.1.4 边界案例、6.4 泄漏防护与 **6.7 源码走读** | parse_sample/Jaccard 示例运行记录；能复述 relabel() 漏斗与就近主语裁决 | 4-5h |
| ⑤ 参数与训练 | 第 7、8 章逐节 + 练习；dry-run 实跑 | dry-run 六项确认记录 + 7.6 步数推算 | 2-3h |
| ⑥ 评测与验收 | 第 8.5-8.7 + 第 9 章 + 指标手算示例 | 指标公式手算记录 | 2h |
| ⑦ 全局观 | 第 10-15 章通读 + 附录 A FAQ | 自测清单全绿 | 2h |

### C.2 有经验工程师路径（0.5-1 天）

第 0.6-0.8 节（QLoRA 显存账与数据划分）→ §3.4（模块地图与调用链）→ 第 6 章（数据契约、泄漏防护与 **6.7 源码走读**，本项目最独特的部分）→ 第 7 章（参数与逐条设计理由）→ 第 8 章（完整实操链路）→ 第 9-11 章（门槛、融合、发布）→ 附录 D 自查后，按第 14 章清单动手复现。

### C.3 运维/复核路径（2-3h）

第 1 章（红线）→ 第 2 章（当前版本与谱系）→ 第 9 章（八门槛逐条）→ 第 10-11 章（接入契约与回滚顺序）→ [SERVICE-RUNBOOK.md](docs/SERVICE-RUNBOOK.md)（上线检查表）→ [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)（证据要求）→ 附录 B 过一遍。

### C.4 全书自测清单（能全答"是"即出师）

- [ ] 能不看书写出四个数据集合（train/dev/devtest/stress）的规模、比喻和职责？
- [ ] 能解释为什么 `high` 必须 `speaker_scope=self`，并举一个隐喻式 high 和一个第三人称 low 的例子？
- [ ] 能说清泄漏防护的"精确哈希 → 3-gram 倒排 → 0.82 阈值"三层机制，以及为什么不能调低阈值？
- [ ] 能从配置推算出 540 步、warmup 约 27 步、最优 checkpoint 在 step 360？
- [ ] 能解释 `-100` 标签掩码的作用和"截断即报错"的设计动机？
- [ ] 能说出 eval_loss 曲线三个数字对应的训练现象与系统的自动应对？
- [ ] 能手推 macro-F1、FPR、P95 的计算并说明为什么验收用融合口径？
- [ ] 能背出八项验收门槛各防什么事故，以及失败后的五步处置？
- [ ] 能解释 max 融合为什么"只能升不能降"，以及 null 与 low 的本质区别？
- [ ] 能列出回滚的四步顺序和每步的理由？

全部通过后，建议从"照着手册做"毕业到"对着 [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) 独立完成一次带完整证据链的实验"——那是本手册的最终学习目标。

---

## 附录 D 从零复刻检查表（代码导向）

第 14 章清单回答"实验证据是否完整"（证据导向）；本清单回答"你能否从零重建这个后端"（代码导向）。按依赖顺序自测，每一项的标准是：**不看源码能独立写出或讲清设计**。全部打勾 ≈ 具备完全复刻能力。

- [ ] 1. `paths.py`：能写出 `training_root()` 的环境变量回退与 `under()` 的 `relative_to` 越界拒绝（§3.2）。
- [ ] 2. `data_contract.py`：能默写 `RISK_SYSTEM_PROMPT` 的三段结构（自身意向判定 + 三级定义与边界 + 只输出 JSON）；能列出 `validate_sample` 的全部校验；能解释 `normalize_message` 为什么用 NFKC+去空白+小写（泄漏检测的稳定性键）。
- [ ] 3. `base_model_gate.py`：能列出 `EXPECTED_*` 常量与 `family_scale_match` 的判定项；能说清 `status` 与 Ollama provenance 的绑定关系（§5.2）。
- [ ] 4. 数据构建 `prepare_risk_sft_v4.py`：能画出 `relabel()` 六分支漏斗并解释分支顺序不可调换（§6.7.3）；能描述 `_nearest_subject_is_self` 的窗口裁决与三重守卫各防什么（§6.7.2）；能推演 `take()` 对给定候选池的输出；能解释 dev_medium 预留、≤80 降密度、22% 第三人称配额三个机制；能列出 manifest 必备字段（§6.7.8）。
- [ ] 5. legacy 管线（`source_ingest.py`/`leakage_guard.py`）：能解释 `git show HEAD:` 只读设计、他人语境覆盖源标签的弱映射（§3.4.3）；能说清倒排索引预筛与"max(Jaccard, SequenceMatcher)"口径，以及它与 v9 内联实现的分工（§6.4.2）。
- [ ] 6. `train_risk_qlora.py`：能按序复述 4-bit 加载 → `prepare_model_for_kbit_training` → 目标模块交集 → tokenize（-100 掩码、截断报错）→ Trainer 参数 → manifest 的完整链路（§7、§7.5）。
- [ ] 7. `merge_risk_qlora.py`：能解释 `safe_merge`、空目录防覆盖、`do_sample=false` 三个动作的理由（§8.4）。
- [ ] 8. `metrics.py`：能手算 `classification_report` 的全部字段，含 P95 的取法与 macro-F1 的不加权平均（§8.5）。
- [ ] 9. `eval_risk_qlora.py`/`eval_risk_devtest.py`：能解释五组报告、`_fuse` 只升不降、跨仓库规则基线导入（§8.5）、devtest 同口径跨脚本导入与"解析失败按协议记 low"（§8.6）。
- [ ] 10. `serve_risk_qlora.py`：能写出 `/health`、`/assess` 契约、宽容 JSON 解析五步、400/500/null 的错误语义（§8.7）。

完成后，你应能在白纸上画出 §3.4.1 的完整依赖图，并为每条边回答"为什么这样连接"。
