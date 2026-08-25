# Aegis 风险 QLoRA 训练沿革索引（最终谱系）

> 版次裁定（2026-08-22 用户确认）：**旧 v1 = 第一版；v2 视为不存在（历史上本无独立
> v2 训练，eval-v2 文件实为 v1 模型复评）；旧 v3 = 第二版；旧 v4 = 第三版；
> 旧 v5 = 第四版**，后续顺延。
> 磁盘目录与历史实验编号保留旧编号（如 `risk_sft_v3`、`checkpoints\...-v5`），以本表映射为准。
> 模型重工件（checkpoint、merged 导出、GGUF）和原始评测 JSON/日志未随当前仓库保留；本索引只保留已审查的摘要与可复核结论。

## 主谱系

| 版次 | 旧编号 | 数据集 | 模型工件（可清理） | 报告文件（本目录） | 结果 |
| --- | --- | --- | --- | --- | --- |
| 第一版 | v1 | risk_sft_v1 / risk_sft_v2（720+120 配额制） | checkpoints\...-v1、exports\...-v1-merged（已清理） | 原始 JSON 未随仓库保留；结论见本索引 | 验收失败：隐喻新增 +3 < 4；FPR 0 通过 |
| 第二版 | v3 | risk_sft_v3（840+140） | checkpoints\...-v3（已清理） | `V3-TRAINING-EVAL-SUMMARY.md`；原始 JSON 未随仓库保留 | 8 门槛 7 过 1 败：FPR +4.84pp；隐喻新增 +8 |
| 第三版 | v4 | risk_sft_v4（1703+198+devtest 1414） | checkpoints\...-v4、exports\...-v4-merged（已清理） | 原始 JSON/日志未随仓库保留；摘要结论见 V5/V9 对照 | 8 门槛 7 过 1 败：FPR +3.23pp（差 1 条过线）；dev-test acc 0.910 |
| 第四版 | v5 | risk_sft_v5 | checkpoints\...-v5、exports\...-v5-merged（已清理） | `V5-TRAINING-EVAL-SUMMARY.md` | **八门槛全部通过（历史首次）**：FPR 0（零误升级）、隐喻 +8、P95 0.93s |
| 第五版 | v7 | risk_sft_v7 | checkpoints\...-v7、exports\...-v7-merged（已清理） | 原始 JSON 未随仓库保留；结论保留于版本谱系 | 7/8：第三人称 0.55→0.82、medium 召回 0.29→0.77；但 FPR 复发（corp-084/091 自我否定误升级） |
| 第六版 | v8 | risk_sft_v8 | checkpoints\...-v8、exports\...-v8-merged（已清理） | 原始 JSON 未随仓库保留；结论保留于版本谱系 | 7/8：acc 0.782、第三人称 0.91、medium 0.71；同样卡 corp-084/091 |
| **第七版** | **v9** | **risk_sft_v9（提示词 v2）** | checkpoints\...-v9、exports\...-v9-merged（已清理） | `V9-TRAINING-EVAL-SUMMARY.md`；原始 JSON 未随仓库保留 | ✅ **八门槛全过 + 全指标最优**：FPR 0、corp-084/091 修复、medium 召回 0.88、acc 0.782、第三人称 0.82、隐喻 +6 |

## 提示词契约变更记录（方案 B，2026-08-24）

v1 → v2：高危示例移除「不配」「活着多余」（与冻结金标细线冲突：拖累/不配拥有=medium vs
消失/不配活着=high）。三处副本同步：`data_contract.py`、`app/llm/client.py`、数据脚本。
**旧版模型（第一~六版）评测时必须绑定 v1 提示词**（各版训练期契约），本索引即为依据。

> v2 编号作废：历史上不存在独立 v2 训练；原 v1 模型复评文件未随当前仓库保留，结论已并入本索引。

## 数据集版本说明（train/dev 为 messages 格式；冻结 stress 87 条永不入训）

| 数据集 | 服务版次 | 规模与配额 | 来源构成 | 备注 |
| --- | --- | --- | --- | --- |
| risk_sft_v1 | 第一版 | 720 train / 120 dev（240×3 / 40×3） | 候选池 4737：外部自杀/认知歪曲/SocialCD 弱监督映射 + 项目 base 层 | 首轮配额制数据 |
| risk_sft_v2 | 第一版 | 720 train / 120 dev（240×3 / 40×3） | 池 4746：hongzhiq-suicide 413+71、socialcd 270+44、认知歪曲 10+2、project-base 27+3 | 第一版期内来源迭代 |
| risk_sft_v2_round2 | 第一版→二版过渡 | 90 条合成隐喻隐式 high | 场景模板生成 + 逐条复核 | 9 条被第二版泄漏防护正确拒绝（与冻结集近重复） |
| risk_sft_v3 | 第二版 | 840 train / 140 dev（280×3） | 七路：hongzhiq-suicide 380+57、socialcd 299+52、psysuicide 精选 71+11、synthetic 41+9、hard-negative 9+2、认知歪曲 10+3、project-base 30+6；隐喻 112+20 | 原始 build report 未随当前仓库保留；结论见本索引 |
| risk_sft_v4 | 第三版 | 1703 train / 198 dev / devtest 1414 | consolidated_risk_v1（用户人工清洗）+ 三轮裁决重标（low→high 295、low→medium 68、移除纯攻击 220）+ 48 困难负例 + 6 对照 | 首次引入 devtest 同规则重标；裁决标准见 prepare 脚本注释 |
| risk_sft_v5 | 第四版 | 同 v4 配额 | v4 三处修复：①意念表补回「离开这个世界/说一声再见」（修 train 6 / test 12 gold 漏标）；②被动纯无意义降 medium（含死亡愿望词维持 high）；③第三人称表补父母/老师/孩子 | 目标：FPR ≤2pp 过线 |

## 第三版（旧 v4）关键数字

- 训练：eval_loss 0.0736→0.0525→0.0494（321 步，2h09m，最优即最终）
- 冻结 stress 87：隐喻新增 +7、FPR 3.23pp、JSON 100%、P95 0.97-1.2s；corp-090/099 误升级修复
- dev-test 1414：acc 0.910 / macro-F1 0.795（原始 qwen3.5:2b 约 0.77 / JSON 94%）

## 第四版（旧 v5）关键数字

- 训练：eval_loss 0.0701→0.0490→0.0461（321 步，最优即最终）
- 冻结 stress 87：**8/8 全过**——FPR 0（corp-090/097/099 全部正确 medium）、隐喻新增 +8（13→21）、JSON 100%、P95 0.93s
- dev-test 1414：acc 0.901 / macro-F1 0.769 / high-F1 0.876；弱项 medium F1 0.496

## 下游目标

通过八门槛验收后，按原方案 §8 接入生产：`RISK_QLORA_ENABLED` 开关（默认关）、
RiskGuardian 专属推理通道、`max(规则, QLoRA)` 只升不降融合、异常/超时/解析失败回退规则。
Ollama 导入在本机版本不兼容，服务化路径以隔离 Transformers 环境为准；生产接入需用户最终确认。
