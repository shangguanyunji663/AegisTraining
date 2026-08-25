# AegisTraining

`AegisTraining` 是 Aegis Psych Agent 的隔离训练工程。它负责风险识别模型的数据准备、QLoRA 训练、离线评测和模型发布记录；生产应用位于独立的 [`aegis-psych-agent`](https://github.com/shangguanyunji663/aegis-psych-agent) 仓库。

## 仓库边界

| 仓库/位置 | 内容 |
| --- | --- |
| `aegis-psych-agent` | FastAPI 应用、Agent/RAG/工具治理、前端、测试和部署 |
| `AegisTraining` | 训练源码、数据契约、训练配置、离线评测和审计文档 |
| 外部模型存储 | 基座快照、adapter、merged 权重、GGUF 和其他大工件 |

本仓库**不提交**基座模型、merged 模型、checkpoint、GGUF、HF cache、CUDA 环境、虚拟环境、原始数据、运行日志或中间产物。当前本机工作目录约 36G，其中单个基座权重约 4.5G；不要使用 `git add -f` 绕过 `.gitignore`。

心理健康、自杀和风险识别语料必须先完成授权、脱敏、许可证和公开范围审查。未完成审查的数据只保留在受控本地目录。

## 目录

```text
.
├── external-data/                # 被忽略的外部数据 checkout（仅供旧流程/重建工具使用）
├── training/
│   ├── src/aegis_training/       # 数据契约、泄漏检查、指标和路径工具
│   ├── scripts/                  # 数据准备、训练、合并、评测和推理服务入口
│   ├── configs/                  # 可复现实验配置
│   ├── data/                     # 仅保留契约/说明/经审查的小型样例
│   └── requirements-qlora.txt    # 与生产依赖隔离的 GPU 训练依赖
├── tools/                        # legacy/v3 检查工具；当前入口见 training/scripts/
├── docs/                         # 数据审计、训练协议和模型发布记录
├── reports/                      # 唯一报告目录，只保留轻量、经审查的摘要
└── .gitignore
```

## 路径配置

脚本不依赖某台机器的绝对路径。Windows 示例：

```bat
set AEGIS_TRAINING_ROOT=D:\AegisTraining
set AEGIS_PROJECT_ROOT=D:\PythonProject\aegis-psych-agent
set AEGIS_PROJECT_CORPUS=%AEGIS_PROJECT_ROOT%\eval\fixtures\representative_corpus.json
```

- `AEGIS_TRAINING_ROOT`：训练仓库 checkout 和本地训练工件根目录。
- `AEGIS_PROJECT_ROOT`：可选的生产仓库 checkout；只读其已提交的评测 fixture。
- `AEGIS_PROJECT_CORPUS`：可选的版本化评测语料路径。也可以直接通过 `--corpus` 传入。
- `external-data/`：本地外部数据 checkout（被忽略，不属于当前 v4 主流程的直接输入）。
- 模型、checkpoint、导出文件和报告应放在训练根目录下的被忽略目录中。

如果训练仓库和生产仓库不在同一台机器上，请把需要的、已经审查的 fixture 作为版本化契约或脱敏副本提供给训练流程；不要写死本机路径。

## 初始化环境

建议使用独立的 Python 3.11 GPU 环境。不要把训练依赖追加到生产项目的 `requirements.txt`：

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r training\requirements-qlora.txt
```

PyTorch/CUDA wheel 应按本机驱动和 CUDA 版本从官方渠道单独安装。训练前先用固定 revision 的官方 Qwen3.5-2B-Base 快照执行 gate；本仓库不包含该快照。

## 常用入口

准备第一版候选数据（示例路径均可替换）：

```bat
python training\scripts\prepare_risk_sft.py ^
  --source-root "%AEGIS_TRAINING_ROOT%\external-data\SupervisedVsLLM-EfficacyEval" ^
  --project-root "%AEGIS_PROJECT_ROOT%" ^
  --output-root "%AEGIS_TRAINING_ROOT%\data\archive\risk_sft_v2"
```

准备当前审计版本的数据，并显式指定冻结评测语料：

```bat
python training\scripts\prepare_risk_sft_v4.py ^
  --consolidated-root "%AEGIS_TRAINING_ROOT%\training\data\consolidated_risk_v1" ^
  --corpus "%AEGIS_PROJECT_CORPUS%" ^
  --output-root "%AEGIS_TRAINING_ROOT%\data\risk_sft_v9"
```

先执行环境/模型 dry run，再开始训练：

```bat
python training\scripts\train_risk_qlora.py ^
  --data-root "%AEGIS_TRAINING_ROOT%\data\risk_sft_v9" ^
  --snapshot-dir "%AEGIS_TRAINING_ROOT%\models\Qwen3.5-2B-Base" ^
  --dry-run

python training\scripts\train_risk_qlora.py ^
  --data-root "%AEGIS_TRAINING_ROOT%\data\risk_sft_v9" ^
  --snapshot-dir "%AEGIS_TRAINING_ROOT%\models\Qwen3.5-2B-Base"
```

合并 adapter 时使用新的、空的、版本化输出目录：

```bat
python training\scripts\merge_risk_qlora.py ^
  --snapshot-dir "%AEGIS_TRAINING_ROOT%\models\Qwen3.5-2B-Base" ^
  --adapter-dir "%AEGIS_TRAINING_ROOT%\checkpoints\aegis-risk-qwen3.5-2b-v9\adapter" ^
  --output-dir "%AEGIS_TRAINING_ROOT%\exports\aegis-risk-qwen3.5-2b-v9-merged"
```

风险服务入口：

```bat
python training\scripts\serve_risk_qlora.py ^
  --model-dir "%AEGIS_TRAINING_ROOT%\exports\aegis-risk-qwen3.5-2b-v9-merged" ^
  --host 127.0.0.1 --port 8301
```

该服务的 `127.0.0.1` 监听仅用于隔离环境的本机 smoke test。生产应用的服务端 URL 校验只允许 `http/https`，并拒绝 `localhost`、环回、私有和保留地址；生产部署应使用经过审批的、可达且受保护的公网 HTTPS endpoint，不能把本地训练服务直接暴露给生产应用。

## 模型发布

模型权重应发布到 Hugging Face、受控对象存储或经过审查的 GitHub Release，而不是代码仓库。每个发布版本必须记录：

- 模型/adapter 版本和基座 revision；
- 下载地址与许可证；
- SHA-256 校验和；
- 训练数据版本、提示词契约版本和验收报告；
- 是否允许生产接入以及回滚版本。

发布记录模板见 [`docs/MODEL-RELEASES.md`](docs/MODEL-RELEASES.md)。没有完整元数据的权重不视为可发布模型。

## 提交前检查

不要直接执行 `git add .`。提交前逐项检查：

```bash
git status --short
git ls-files
git ls-files -z | xargs -0 -n1 git check-ignore -v --no-index 2>/dev/null
```

另外检查历史大文件和敏感信息；补 `.gitignore` 只能阻止未来跟踪，不能从已有 Git 历史中删除权重或密钥。若历史已经包含大文件，应单独规划历史清理和远端重写。

## 安全边界

训练模型只负责风险等级 JSON 分类，不能替代专业心理咨询或危机干预。生产接入必须保留规则通道兜底，并以冻结集、外部审阅和可回滚发布流程作为上线条件。
