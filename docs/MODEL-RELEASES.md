# Model Release Records

只记录已经完成审查的模型工件元数据，不在 Git 中保存权重本身。

## Release checklist

- [ ] 基座模型 repo、固定 revision 和许可证已记录
- [ ] 训练数据版本、来源授权、脱敏状态和 `label_method` 已记录
- [ ] system prompt / risk contract 版本已记录
- [ ] adapter 或 merged 模型的下载位置已确认
- [ ] 每个文件已计算 SHA-256
- [ ] 冻结 holdout、外部测试集和性能结果已附上
- [ ] 生产接入开关、回滚版本和已知限制已说明
- [ ] 模型没有被误称为临床有效性或医学诊断系统

## Release template

```yaml
version: aegis-risk-qwen3.5-2b-v9
status: research-only # research-only | release-candidate | production-approved
base_model:
  repo: Qwen/Qwen3.5-2B-Base
  revision: <fixed revision>
  license: <license>
artifacts:
  - name: adapter
    uri: <external URI>
    sha256: <sha256>
  - name: merged
    uri: <external URI>
    sha256: <sha256>
prompt_contract: v2
data_manifest: <manifest URI or commit>
evaluation:
  report: <report URI or commit>
  acceptance_passed: false
known_limitations:
  - <limitation>
rollback_version: <previous approved version or none>
```
