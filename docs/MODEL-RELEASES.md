# Model Release Records

只记录已经完成审查的模型工件元数据，不在 Git 中保存权重本身。

## Status definitions

- `research-only`：可用于研究和误差分析，未通过全部门槛或证据不完整，禁止生产接入。
- `release-candidate`：冻结集门槛通过，证据包基本完整，等待外部审阅/部署审批；默认仍关闭生产开关。
- `production-approved`：完成外部审阅、部署安全检查、回滚演练和明确审批，可按记录接入生产。
- `rejected`：门槛失败、来源/许可证不清、泄漏或服务安全条件不满足；必须保留失败原因。

`V9-TRAINING-EVAL-SUMMARY.md` 的“八门槛全过”只证明评测摘要结论，不自动等于 `production-approved`。只有本模板中的 release record、工件 hash、数据/配置/环境证据、审批人和回滚版本全部完成，才能改变状态。

## Release checklist

- [ ] 基座模型 repo、固定 revision、快照 hash 和许可证已记录
- [ ] 训练仓库 commit、配置 hash、环境版本和 GPU 已记录
- [ ] 训练数据版本、manifest hash、来源授权、脱敏状态和 `label_method` 已记录
- [ ] system prompt / risk contract 原文、版本和 hash 已记录
- [ ] adapter 或 merged 模型的下载位置、文件清单和 SHA-256 已确认
- [ ] 冻结 holdout、devtest、外部测试集和性能结果已附上
- [ ] 训练 manifest、merge manifest 和 raw predictions 已归档
- [ ] 生产接入开关、回滚版本、回滚工件和已知限制已说明
- [ ] 生产服务的 TLS、认证、限流、日志脱敏和 readiness 已审查
- [ ] 审批人、审批日期和发布状态已记录
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
