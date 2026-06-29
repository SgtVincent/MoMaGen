# MoMaGen multi-agent review rule

本规则用于 MoMaGen 复现 / 修复任务，尤其是涉及 BEHAVIOR-1K、OmniGibson、R1Pro、轨迹生成、navigation + manipulation pipeline 的调试。

## 触发频率

- 每完成一个完整推进轮次后必须触发一次 multi-agent review；一个轮次至少包括：
  1. 明确当前目的 / 假设；
  2. 完成一组代码修改或实验；
  3. 有可观察结果（日志、测试、smoke、失败堆栈或数据表）；
  4. 形成下一步 replan。
- 如果一个轮次超过 2 小时仍未闭环，也必须中途触发一次 review。
- 如果连续 2 次实验失败且失败模式没有显著变化，必须触发 review 后再继续。
- 如果用户指出路线偏离原始方法，必须立即触发 review，并把纠偏结论写入飞书文档。

## 必须使用的 reviewer 模型

每次 review 至少启动两个独立 reviewer：

1. `gemini-3.1-pro`
2. `deepseek-v4-pro`

如果使用 `/agent-new`，应显式指定模型；如果当前工具接口不支持模型参数，必须在 review 记录中注明“模型选择接口不可用”，并仍然用两个独立 agent 按不同审查重点执行 review。

## Reviewer 输入要求

每个 reviewer 必须收到以下信息：

- 当前目的：本轮想验证 / 修复什么。
- 当前进展：已改了哪些文件、跑了哪些命令、得到哪些日志。
- 当前实现：关键代码路径与核心 diff 摘要。
- 当前结果：成功 / 失败证据，包括日志路径和关键数值。
- 当前 replan：下一步打算做什么。
- 明确要求 reviewer 检查：
  - 是否偏离 MoMaGen 原始方法；
  - 是否把 navigation / base transport 错误替换成 arm-only IK；
  - 是否有过度宽松 fallback；
  - 是否有隐藏的 API / embodiment drift；
  - 是否应先做实验而不是继续写代码；
  - 是否需要更新飞书文档。

## Reviewer 输出要求

每个 reviewer 必须输出结构化结论：

1. `Verdict`：继续 / 暂停 / 回滚 / 改路线。
2. `Major concerns`：最重要的风险。
3. `Evidence check`：当前证据是否支持结论。
4. `Method alignment`：是否符合 MoMaGen 原始方法。
5. `Next actions`：建议的下一步，按优先级排序。

## 主 agent 的后续动作

- review 后必须汇总两个 reviewer 的共同意见和分歧。
- 如果 reviewer 指出路线错误，优先纠偏，不要继续沿错误方向堆 patch。
- 如果 reviewer 结论影响当前飞书状态页，必须更新飞书文档。
- 不允许等用户发现偏离后才 review。
