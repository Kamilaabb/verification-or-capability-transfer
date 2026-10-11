# Authority v5 Protocol

状态：稳定研究规则。具体数据、模型、温度、prompt 和 seed 不写在这里。

## 1. 研究问题

当 Generator `G` 已经作答、Verifier `V` 已经给出一次固定检查报告后，外部 Controller `C` 能否通过限制 V 的执行权限，改善最终正确率以及 repair–harm 权衡？

递进命题：

- H1：实例级 C3 优于 V 自决的 M2；
- H2：C3 优于所有固定权限 M0/M1/M2；
- H3：三档 C3 优于信息和模型匹配的二档 C2。

Pilot 结果只作探索。正式主张需要冻结方法后的独立 test。

## 2. 时序和信息边界

```text
problem
  -> G 生成 Condition 预先配置数量的初始样本；sample_index=0 无条件成为 y0
  -> V 读取 problem 与 y0，只生成一次报告
  -> 从动作执行前的信息构造 scalar evidence
  -> C 选择 authority cap
  -> executor 计算 min(cap, request)
  -> 必要时 G 根据 critique 生成 y1
  -> grader 在 C 决策之后计算结果标签
```

离线实验可以预先生成各权限分支以构造反事实表，但 Controller 永远不能看到分支结果。

## 3. V 报告

一次有效报告包含：

```json
{
  "verdict": "correct | incorrect | uncertain",
  "critique": "可独立交给 G 的检查意见",
  "candidate_answer": "V 的完整候选答案",
  "requested_action": "KEEP | CRITIQUE_REVISE | OVERRIDE"
}
```

所有权限策略共享同一份 V 报告。不得针对 M0/M1/M2 重新询问 V。G rewrite 只收到 problem、y0 和 critique，不收到 candidate_answer。

## 4. 权限和唯一执行映射

```text
KEEP = 0
CRITIQUE_REVISE = 1
OVERRIDE = 2

M0 = 0  无执行权限
M1 = 1  最多允许 critique-and-revise
M2 = 2  允许 V 按自己的请求行动

effective_action = min(authority_cap, requested_action)
```

|V 请求|M0 / z0|M1 / z1|M2 / z2|
|---|---|---|---|
|KEEP|u0|u0|u0|
|CRITIQUE_REVISE|u0|u1|u1|
|OVERRIDE|u0|u1|u2|

其中 `u0/u1/u2` 是 y0/y1/y2 的正确性。后续策略、oracle 和训练标签只基于 z0/z1/z2。

## 5. 失败语义

失败必须可表示，不能伪造一个“有效”V报告：

- V JSON 在一次格式修复后仍失败：`v_report=null`，系统请求回退为 KEEP；
- Controller JSON 在一次格式修复后仍失败：回退为 M0，并单独报告 fallback rate；
- `requested_action=KEEP`：不运行 rewrite，`y1/u1=null`；
- rewrite 调用、抽取或模型输出失败：实际执行回退为 KEEP，因此该分支最终正确性使用 u0，同时保留 `u1=null` 和失败状态；
- 模型给出不可抽取答案：属于模型无效输出，可按冻结 grader 规则记为错误；
- grader exception、timeout 或 gold parse failure：属于实验管线错误，结果为 null，run 不得 validated。

主分析使用同一 problem×V 集合，不得为不同策略分别删除失败样本。

## 6. Controller 只能看到什么

Controller 只接收 Condition 选定的版本化 feature contract 所注册的标量、布尔值和缺失标志。实际 payload 必须匹配该 contract 指定的 input schema，且只含 `features` 与 `missingness`。严格禁止：

- gold/reference answer；
- u0/u1/u2、z0/z1/z2、oracle 或 best-cap label；
- grader 输出；
- y1 或其他动作执行后信息；
- problem、y0、G samples、critique、candidate_answer 的原文；
- 测试标签事后设计的字段或阈值。

物理隔离规则：feature evidence 在 grading 之前写入独立文件；Controller runner 不能读取 counterfactual label 文件，也不能把 evidence 存储记录里的 problem/cell/run 标识传给模型。

## 7. Scalar contract

精确启用特征、公式、范围和缺失规则由 Condition 引用的 feature contract 决定。`features_scalar_v1.yaml` 是八样本自一致性版本；`features_scalar_single_v1.yaml` 是单次 G、单次 V 的低成本版本。两者都只使用动作前的可解释信号：

- G 自一致性（仅多样本 contract）与可解析性；
- V verdict；
- V requested action；
- G 与 V candidate 的答案一致性。

长度特征和手工乘积项不进入主特征；可以在新 feature contract 中作为消融。单样本 contract 不保留会退化为常数的众数、熵或支持率。训练型 Controller 必须报告 G-only、V-only 和 G+V 三组消融。

标量的名字、公式、范围和缺失语义是版本化的研究契约；Condition 选择一个精确 `feature_set_id`。改字段或公式时建立新 feature contract 和新 condition。若仍处在相同的证据类别与决策时点，不必升级整个 protocol；若扩展了允许的信息类别或把决策移到动作之后，则必须建立新 protocol。

无论是哪种版本变化，都不自动等于重跑 G/V：如果已保存 raw output 足够重算，只重做 evidence 及其下游。

## 8. 必须保存的输出

完整保留规则见 `OUTPUT_CONTRACT.md`。原则是保存足以审计和离线重算的最小信息：raw model output、extracted answer、parse/grader 状态、模型与 prompt 身份、seed、token、延迟、stop reason、重试、上游 run ID 和 checksum。

## 9. 对照和指标

必做策略：Fixed M0、Fixed M1、Fixed M2、C2 `{M0,M2}`、C3 `{M0,M1,M2}`。

```text
Accuracy(pi) = mean(z_pi)
Repair(pi)   = mean((u0 == 0) and (z_pi == 1))
Harm(pi)     = mean((u0 == 1) and (z_pi == 0))
```

必须验证：

```text
Accuracy(pi) - Accuracy(M0) = Repair(pi) - Harm(pi)
```

同时报告条件 repair/harm、authority 与 effective-action 分布、rewrite rate、fallback、tokens、延迟，以及离线生成成本与部署路径成本。

诊断指标包括 sample oracle、posthoc best fixed、oracle gap、M1 独占成功和 OVERRIDE 子集的 u1/u2 四格表；它们不是可部署策略。

## 10. 统计

- 所有比较按 problem 配对；
- 同题的 V cells、branches 和 replicates必须一起重采样或进入同一 fold；
- A/B cell 分开报告，pooled 结果不能把同题多行当独立题；
- C2/C3 使用相同 C、证据、预算和 replicate 集，只改变菜单；
- 多 seed 报告每个 replicate、均值、标准差和范围；不得选择最佳 seed；
- 多数投票属于额外 ensemble，因为它改变部署调用成本；
- pilot300 的点估计和置信区间只作探索。

## 11. 什么变化需要新 protocol

以下变化建立新 protocol：权限语义、执行映射、Controller 决策时点、主要研究对象或允许的证据类别改变。

以下通常只需要新 condition：模型、prompt、温度、token 预算、具体 feature contract、grader版本或 Controller算法改变。
