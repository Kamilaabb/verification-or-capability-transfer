# Authority v6 Protocol

状态：development300 设计已冻结；服务器兼容性 smoke 尚待执行。

`authority_v6` 保留 `authority_v5` 的研究问题、M0/M1/M2 语义、唯一执行映射、
失败回退、grader、信息隔离、统计和报告要求。以下文件按字节引用且不在本版本中
改写：

- `protocols/authority_v5/README.md` 的第 1–6、8–10 节；
- `protocols/authority_v5/grader_math_verify_v1.yaml`；
- `protocols/authority_v5/OUTPUT_CONTRACT.md` 中除本文件明确扩展之外的规则。

本版本只做三项研究语义扩展：允许动作前 token likelihood 标量、正式注册多输出
cap correctness 训练目标、允许指定的 300 题作为开发和调参数据。旧 protocol、
dataset、condition 和完成 run 均保持不变。

## 1. 时序与泄漏边界

决策时序仍为：

```text
problem -> G y0 -> V report -> scalar evidence -> C -> action -> grading
```

logprob 必须来自产生已注册 `y0` 或 V report 的同一次调用。不得进行看到 gold、
grader 或分支结果后的重打分。Controller payload 仍只能包含 `features` 和
`missingness`；token、raw text、problem ID、run metadata、gold、u/z 和 Oracle
字段均不得进入推理 payload。

## 2. Logprob 证据

`scalar_single_v2` 允许 sampled-token natural logprob。服务器必须使用并记录
`raw_logprobs` 模式，使数值位于 temperature、top-p/top-k 和 structured-output
mask 之前的模型分布。若 runtime 无法证明该语义，兼容性 gate 失败。

原始 G/V 记录保存每个生成 token 的 token ID（若 runtime 提供）、token bytes、
UTF-8 byte span 和 sampled-token logprob。对 thinking V，trace 可以同时包含
reasoning；最终 `message.content` 必须是可唯一定位的完整后缀（或唯一连续子串）。
无法精确定位时 trace 标为失败，相应特征为 null，不能猜测对齐。

字段局部均值包括所有与注册 byte span 有非空交集的生成 token。该规则允许边界
token 同时包含 JSON 标点或 LaTeX 标记，但保证可复算。V format repair 使用不同
条件提示；`retry_ok` 的 V logprob 和 critique 长度全部记为 missing。

精确字段、公式和 missingness 见 `features_scalar_single_v2.yaml`。

## 3. 多输出训练目标

训练主目标是：

```text
cap_outcomes = {M0: z0, M1: z1, M2: z2}
```

即固定顺序 `(z0,z1,z2)` 的三维 binary vector。`oracle_actions` 是所有值为 1
的 cap 集合；`minimal_oracle_action` 只用于最低权限诊断和平局解释，不得作为
普通三分类训练标签。完整规则见 `labels_cap_outcomes_v1.yaml`。

Controller training 可以预测三个 `P(zk=1 | x)`，模型选择和阈值只能使用开发/
validation labels。评估样本的标签继续在 Controller inference 之后产生并物理隔离。

## 4. Development300

`math500-r6e4ed1-l3to5-dev300-s20260910` 复用旧 `pilot300` 的完全相同 manifest
bytes，但在新 dataset 身份下明确允许探索、训练、交叉验证、阈值选择和 feature
ablation。它不再提供独立测试证据；确认性结论必须使用后续冻结且未参与调参的
独立 test dataset。

必须报告八种 `(z0,z1,z2)` pattern，以及 `z1-z0`、`z2-z1` 的 repair/harm
事件数。若反事实差异稀少，应报告不可识别性，不能用重复调参掩盖标签不足。
