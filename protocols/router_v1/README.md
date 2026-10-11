# Router v1 Protocol

`authority_v6` 的执行规则是 `effective_action = min(authority_cap, requested_action)`。
该映射使 Controller 只能否决 V，不能主动发起，因而 V 请求 KEEP 的行在测量之前就被
构造性地抹平（c013 为 156/300，c003 为 223/300）。`docs/analysis/gate1_c013.md`
记录了由此产生的功效后果。

`router_v1` 去掉该映射。它不注册 authority、不注册 request 动作、不做三级 cap。
一个模型先作答，再自查；router 只决定**哪些行值得付第二遍**。

c014 的首个 300 行 pilot 已在 Gate 0 停止：A0/A1 各答对 198 道，但只有 1 个 repair 和
1 个 harm，决策相关行 2/300，远低于注册门槛 20。该结果不支持训练 router；详见
`docs/analysis/gate0_c014.md`。

## 1. 两臂与执行

```text
pass 1   y0 = M(problem)                    无条件对每一行执行
pass 2   y_self = M(problem, y0)            router 决定是否执行
```

`y0` 无条件取 `sample_index = 0`，与 `authority_v6` 相同，不得改选第一个可解析样本。

pass 2 **总是输出一个 final answer**。模型判定 y0 正确时重新给出 y0，判定错误时给出
修改后的答案。因此 `y_self` 对每一行都有定义，"自查认为没问题却改坏了"这一类事件
可以被观测到——它正是 harm 通道，不得由协议抹掉。

固定策略只有两个，没有中间级：

```text
A0  永不自查      每行部署 y0
A1  总是自查      每行部署 y_self
```

router 是介于两者之间的策略族。

## 2. 时序与泄漏边界

router 的输入只能来自 pass 1，且必须在 pass 2 发生之前写入。禁止向 router 推理
暴露：gold、grader 输出、`u0`、`u_self`、`y_self` 或其任何派生量、problem_id、
run metadata。证据文件与评价标签文件物理分离，沿用 `authority_v6` 第 1 节。

pass 1 以 `temperature = 0` 且关闭 thinking 运行，因此采样 token 即 argmax。vLLM
普通响应归档 sampled-token logprob；同一生成 forward 内的 no-op logits processor 在 GPU
上把 raw tensor 归约成 sampled/max raw logit、logsumexp、entropy、top1-top2 margin 与
top-20 token/raw-logit。两路以同一 request capture ID 和生成 token ID 合并，并验证
`logprob = sampled_raw_logit - logsumexp`。full-vocabulary logits 不离开 GPU、不落盘；
第二次生成和 Transformers teacher-forced replay 都不属于 R2 来源。

## 3. 成本是一等结果

本协议的主结果是**在准确率不输 A1 的前提下，pass 2 的调用比例与 token 预算**。

理由是有效样本量，不是精确性：准确率差异只由两臂结果不同的 `R + H` 行承载，
而成本差异由**每一行**承载，因为每行都贡献它的 pass-2 token 乘以调用指示。

**成本不是无误差的。** 对一批固定的题，实际花掉多少 token 是精确的；但任何关于
总体的主张（"这类任务平均省 X%"）仍有抽样不确定性，换一批题会改变调用率与答案长度。
因此**准确率与成本都必须做 problem-level bootstrap CI**。

一个 router 必须对照**较强的那个**固定臂报告，而完美 router 相对 best fixed 的头寸是
`min(R, H)`，不是 `H`。平均效应接近零意味着两臂打平，那**恰恰是头寸最大的情形**，
不是没有 routing 价值。因此本协议不注册任何关于平均效应的去留判据。
见 `docs/plan/router_c014_design.md` 第 3、4 节。

每个 run 必须报告：

```text
tokens_pass1          逐行 output_tokens 之和
tokens_pass2          实际执行的 pass 2 的逐行 output_tokens 之和
router_call_fraction  实际调用 pass 2 的行数 / 总行数
```

成本一律只数 `output_tokens`；该字段已含 thinking token，不得再与
`reasoning_tokens` 相加。

## 4. 失败语义

pass 2 调用失败、解析失败或答案不可抽取时，记 `u_self = null`，**部署回退到 y0**，
因此该行的 A1 结果为 `u0`。这与 `authority_v6` 中失败 rewrite 回退 KEEP 的处理一致。
router 解析失败回退为"不调用 pass 2"。

## 5. 不可协商

- 数据 server-only，本地验证不得依赖数据。
- 已完成的 run 不可变；换模型、prompt、温度、特征契约或训练超参即新 condition，
  换种子即新 replicate。
- 复用有效的上游产物：只改 router 不重跑 pass 1 / pass 2；只改报告口径从 analysis 开始。
- 每个 analysis run 必须产出 `reports/summary.md`。
