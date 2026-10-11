# Output Contract v1

目标：既保留可审计、可重算的信息，又让人不必直接阅读机器记录。

## 1. 分层保存

### G output

每题保存 Condition 配置的全部 G samples：raw output、extracted answer、parse status、sample index、seed、token usage、latency、finish reason、model revision、prompt hash 和 chat-template hash。`sample_index=0` 无条件是 y0。记录 schema 必须与 Condition 的样本数匹配。

### V report

每个 problem×V condition 保存：source G run、raw output、解析后的 report 或 null、parse/fallback 状态、seed、usage、model revision 和 prompt hash。所有权限共享这条记录。

### Rewrite

保存 source V record、是否运行、raw output、extracted y1、失败状态、seed 和 usage。KEEP 不运行；rewrite 模型失败时实际路径回退到 y0。

### Feature evidence

在 grading 之前单独生成。存储记录含对账所需的 record key、Condition 所选 feature contract 的精确字段、missingness 和 evidence hash。

真正传给任何 Controller 的 JSON 必须匹配 Condition 指定的 controller input schema：只有 `features` 和 `missingness`。`problem_id`、`cell_id`、run metadata 虽可存在于存储记录中，但绝不能进入 Controller payload。

### Counterfactual labels

在 grading 后保存 u0/u1/u2、z0/z1/z2、grader状态和所引用的 evidence hash。不得把整条 label record 传给 Controller。

### Controller

保存 C 类型、model/checkpoint、训练配置或 prompt、menu、replicate、call seed、raw/final output、cap、effective action、fallback、usage、source evidence hash 和实际 input payload hash；LLM C 的输出分别匹配 binary/ternary payload schema；不在 decision record 中写正确性标签。

### Run

保存 resolved condition、上游 run IDs、Git commit、运行环境、硬件、开始/结束时间、日志、文件 checksum 和验证状态。

## 2. 人类可读主结果

每个 analysis run 必须生成 `reports/summary.md`，顺序固定：

1. 一句话结论；
2. 本次数据、G/V/C 和条件；
3. 完整性、parse、grader、fallback gate；
4. Fixed M0/M1/M2、C2、C3 主表；
5. H1/H2/H3 的配对差值和区间；
6. repair、harm、权限与实际动作分布；
7. seed 稳定性；
8. tokens、延迟和实际部署成本；
9. 限制与下一步。

`metrics.json` 可以包含任意数量的机器指标，但不得替代 `summary.md`。

## 3. 最小目录

```text
runs/<run_id>/
├── RUN_MANIFEST.json
├── condition.resolved.yaml
├── checksums.sha256
├── logs/
├── records/
│   ├── g.jsonl
│   ├── v.jsonl
│   ├── rewrites.jsonl
│   ├── controller_evidence.jsonl
│   ├── counterfactuals.jsonl
│   └── controller_decisions.jsonl
└── reports/
    ├── summary.md
    ├── metrics.json
    └── qa_report.json
```

某个 stage 只产生自己负责的文件，并通过 source run ID 引用上游；不复制整个上游 bundle。

Condition 若通过 `extends + overrides` 编写，run 中仍必须保存完全展开后的 `condition.resolved.yaml`，不得要求复现者猜测当时的继承结果。
