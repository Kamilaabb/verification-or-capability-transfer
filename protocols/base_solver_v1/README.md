# Base Solver v1 Protocol

这个协议只测一次基础求解，不包含 verifier、pass 2、Controller 或 finalizer。

```text
y0 = M(problem)
```

`sample_index=0` 无条件作为 y0。模型响应中的 reasoning 与 final content 分开原样保存；
gold 和 grader 输出只在生成记录落盘后读取。

## 评分规则

- 若 final content 能抽取数学答案，就使用冻结的 math-verify grader 正常评分。
- `finish_reason=length` 只作为截断诊断，**不自动判错**；截断行只要仍能抽取答案就照常评分。
- 不能抽取答案记 `u0=0, grader_status=model_invalid`。
- grader 自身异常记 `u0=null, grader_status=grader_error`，不得静默当错。
- 不调用 finalizer，不做格式修复，不重试生成。

每个 analysis 必须报告准确率、截断率、截断且可评分/正确数量、不可抽取率、实际 output
tokens 以及 reasoning 是否被保存。development 结果不支持确认性主张。
