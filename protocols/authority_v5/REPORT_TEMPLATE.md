# Authority Allocation Result

## 一句话结论

> 待填写：本轮是工程验证、探索性信号还是确认性结果；不要只写“效果更好”。

## 实验身份

|字段|值|
|---|---|
|Protocol||
|Dataset / split||
|Condition||
|Replicate plan||
|G||
|V condition||
|Controller||

## 数据质量

|问题数|完整记录|V fallback|rewrite failure|Controller fallback|grader error|
|---:|---:|---:|---:|---:|---:|
|待填写|待填写|待填写|待填写|待填写|待填写|

## 主结果

|策略|Accuracy|Repair|Harm|条件修复率|条件改坏率|平均部署 tokens|
|---|---:|---:|---:|---:|---:|---:|
|Fixed M0|||||||
|Fixed M1|||||||
|Fixed M2|||||||
|C2|||||||
|C3|||||||

## 核心比较

|命题|差值|95% CI|解释|
|---|---:|---:|---|
|H1: C3 − M2||||
|H2a: C3 − M0||||
|H2b: C3 − M1||||
|H2c: C3 − M2||||
|H3: C3 − C2||||

H2 只有在 H2a/H2b/H2c 都成立时才算成立；`best fixed` 可作摘要，不得替代三个预注册的配对比较。

## Seed 稳定性

列出每个 replicate 的 C2/C3 结果、均值、标准差和范围。不得只展示最佳 seed。

## 请求、权限和实际动作

简要解释 V 请求了什么、C 限制了什么，以及最终真正执行了什么。

## 成本

分别报告离线反事实生成成本和部署时实际选中路径成本。

## 限制与下一步

明确说明样本量、模型/任务适用范围、是否触碰开发数据，以及哪些结论尚不能声称。
