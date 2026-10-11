# 结果库清单

逐题数据在 `db/`，只留在服务器。本页只有计数，可以提交。

|域|配对数|设计配对|两方向都有的设计配对|方向（各切分合计）|
|---|---:|---:|---:|---:|
|math|45|12|12|62|
|knowledge|36|12|12|60|
|code|45|12|12|30|
|nl_reasoning|19|12|12|37|
|math_olympiad|2|0|0|4|

## 单独作答（正确/题数）

|域|模型|各切分|
|---|---|---|
|code|Llama-3.2-3B-Instruct|dev 205/832|
|code|Ministral-8B-Instruct-2410|dev 312/832|
|code|Mistral-Small-24B-Instruct-2501|dev 360/832|
|code|Qwen2.5-3B-Instruct|dev 259/832|
|code|Qwen2.5-7B-Instruct|dev 282/832|
|code|Qwen2.5-Coder-7B-Instruct|dev 339/832|
|code|Qwen3-32B|dev 364/832|
|code|Qwen3-4B|dev 316/832|
|code|Qwen3-8B|dev 353/832|
|code|Seed-Coder-8B-Instruct|dev 365/832|
|knowledge|Llama-3.2-3B-Instruct|dev 361/1000；test 600/1500|
|knowledge|Ministral-8B-Instruct-2410|dev 384/1000；test 628/1500|
|knowledge|Mistral-Small-24B-Instruct-2501|dev 670/1000；test 961/1500|
|knowledge|Qwen2.5-3B-Instruct|dev 426/1000；test 645/1500|
|knowledge|Qwen2.5-72B-Instruct|dev 722/1000；test 1084/1500|
|knowledge|Qwen2.5-7B-Instruct|dev 564/1000；test 858/1500|
|knowledge|Qwen3-32B|dev 694/1000；test 1050/1500|
|knowledge|Qwen3-4B|dev 605/1000；test 872/1500|
|knowledge|Qwen3-8B|dev 637/1000；test 960/1500|
|math|Llama-3.1-8B-Instruct|dev 403/1000；test 620/1500|
|math|Llama-3.2-3B-Instruct|dev 382/1000；test 609/1500|
|math|Ministral-8B-Instruct-2410|dev 493/1000；test 778/1500|
|math|Mistral-Small-24B-Instruct-2501|dev 642/1000；test 992/1500|
|math|Qwen2.5-3B-Instruct|dev 578/1000；test 901/1500|
|math|Qwen2.5-72B-Instruct|dev 763/1000；test 1181/1500|
|math|Qwen2.5-7B-Instruct|dev 682/1000；test 1039/1500|
|math|Qwen3-32B|dev 841/1000；test 1286/1500|
|math|Qwen3-4B|dev 769/1000；test 1188/1500|
|math|Qwen3-8B|dev 788/1000；test 1226/1500|
|math_olympiad|Ministral-8B-Instruct-2410|dev 160/571|
|math_olympiad|Qwen3-4B|dev 313/571|
|math_olympiad|Qwen3-8B|dev 303/571|
|nl_reasoning|Llama-3.2-3B-Instruct|dev 394/901|
|nl_reasoning|Ministral-8B-Instruct-2410|dev 467/901|
|nl_reasoning|Mistral-Small-24B-Instruct-2501|dev 645/901|
|nl_reasoning|Qwen2.5-3B-Instruct|dev 497/901|
|nl_reasoning|Qwen2.5-7B-Instruct|dev 556/901|
|nl_reasoning|Qwen3-32B|dev 701/901|
|nl_reasoning|Qwen3-4B|dev 609/901|
|nl_reasoning|Qwen3-8B|dev 665/901|

## 较强方无法确定的配对

- 无

## 缺少的设计配对方向

- 无

## 有歧义、未纳入的归档（需在 labels.yaml 的 prefer 里指定）

- 无

## 读取问题

- 无

## 忽略的归档

- 20260919T051824Z__c007__gate1__r005: no grades or no model
- 20260920T005750Z__c008__gate2__r006: no grades or no model
- 未登记的 (数据集, 清单) bigcodebench-b74c0d0-complete-s20260917 / gate0：2 个归档
- 未登记的 (数据集, 清单) bigcodebench-b74c0d0-instruct-s20260917 / 无：2 个归档 （已知不纳入：early Gate 0 pilots c005/c006）
- 未登记的 (数据集, 清单) bigcodebench-b74c0d0-instruct-s20260917 / gate0：1 个归档 （已知不纳入：early Gate 0 pilot c009 (300 rows)）
- 未登记的 (数据集, 清单) math500-r6e4ed1-l3to5-dev300-s20260910 / 无：2 个归档
- 未登记的 (数据集, 清单) math500-r6e4ed1-l3to5-dev300-s20260910 / pilot300：8 个归档
- 未登记的 (数据集, 清单) math500-r6e4ed1-l3to5-s20260910 / 无：3 个归档
- 未登记的 (数据集, 清单) mathfull-r465bcdb-l3to5-s20260924 / dev1000：31 个归档 【数据集已登记、清单名未登记，请检查 labels.yaml】
- 未登记的 (数据集, 清单) mmmupro-r563f3e8-std10-s20260926 / dev：4 个归档

## 选择器与 JEV：逐对与归档一致

每行保留 split（dev/test）及 evaluation（dev/test/cv）。代码、推理的 cv 标识共同比较题集；只有选择器经过交叉拟合。
H01–H08 的卡片与题集保持不变；RQ3/RQ4 将另行筛选数学、知识的 test1500。

|域|配对|题集|system|正确/题数|设计配对|
|---|---|---|---|---:|---|
|math|Qwen3-4B / Qwen3-8B|test|selector_primary|1246/1500|True|
|math|Qwen3-4B / Qwen3-8B|test|selector_secondary|1251/1500|True|
|math|Qwen3-32B / Qwen3-4B|test|selector_primary|1304/1500|False|
|math|Qwen3-32B / Qwen3-4B|test|selector_secondary|1298/1500|False|
|math|Qwen3-32B / Qwen3-8B|test|selector_primary|1306/1500|True|
|math|Qwen3-32B / Qwen3-8B|test|selector_secondary|1296/1500|True|
|math|Qwen2.5-3B-Instruct / Qwen3-4B|test|selector_primary|1195/1500|False|
|math|Qwen2.5-3B-Instruct / Qwen3-4B|test|selector_secondary|1191/1500|False|
|math|Qwen2.5-3B-Instruct / Qwen3-8B|test|selector_primary|1231/1500|False|
|math|Qwen2.5-3B-Instruct / Qwen3-8B|test|selector_secondary|1212/1500|False|
|math|Qwen2.5-3B-Instruct / Qwen2.5-7B-Instruct|test|selector_primary|1057/1500|True|
|math|Qwen2.5-3B-Instruct / Qwen2.5-7B-Instruct|test|selector_secondary|1049/1500|True|
|math|Mistral-Small-24B-Instruct-2501 / Qwen2.5-3B-Instruct|test|selector_primary|1029/1500|False|
|math|Mistral-Small-24B-Instruct-2501 / Qwen2.5-3B-Instruct|test|selector_secondary|1026/1500|False|
|math|Qwen2.5-3B-Instruct / Qwen3-32B|test|selector_primary|1291/1500|True|
|math|Qwen2.5-3B-Instruct / Qwen3-32B|test|selector_secondary|1280/1500|True|
|math|Qwen2.5-3B-Instruct / Qwen2.5-72B-Instruct|test|selector_primary|1182/1500|False|
|math|Qwen2.5-3B-Instruct / Qwen2.5-72B-Instruct|test|selector_secondary|1167/1500|False|
|math|Qwen2.5-7B-Instruct / Qwen3-4B|test|selector_primary|1205/1500|False|
|math|Qwen2.5-7B-Instruct / Qwen3-4B|test|selector_secondary|1198/1500|False|
|math|Qwen2.5-7B-Instruct / Qwen3-8B|test|selector_primary|1231/1500|False|
|math|Qwen2.5-7B-Instruct / Qwen3-8B|test|selector_secondary|1217/1500|False|
|math|Qwen2.5-7B-Instruct / Qwen3-32B|test|selector_primary|1297/1500|False|
|math|Qwen2.5-7B-Instruct / Qwen3-32B|test|selector_secondary|1275/1500|False|
|math|Qwen2.5-72B-Instruct / Qwen2.5-7B-Instruct|test|selector_primary|1194/1500|False|
|math|Qwen2.5-72B-Instruct / Qwen2.5-7B-Instruct|test|selector_secondary|1185/1500|False|
|math|Ministral-8B-Instruct-2410 / Qwen3-4B|test|selector_primary|1196/1500|False|
|math|Ministral-8B-Instruct-2410 / Qwen3-4B|test|selector_secondary|1193/1500|False|
|math|Ministral-8B-Instruct-2410 / Qwen3-8B|test|selector_primary|1233/1500|True|
|math|Ministral-8B-Instruct-2410 / Qwen3-8B|test|selector_secondary|1222/1500|True|
|math|Ministral-8B-Instruct-2410 / Qwen2.5-3B-Instruct|test|selector_primary|925/1500|True|
|math|Ministral-8B-Instruct-2410 / Qwen2.5-3B-Instruct|test|selector_secondary|920/1500|True|
|math|Ministral-8B-Instruct-2410 / Qwen2.5-7B-Instruct|test|selector_primary|1042/1500|False|
|math|Ministral-8B-Instruct-2410 / Qwen2.5-7B-Instruct|test|selector_secondary|1032/1500|False|
|math|Ministral-8B-Instruct-2410 / Mistral-Small-24B-Instruct-2501|test|selector_primary|1003/1500|False|
|math|Ministral-8B-Instruct-2410 / Mistral-Small-24B-Instruct-2501|test|selector_secondary|985/1500|False|
|math|Ministral-8B-Instruct-2410 / Qwen3-32B|test|selector_primary|1291/1500|False|
|math|Ministral-8B-Instruct-2410 / Qwen3-32B|test|selector_secondary|1287/1500|False|
|math|Ministral-8B-Instruct-2410 / Qwen2.5-72B-Instruct|test|selector_primary|1182/1500|False|
|math|Ministral-8B-Instruct-2410 / Qwen2.5-72B-Instruct|test|selector_secondary|1168/1500|False|
|math|Mistral-Small-24B-Instruct-2501 / Qwen3-4B|test|selector_primary|1193/1500|True|
|math|Mistral-Small-24B-Instruct-2501 / Qwen3-4B|test|selector_secondary|1187/1500|True|
|math|Mistral-Small-24B-Instruct-2501 / Qwen3-8B|test|selector_primary|1237/1500|False|
|math|Mistral-Small-24B-Instruct-2501 / Qwen3-8B|test|selector_secondary|1229/1500|False|
|math|Mistral-Small-24B-Instruct-2501 / Qwen2.5-7B-Instruct|test|selector_primary|1087/1500|True|
|math|Mistral-Small-24B-Instruct-2501 / Qwen2.5-7B-Instruct|test|selector_secondary|1078/1500|True|
|math|Mistral-Small-24B-Instruct-2501 / Qwen3-32B|test|selector_primary|1296/1500|False|
|math|Mistral-Small-24B-Instruct-2501 / Qwen3-32B|test|selector_secondary|1285/1500|False|
|math|Mistral-Small-24B-Instruct-2501 / Qwen2.5-72B-Instruct|test|selector_primary|1188/1500|False|
|math|Mistral-Small-24B-Instruct-2501 / Qwen2.5-72B-Instruct|test|selector_secondary|1179/1500|False|
|math|Llama-3.1-8B-Instruct / Qwen3-4B|test|selector_primary|1192/1500|False|
|math|Llama-3.1-8B-Instruct / Qwen3-4B|test|selector_secondary|1189/1500|False|
|math|Llama-3.1-8B-Instruct / Qwen3-8B|test|selector_primary|1230/1500|True|
|math|Llama-3.1-8B-Instruct / Qwen3-8B|test|selector_secondary|1228/1500|True|
|math|Llama-3.1-8B-Instruct / Qwen2.5-3B-Instruct|test|selector_primary|900/1500|False|
|math|Llama-3.1-8B-Instruct / Qwen2.5-3B-Instruct|test|selector_secondary|881/1500|False|
|math|Llama-3.1-8B-Instruct / Qwen2.5-7B-Instruct|test|selector_primary|1041/1500|False|
|math|Llama-3.1-8B-Instruct / Qwen2.5-7B-Instruct|test|selector_secondary|1032/1500|False|
|math|Llama-3.1-8B-Instruct / Ministral-8B-Instruct-2410|test|selector_primary|789/1500|False|
|math|Llama-3.1-8B-Instruct / Ministral-8B-Instruct-2410|test|selector_secondary|788/1500|False|
|math|Llama-3.1-8B-Instruct / Mistral-Small-24B-Instruct-2501|test|selector_primary|994/1500|False|
|math|Llama-3.1-8B-Instruct / Mistral-Small-24B-Instruct-2501|test|selector_secondary|983/1500|False|
|math|Llama-3.1-8B-Instruct / Qwen3-32B|test|selector_primary|1287/1500|False|
|math|Llama-3.1-8B-Instruct / Qwen3-32B|test|selector_secondary|1284/1500|False|
|math|Llama-3.1-8B-Instruct / Qwen2.5-72B-Instruct|test|selector_primary|1181/1500|False|
|math|Llama-3.1-8B-Instruct / Qwen2.5-72B-Instruct|test|selector_secondary|1173/1500|False|
|math|Llama-3.2-3B-Instruct / Qwen3-4B|test|selector_primary|1193/1500|False|
|math|Llama-3.2-3B-Instruct / Qwen3-4B|test|selector_secondary|1191/1500|False|
|math|Llama-3.2-3B-Instruct / Qwen3-8B|test|selector_primary|1230/1500|False|
|math|Llama-3.2-3B-Instruct / Qwen3-8B|test|selector_secondary|1230/1500|False|
|math|Llama-3.2-3B-Instruct / Qwen2.5-3B-Instruct|test|selector_primary|912/1500|False|
|math|Llama-3.2-3B-Instruct / Qwen2.5-3B-Instruct|test|selector_secondary|902/1500|False|
|math|Llama-3.2-3B-Instruct / Qwen2.5-7B-Instruct|test|selector_primary|1040/1500|True|
|math|Llama-3.2-3B-Instruct / Qwen2.5-7B-Instruct|test|selector_secondary|1024/1500|True|
|math|Llama-3.2-3B-Instruct / Ministral-8B-Instruct-2410|test|selector_primary|793/1500|True|
|math|Llama-3.2-3B-Instruct / Ministral-8B-Instruct-2410|test|selector_secondary|786/1500|True|
|math|Llama-3.2-3B-Instruct / Mistral-Small-24B-Instruct-2501|test|selector_primary|994/1500|False|
|math|Llama-3.2-3B-Instruct / Mistral-Small-24B-Instruct-2501|test|selector_secondary|991/1500|False|
|math|Llama-3.2-3B-Instruct / Qwen3-32B|test|selector_primary|1289/1500|False|
|math|Llama-3.2-3B-Instruct / Qwen3-32B|test|selector_secondary|1288/1500|False|
|math|Llama-3.1-8B-Instruct / Llama-3.2-3B-Instruct|test|selector_primary|668/1500|True|
|math|Llama-3.1-8B-Instruct / Llama-3.2-3B-Instruct|test|selector_secondary|668/1500|True|
|math|Llama-3.2-3B-Instruct / Qwen2.5-72B-Instruct|test|selector_primary|1182/1500|False|
|math|Llama-3.2-3B-Instruct / Qwen2.5-72B-Instruct|test|selector_secondary|1178/1500|False|
|math|Qwen2.5-72B-Instruct / Qwen3-4B|test|selector_primary|1228/1500|False|
|math|Qwen2.5-72B-Instruct / Qwen3-4B|test|selector_secondary|1229/1500|False|
|math|Qwen2.5-72B-Instruct / Qwen3-8B|test|selector_primary|1254/1500|False|
|math|Qwen2.5-72B-Instruct / Qwen3-8B|test|selector_secondary|1249/1500|False|
|math|Qwen2.5-72B-Instruct / Qwen3-32B|test|selector_primary|1309/1500|False|
|math|Qwen2.5-72B-Instruct / Qwen3-32B|test|selector_secondary|1301/1500|False|
|knowledge|Mistral-Small-24B-Instruct-2501 / Qwen3-4B|test|selector_primary|990/1500|False|
|knowledge|Mistral-Small-24B-Instruct-2501 / Qwen3-4B|test|selector_secondary|972/1500|False|
|knowledge|Qwen3-32B / Qwen3-4B|test|selector_primary|1084/1500|False|
|knowledge|Qwen3-32B / Qwen3-4B|test|selector_secondary|1066/1500|False|
|knowledge|Qwen2.5-72B-Instruct / Qwen3-4B|test|selector_primary|1081/1500|False|
|knowledge|Qwen2.5-72B-Instruct / Qwen3-4B|test|selector_secondary|1054/1500|False|
|knowledge|Qwen3-32B / Qwen3-8B|test|selector_primary|1085/1500|False|
|knowledge|Qwen3-32B / Qwen3-8B|test|selector_secondary|1080/1500|False|
|knowledge|Qwen2.5-72B-Instruct / Qwen3-8B|test|selector_primary|1095/1500|False|
|knowledge|Qwen2.5-72B-Instruct / Qwen3-8B|test|selector_secondary|1090/1500|False|
|knowledge|Llama-3.2-3B-Instruct / Qwen3-4B|test|selector_primary|886/1500|False|
|knowledge|Llama-3.2-3B-Instruct / Qwen3-4B|test|selector_secondary|886/1500|False|
|knowledge|Llama-3.2-3B-Instruct / Qwen2.5-7B-Instruct|test|selector_primary|865/1500|False|
|knowledge|Llama-3.2-3B-Instruct / Qwen2.5-7B-Instruct|test|selector_secondary|852/1500|False|
|knowledge|Llama-3.2-3B-Instruct / Ministral-8B-Instruct-2410|test|selector_primary|713/1500|False|
|knowledge|Llama-3.2-3B-Instruct / Ministral-8B-Instruct-2410|test|selector_secondary|702/1500|False|
|knowledge|Llama-3.2-3B-Instruct / Qwen3-32B|test|selector_primary|1067/1500|False|
|knowledge|Llama-3.2-3B-Instruct / Qwen3-32B|test|selector_secondary|1064/1500|False|
|knowledge|Llama-3.2-3B-Instruct / Qwen2.5-72B-Instruct|test|selector_primary|1081/1500|False|
|knowledge|Llama-3.2-3B-Instruct / Qwen2.5-72B-Instruct|test|selector_secondary|1078/1500|False|
|knowledge|Qwen2.5-3B-Instruct / Qwen3-4B|test|selector_primary|884/1500|False|
|knowledge|Qwen2.5-3B-Instruct / Qwen3-4B|test|selector_secondary|873/1500|False|
|knowledge|Qwen2.5-3B-Instruct / Qwen3-8B|test|selector_primary|982/1500|False|
|knowledge|Qwen2.5-3B-Instruct / Qwen3-8B|test|selector_secondary|957/1500|False|
|knowledge|Mistral-Small-24B-Instruct-2501 / Qwen2.5-3B-Instruct|test|selector_primary|966/1500|False|
|knowledge|Mistral-Small-24B-Instruct-2501 / Qwen2.5-3B-Instruct|test|selector_secondary|954/1500|False|
|knowledge|Qwen2.5-3B-Instruct / Qwen2.5-72B-Instruct|test|selector_primary|1082/1500|False|
|knowledge|Qwen2.5-3B-Instruct / Qwen2.5-72B-Instruct|test|selector_secondary|1077/1500|False|
|knowledge|Qwen2.5-7B-Instruct / Qwen3-32B|test|selector_primary|1090/1500|False|
|knowledge|Qwen2.5-7B-Instruct / Qwen3-32B|test|selector_secondary|1076/1500|False|
|knowledge|Qwen2.5-72B-Instruct / Qwen2.5-7B-Instruct|test|selector_primary|1084/1500|False|
|knowledge|Qwen2.5-72B-Instruct / Qwen2.5-7B-Instruct|test|selector_secondary|1079/1500|False|
|knowledge|Ministral-8B-Instruct-2410 / Qwen3-4B|test|selector_primary|883/1500|False|
|knowledge|Ministral-8B-Instruct-2410 / Qwen3-4B|test|selector_secondary|877/1500|False|
|knowledge|Ministral-8B-Instruct-2410 / Qwen3-8B|test|selector_primary|978/1500|False|
|knowledge|Ministral-8B-Instruct-2410 / Qwen3-8B|test|selector_secondary|963/1500|False|
|knowledge|Ministral-8B-Instruct-2410 / Qwen2.5-7B-Instruct|test|selector_primary|856/1500|False|
|knowledge|Ministral-8B-Instruct-2410 / Qwen2.5-7B-Instruct|test|selector_secondary|849/1500|False|
|knowledge|Ministral-8B-Instruct-2410 / Mistral-Small-24B-Instruct-2501|test|selector_primary|967/1500|False|
|knowledge|Ministral-8B-Instruct-2410 / Mistral-Small-24B-Instruct-2501|test|selector_secondary|957/1500|False|
|knowledge|Ministral-8B-Instruct-2410 / Qwen3-32B|test|selector_primary|1066/1500|False|
|knowledge|Ministral-8B-Instruct-2410 / Qwen3-32B|test|selector_secondary|1057/1500|False|
|knowledge|Ministral-8B-Instruct-2410 / Qwen2.5-72B-Instruct|test|selector_primary|1085/1500|False|
|knowledge|Ministral-8B-Instruct-2410 / Qwen2.5-72B-Instruct|test|selector_secondary|1082/1500|False|
|knowledge|Qwen2.5-72B-Instruct / Qwen3-32B|test|selector_primary|1118/1500|False|
|knowledge|Qwen2.5-72B-Instruct / Qwen3-32B|test|selector_secondary|1115/1500|False|
|knowledge|Qwen3-4B / Qwen3-8B|test|selector_primary|985/1500|False|
|knowledge|Qwen3-4B / Qwen3-8B|test|selector_secondary|967/1500|False|
|knowledge|Mistral-Small-24B-Instruct-2501 / Qwen3-8B|test|selector_primary|1020/1500|True|
|knowledge|Mistral-Small-24B-Instruct-2501 / Qwen3-8B|test|selector_secondary|1021/1500|True|
|knowledge|Llama-3.2-3B-Instruct / Qwen3-8B|test|selector_primary|976/1500|True|
|knowledge|Llama-3.2-3B-Instruct / Qwen3-8B|test|selector_secondary|963/1500|True|
|knowledge|Llama-3.2-3B-Instruct / Qwen2.5-3B-Instruct|test|selector_primary|702/1500|True|
|knowledge|Llama-3.2-3B-Instruct / Qwen2.5-3B-Instruct|test|selector_secondary|692/1500|True|
|knowledge|Llama-3.2-3B-Instruct / Mistral-Small-24B-Instruct-2501|test|selector_primary|963/1500|True|
|knowledge|Llama-3.2-3B-Instruct / Mistral-Small-24B-Instruct-2501|test|selector_secondary|956/1500|True|
|knowledge|Qwen2.5-3B-Instruct / Qwen2.5-7B-Instruct|test|selector_primary|853/1500|True|
|knowledge|Qwen2.5-3B-Instruct / Qwen2.5-7B-Instruct|test|selector_secondary|847/1500|True|
|knowledge|Qwen2.5-3B-Instruct / Qwen3-32B|test|selector_primary|1075/1500|True|
|knowledge|Qwen2.5-3B-Instruct / Qwen3-32B|test|selector_secondary|1071/1500|True|
|knowledge|Qwen2.5-7B-Instruct / Qwen3-4B|test|selector_primary|922/1500|True|
|knowledge|Qwen2.5-7B-Instruct / Qwen3-4B|test|selector_secondary|911/1500|True|
|knowledge|Qwen2.5-7B-Instruct / Qwen3-8B|test|selector_primary|994/1500|True|
|knowledge|Qwen2.5-7B-Instruct / Qwen3-8B|test|selector_secondary|977/1500|True|
|knowledge|Mistral-Small-24B-Instruct-2501 / Qwen2.5-7B-Instruct|test|selector_primary|981/1500|True|
|knowledge|Mistral-Small-24B-Instruct-2501 / Qwen2.5-7B-Instruct|test|selector_secondary|969/1500|True|
|knowledge|Ministral-8B-Instruct-2410 / Qwen2.5-3B-Instruct|test|selector_primary|725/1500|True|
|knowledge|Ministral-8B-Instruct-2410 / Qwen2.5-3B-Instruct|test|selector_secondary|719/1500|True|
|knowledge|Mistral-Small-24B-Instruct-2501 / Qwen3-32B|test|selector_primary|1083/1500|True|
|knowledge|Mistral-Small-24B-Instruct-2501 / Qwen3-32B|test|selector_secondary|1074/1500|True|
|knowledge|Mistral-Small-24B-Instruct-2501 / Qwen2.5-72B-Instruct|test|selector_primary|1094/1500|True|
|knowledge|Mistral-Small-24B-Instruct-2501 / Qwen2.5-72B-Instruct|test|selector_secondary|1084/1500|True|
|code|Qwen3-4B / Qwen3-8B|cv|selector_primary|358/832|False|
|code|Qwen3-4B / Qwen3-8B|cv|selector_secondary|360/832|False|
|code|Mistral-Small-24B-Instruct-2501 / Qwen3-4B|cv|selector_primary|355/832|False|
|code|Mistral-Small-24B-Instruct-2501 / Qwen3-4B|cv|selector_secondary|355/832|False|
|code|Qwen3-32B / Qwen3-4B|cv|selector_primary|358/832|False|
|code|Qwen3-32B / Qwen3-4B|cv|selector_secondary|362/832|False|
|code|Qwen2.5-Coder-7B-Instruct / Qwen3-4B|cv|selector_primary|344/832|False|
|code|Qwen2.5-Coder-7B-Instruct / Qwen3-4B|cv|selector_secondary|343/832|False|
|code|Qwen3-4B / Seed-Coder-8B-Instruct|cv|selector_primary|359/832|False|
|code|Qwen3-4B / Seed-Coder-8B-Instruct|cv|selector_secondary|359/832|False|
|code|Mistral-Small-24B-Instruct-2501 / Qwen3-8B|cv|selector_primary|368/832|True|
|code|Mistral-Small-24B-Instruct-2501 / Qwen3-8B|cv|selector_secondary|364/832|True|
|code|Qwen3-32B / Qwen3-8B|cv|selector_primary|366/832|False|
|code|Qwen3-32B / Qwen3-8B|cv|selector_secondary|366/832|False|
|code|Qwen3-8B / Seed-Coder-8B-Instruct|cv|selector_primary|376/832|False|
|code|Qwen3-8B / Seed-Coder-8B-Instruct|cv|selector_secondary|370/832|False|
|code|Llama-3.2-3B-Instruct / Qwen3-4B|cv|selector_primary|317/832|False|
|code|Llama-3.2-3B-Instruct / Qwen3-4B|cv|selector_secondary|311/832|False|
|code|Llama-3.2-3B-Instruct / Qwen3-8B|cv|selector_primary|350/832|True|
|code|Llama-3.2-3B-Instruct / Qwen3-8B|cv|selector_secondary|342/832|True|
|code|Llama-3.2-3B-Instruct / Qwen2.5-3B-Instruct|cv|selector_primary|248/832|False|
|code|Llama-3.2-3B-Instruct / Qwen2.5-3B-Instruct|cv|selector_secondary|252/832|False|
|code|Llama-3.2-3B-Instruct / Qwen2.5-7B-Instruct|cv|selector_primary|287/832|True|
|code|Llama-3.2-3B-Instruct / Qwen2.5-7B-Instruct|cv|selector_secondary|280/832|True|
|code|Llama-3.2-3B-Instruct / Ministral-8B-Instruct-2410|cv|selector_primary|308/832|False|
|code|Llama-3.2-3B-Instruct / Ministral-8B-Instruct-2410|cv|selector_secondary|300/832|False|
|code|Llama-3.2-3B-Instruct / Mistral-Small-24B-Instruct-2501|cv|selector_primary|360/832|False|
|code|Llama-3.2-3B-Instruct / Mistral-Small-24B-Instruct-2501|cv|selector_secondary|356/832|False|
|code|Llama-3.2-3B-Instruct / Qwen3-32B|cv|selector_primary|362/832|False|
|code|Llama-3.2-3B-Instruct / Qwen3-32B|cv|selector_secondary|364/832|False|
|code|Llama-3.2-3B-Instruct / Qwen2.5-Coder-7B-Instruct|cv|selector_primary|339/832|False|
|code|Llama-3.2-3B-Instruct / Qwen2.5-Coder-7B-Instruct|cv|selector_secondary|337/832|False|
|code|Llama-3.2-3B-Instruct / Seed-Coder-8B-Instruct|cv|selector_primary|363/832|True|
|code|Llama-3.2-3B-Instruct / Seed-Coder-8B-Instruct|cv|selector_secondary|364/832|True|
|code|Qwen2.5-3B-Instruct / Qwen3-4B|cv|selector_primary|310/832|False|
|code|Qwen2.5-3B-Instruct / Qwen3-4B|cv|selector_secondary|308/832|False|
|code|Qwen2.5-3B-Instruct / Qwen3-8B|cv|selector_primary|346/832|False|
|code|Qwen2.5-3B-Instruct / Qwen3-8B|cv|selector_secondary|346/832|False|
|code|Qwen2.5-3B-Instruct / Qwen2.5-7B-Instruct|cv|selector_primary|285/832|True|
|code|Qwen2.5-3B-Instruct / Qwen2.5-7B-Instruct|cv|selector_secondary|284/832|True|
|code|Ministral-8B-Instruct-2410 / Qwen2.5-3B-Instruct|cv|selector_primary|301/832|False|
|code|Ministral-8B-Instruct-2410 / Qwen2.5-3B-Instruct|cv|selector_secondary|297/832|False|
|code|Mistral-Small-24B-Instruct-2501 / Qwen2.5-3B-Instruct|cv|selector_primary|354/832|False|
|code|Mistral-Small-24B-Instruct-2501 / Qwen2.5-3B-Instruct|cv|selector_secondary|348/832|False|
|code|Qwen2.5-3B-Instruct / Qwen3-32B|cv|selector_primary|364/832|False|
|code|Qwen2.5-3B-Instruct / Qwen3-32B|cv|selector_secondary|347/832|False|
|code|Qwen2.5-3B-Instruct / Qwen2.5-Coder-7B-Instruct|cv|selector_primary|337/832|False|
|code|Qwen2.5-3B-Instruct / Qwen2.5-Coder-7B-Instruct|cv|selector_secondary|336/832|False|
|code|Qwen2.5-3B-Instruct / Seed-Coder-8B-Instruct|cv|selector_primary|358/832|False|
|code|Qwen2.5-3B-Instruct / Seed-Coder-8B-Instruct|cv|selector_secondary|357/832|False|
|code|Qwen2.5-7B-Instruct / Qwen3-4B|cv|selector_primary|317/832|True|
|code|Qwen2.5-7B-Instruct / Qwen3-4B|cv|selector_secondary|317/832|True|
|code|Qwen2.5-7B-Instruct / Qwen3-8B|cv|selector_primary|358/832|False|
|code|Qwen2.5-7B-Instruct / Qwen3-8B|cv|selector_secondary|351/832|False|
|code|Ministral-8B-Instruct-2410 / Qwen2.5-7B-Instruct|cv|selector_primary|305/832|False|
|code|Ministral-8B-Instruct-2410 / Qwen2.5-7B-Instruct|cv|selector_secondary|312/832|False|
|code|Mistral-Small-24B-Instruct-2501 / Qwen2.5-7B-Instruct|cv|selector_primary|359/832|False|
|code|Mistral-Small-24B-Instruct-2501 / Qwen2.5-7B-Instruct|cv|selector_secondary|349/832|False|
|code|Qwen2.5-7B-Instruct / Qwen3-32B|cv|selector_primary|366/832|True|
|code|Qwen2.5-7B-Instruct / Qwen3-32B|cv|selector_secondary|366/832|True|
|code|Qwen2.5-7B-Instruct / Qwen2.5-Coder-7B-Instruct|cv|selector_primary|339/832|False|
|code|Qwen2.5-7B-Instruct / Qwen2.5-Coder-7B-Instruct|cv|selector_secondary|336/832|False|
|code|Qwen2.5-7B-Instruct / Seed-Coder-8B-Instruct|cv|selector_primary|361/832|False|
|code|Qwen2.5-7B-Instruct / Seed-Coder-8B-Instruct|cv|selector_secondary|358/832|False|
|code|Ministral-8B-Instruct-2410 / Qwen3-4B|cv|selector_primary|330/832|True|
|code|Ministral-8B-Instruct-2410 / Qwen3-4B|cv|selector_secondary|330/832|True|
|code|Ministral-8B-Instruct-2410 / Qwen3-8B|cv|selector_primary|353/832|False|
|code|Ministral-8B-Instruct-2410 / Qwen3-8B|cv|selector_secondary|353/832|False|
|code|Ministral-8B-Instruct-2410 / Mistral-Small-24B-Instruct-2501|cv|selector_primary|353/832|True|
|code|Ministral-8B-Instruct-2410 / Mistral-Small-24B-Instruct-2501|cv|selector_secondary|353/832|True|
|code|Ministral-8B-Instruct-2410 / Qwen3-32B|cv|selector_primary|363/832|False|
|code|Ministral-8B-Instruct-2410 / Qwen3-32B|cv|selector_secondary|363/832|False|
|code|Ministral-8B-Instruct-2410 / Qwen2.5-Coder-7B-Instruct|cv|selector_primary|342/832|True|
|code|Ministral-8B-Instruct-2410 / Qwen2.5-Coder-7B-Instruct|cv|selector_secondary|342/832|True|
|code|Ministral-8B-Instruct-2410 / Seed-Coder-8B-Instruct|cv|selector_primary|364/832|False|
|code|Ministral-8B-Instruct-2410 / Seed-Coder-8B-Instruct|cv|selector_secondary|365/832|False|
|code|Mistral-Small-24B-Instruct-2501 / Qwen3-32B|cv|selector_primary|377/832|True|
|code|Mistral-Small-24B-Instruct-2501 / Qwen3-32B|cv|selector_secondary|379/832|True|
|code|Mistral-Small-24B-Instruct-2501 / Seed-Coder-8B-Instruct|cv|selector_primary|373/832|True|
|code|Mistral-Small-24B-Instruct-2501 / Seed-Coder-8B-Instruct|cv|selector_secondary|370/832|True|
|code|Qwen3-32B / Seed-Coder-8B-Instruct|cv|selector_primary|365/832|False|
|code|Qwen3-32B / Seed-Coder-8B-Instruct|cv|selector_secondary|371/832|False|
|code|Qwen2.5-Coder-7B-Instruct / Qwen3-8B|cv|selector_primary|365/832|False|
|code|Qwen2.5-Coder-7B-Instruct / Qwen3-8B|cv|selector_secondary|363/832|False|
|code|Mistral-Small-24B-Instruct-2501 / Qwen2.5-Coder-7B-Instruct|cv|selector_primary|368/832|False|
|code|Mistral-Small-24B-Instruct-2501 / Qwen2.5-Coder-7B-Instruct|cv|selector_secondary|362/832|False|
|code|Qwen2.5-Coder-7B-Instruct / Qwen3-32B|cv|selector_primary|371/832|False|
|code|Qwen2.5-Coder-7B-Instruct / Qwen3-32B|cv|selector_secondary|377/832|False|
|code|Qwen2.5-Coder-7B-Instruct / Seed-Coder-8B-Instruct|cv|selector_primary|361/832|False|
|code|Qwen2.5-Coder-7B-Instruct / Seed-Coder-8B-Instruct|cv|selector_secondary|364/832|False|
|nl_reasoning|Llama-3.2-3B-Instruct / Qwen3-4B|cv|selector_primary|610/901|True|
|nl_reasoning|Llama-3.2-3B-Instruct / Qwen3-4B|cv|selector_secondary|607/901|True|
|nl_reasoning|Llama-3.2-3B-Instruct / Qwen2.5-3B-Instruct|cv|selector_primary|497/901|True|
|nl_reasoning|Llama-3.2-3B-Instruct / Qwen2.5-3B-Instruct|cv|selector_secondary|497/901|True|
|nl_reasoning|Llama-3.2-3B-Instruct / Qwen3-32B|cv|selector_primary|707/901|True|
|nl_reasoning|Llama-3.2-3B-Instruct / Qwen3-32B|cv|selector_secondary|700/901|True|
|nl_reasoning|Qwen3-32B / Qwen3-8B|cv|selector_primary|719/901|True|
|nl_reasoning|Qwen3-32B / Qwen3-8B|cv|selector_secondary|711/901|True|
|nl_reasoning|Qwen3-4B / Qwen3-8B|cv|selector_primary|677/901|True|
|nl_reasoning|Qwen3-4B / Qwen3-8B|cv|selector_secondary|669/901|True|
|nl_reasoning|Mistral-Small-24B-Instruct-2501 / Qwen3-4B|cv|selector_primary|654/901|True|
|nl_reasoning|Mistral-Small-24B-Instruct-2501 / Qwen3-4B|cv|selector_secondary|653/901|True|
|nl_reasoning|Qwen3-32B / Qwen3-4B|cv|selector_primary|705/901|True|
|nl_reasoning|Qwen3-32B / Qwen3-4B|cv|selector_secondary|699/901|True|
|nl_reasoning|Qwen2.5-3B-Instruct / Qwen2.5-7B-Instruct|cv|selector_primary|557/901|True|
|nl_reasoning|Qwen2.5-3B-Instruct / Qwen2.5-7B-Instruct|cv|selector_secondary|557/901|True|
|nl_reasoning|Mistral-Small-24B-Instruct-2501 / Qwen2.5-3B-Instruct|cv|selector_primary|648/901|True|
|nl_reasoning|Mistral-Small-24B-Instruct-2501 / Qwen2.5-3B-Instruct|cv|selector_secondary|639/901|True|
|nl_reasoning|Qwen2.5-7B-Instruct / Qwen3-4B|cv|selector_primary|613/901|True|
|nl_reasoning|Qwen2.5-7B-Instruct / Qwen3-4B|cv|selector_secondary|604/901|True|
|nl_reasoning|Ministral-8B-Instruct-2410 / Qwen2.5-3B-Instruct|cv|selector_primary|515/901|True|
|nl_reasoning|Ministral-8B-Instruct-2410 / Qwen2.5-3B-Instruct|cv|selector_secondary|515/901|True|
|nl_reasoning|Mistral-Small-24B-Instruct-2501 / Qwen3-8B|cv|selector_primary|683/901|True|
|nl_reasoning|Mistral-Small-24B-Instruct-2501 / Qwen3-8B|cv|selector_secondary|679/901|True|
|math|Qwen3-4B / Qwen3-8B|test|JEV|1304/1500|True|
|math|Qwen3-32B / Qwen3-4B|test|JEV|1332/1500|False|
|math|Qwen3-32B / Qwen3-8B|test|JEV|1335/1500|True|
|math|Qwen2.5-3B-Instruct / Qwen3-4B|test|JEV|1213/1500|False|
|math|Qwen2.5-3B-Instruct / Qwen3-8B|test|JEV|1240/1500|False|
|math|Qwen2.5-3B-Instruct / Qwen2.5-7B-Instruct|test|JEV|1093/1500|True|
|math|Mistral-Small-24B-Instruct-2501 / Qwen2.5-3B-Instruct|test|JEV|1071/1500|False|
|math|Qwen2.5-3B-Instruct / Qwen3-32B|test|JEV|1293/1500|True|
|math|Qwen2.5-3B-Instruct / Qwen2.5-72B-Instruct|test|JEV|1187/1500|False|
|math|Qwen2.5-7B-Instruct / Qwen3-4B|test|JEV|1250/1500|False|
|math|Qwen2.5-7B-Instruct / Qwen3-8B|test|JEV|1268/1500|False|
|math|Qwen2.5-7B-Instruct / Qwen3-32B|test|JEV|1316/1500|False|
|math|Qwen2.5-72B-Instruct / Qwen2.5-7B-Instruct|test|JEV|1216/1500|False|
|math|Ministral-8B-Instruct-2410 / Qwen3-4B|test|JEV|1215/1500|False|
|math|Ministral-8B-Instruct-2410 / Qwen3-8B|test|JEV|1247/1500|True|
|math|Ministral-8B-Instruct-2410 / Qwen2.5-3B-Instruct|test|JEV|986/1500|True|
|math|Ministral-8B-Instruct-2410 / Qwen2.5-7B-Instruct|test|JEV|1075/1500|False|
|math|Ministral-8B-Instruct-2410 / Mistral-Small-24B-Instruct-2501|test|JEV|1033/1500|False|
|math|Ministral-8B-Instruct-2410 / Qwen3-32B|test|JEV|1297/1500|False|
|math|Ministral-8B-Instruct-2410 / Qwen2.5-72B-Instruct|test|JEV|1195/1500|False|
|math|Mistral-Small-24B-Instruct-2501 / Qwen3-4B|test|JEV|1236/1500|True|
|math|Mistral-Small-24B-Instruct-2501 / Qwen3-8B|test|JEV|1266/1500|False|
|math|Mistral-Small-24B-Instruct-2501 / Qwen2.5-7B-Instruct|test|JEV|1132/1500|True|
|math|Mistral-Small-24B-Instruct-2501 / Qwen3-32B|test|JEV|1303/1500|False|
|math|Mistral-Small-24B-Instruct-2501 / Qwen2.5-72B-Instruct|test|JEV|1212/1500|False|
|math|Llama-3.1-8B-Instruct / Qwen3-4B|test|JEV|1197/1500|False|
|math|Llama-3.1-8B-Instruct / Qwen3-8B|test|JEV|1236/1500|True|
|math|Llama-3.1-8B-Instruct / Qwen2.5-3B-Instruct|test|JEV|947/1500|False|
|math|Llama-3.1-8B-Instruct / Qwen2.5-7B-Instruct|test|JEV|1061/1500|False|
|math|Llama-3.1-8B-Instruct / Ministral-8B-Instruct-2410|test|JEV|851/1500|False|
|math|Llama-3.1-8B-Instruct / Mistral-Small-24B-Instruct-2501|test|JEV|1014/1500|False|
|math|Llama-3.1-8B-Instruct / Qwen3-32B|test|JEV|1290/1500|False|
|math|Llama-3.1-8B-Instruct / Qwen2.5-72B-Instruct|test|JEV|1187/1500|False|
|math|Llama-3.2-3B-Instruct / Qwen3-4B|test|JEV|1201/1500|False|
|math|Llama-3.2-3B-Instruct / Qwen3-8B|test|JEV|1234/1500|False|
|math|Llama-3.2-3B-Instruct / Qwen2.5-3B-Instruct|test|JEV|951/1500|False|
|math|Llama-3.2-3B-Instruct / Qwen2.5-7B-Instruct|test|JEV|1056/1500|True|
|math|Llama-3.2-3B-Instruct / Ministral-8B-Instruct-2410|test|JEV|854/1500|True|
|math|Llama-3.2-3B-Instruct / Mistral-Small-24B-Instruct-2501|test|JEV|1018/1500|False|
|math|Llama-3.2-3B-Instruct / Qwen3-32B|test|JEV|1291/1500|False|
|math|Llama-3.1-8B-Instruct / Llama-3.2-3B-Instruct|test|JEV|752/1500|True|
|math|Llama-3.2-3B-Instruct / Qwen2.5-72B-Instruct|test|JEV|1187/1500|False|
|math|Qwen2.5-72B-Instruct / Qwen3-4B|test|JEV|1276/1500|False|
|math|Qwen2.5-72B-Instruct / Qwen3-8B|test|JEV|1300/1500|False|
|math|Qwen2.5-72B-Instruct / Qwen3-32B|test|JEV|1319/1500|False|
|knowledge|Qwen3-4B / Qwen3-8B|test|JEV|1026/1500|False|
|knowledge|Mistral-Small-24B-Instruct-2501 / Qwen3-4B|test|JEV|1058/1500|False|
|knowledge|Qwen3-32B / Qwen3-4B|test|JEV|1098/1500|False|
|knowledge|Qwen2.5-72B-Instruct / Qwen3-4B|test|JEV|1128/1500|False|
|knowledge|Mistral-Small-24B-Instruct-2501 / Qwen3-8B|test|JEV|1071/1500|True|
|knowledge|Qwen3-32B / Qwen3-8B|test|JEV|1093/1500|False|
|knowledge|Qwen2.5-72B-Instruct / Qwen3-8B|test|JEV|1129/1500|False|
|knowledge|Llama-3.2-3B-Instruct / Qwen3-4B|test|JEV|947/1500|False|
|knowledge|Llama-3.2-3B-Instruct / Qwen3-8B|test|JEV|1010/1500|True|
|knowledge|Llama-3.2-3B-Instruct / Qwen2.5-3B-Instruct|test|JEV|809/1500|True|
|knowledge|Llama-3.2-3B-Instruct / Qwen2.5-7B-Instruct|test|JEV|923/1500|False|
|knowledge|Llama-3.2-3B-Instruct / Ministral-8B-Instruct-2410|test|JEV|763/1500|False|
|knowledge|Llama-3.2-3B-Instruct / Mistral-Small-24B-Instruct-2501|test|JEV|1005/1500|True|
|knowledge|Llama-3.2-3B-Instruct / Qwen3-32B|test|JEV|1077/1500|False|
|knowledge|Llama-3.2-3B-Instruct / Qwen2.5-72B-Instruct|test|JEV|1103/1500|False|
|knowledge|Qwen2.5-3B-Instruct / Qwen3-4B|test|JEV|937/1500|False|
|knowledge|Qwen2.5-3B-Instruct / Qwen3-8B|test|JEV|997/1500|False|
|knowledge|Qwen2.5-3B-Instruct / Qwen2.5-7B-Instruct|test|JEV|919/1500|True|
|knowledge|Mistral-Small-24B-Instruct-2501 / Qwen2.5-3B-Instruct|test|JEV|1002/1500|False|
|knowledge|Qwen2.5-3B-Instruct / Qwen3-32B|test|JEV|1074/1500|True|
|knowledge|Qwen2.5-3B-Instruct / Qwen2.5-72B-Instruct|test|JEV|1104/1500|False|
|knowledge|Qwen2.5-7B-Instruct / Qwen3-4B|test|JEV|1002/1500|True|
|knowledge|Qwen2.5-7B-Instruct / Qwen3-8B|test|JEV|1038/1500|True|
|knowledge|Mistral-Small-24B-Instruct-2501 / Qwen2.5-7B-Instruct|test|JEV|1041/1500|True|
|knowledge|Qwen2.5-7B-Instruct / Qwen3-32B|test|JEV|1097/1500|False|
|knowledge|Qwen2.5-72B-Instruct / Qwen2.5-7B-Instruct|test|JEV|1121/1500|False|
|knowledge|Ministral-8B-Instruct-2410 / Qwen3-4B|test|JEV|936/1500|False|
|knowledge|Ministral-8B-Instruct-2410 / Qwen3-8B|test|JEV|1005/1500|False|
|knowledge|Ministral-8B-Instruct-2410 / Qwen2.5-3B-Instruct|test|JEV|800/1500|True|
|knowledge|Ministral-8B-Instruct-2410 / Qwen2.5-7B-Instruct|test|JEV|913/1500|False|
|knowledge|Ministral-8B-Instruct-2410 / Mistral-Small-24B-Instruct-2501|test|JEV|979/1500|False|
|knowledge|Ministral-8B-Instruct-2410 / Qwen3-32B|test|JEV|1079/1500|False|
|knowledge|Ministral-8B-Instruct-2410 / Qwen2.5-72B-Instruct|test|JEV|1087/1500|False|
|knowledge|Mistral-Small-24B-Instruct-2501 / Qwen3-32B|test|JEV|1101/1500|True|
|knowledge|Mistral-Small-24B-Instruct-2501 / Qwen2.5-72B-Instruct|test|JEV|1126/1500|True|
|knowledge|Qwen2.5-72B-Instruct / Qwen3-32B|test|JEV|1145/1500|False|
|code|Llama-3.2-3B-Instruct / Ministral-8B-Instruct-2410|cv|JEV|311/832|False|
|code|Llama-3.2-3B-Instruct / Mistral-Small-24B-Instruct-2501|cv|JEV|355/832|False|
|code|Llama-3.2-3B-Instruct / Qwen2.5-3B-Instruct|cv|JEV|283/832|False|
|code|Llama-3.2-3B-Instruct / Qwen2.5-7B-Instruct|cv|JEV|309/832|True|
|code|Llama-3.2-3B-Instruct / Qwen2.5-Coder-7B-Instruct|cv|JEV|333/832|False|
|code|Llama-3.2-3B-Instruct / Qwen3-32B|cv|JEV|362/832|False|
|code|Llama-3.2-3B-Instruct / Qwen3-4B|cv|JEV|323/832|False|
|code|Llama-3.2-3B-Instruct / Qwen3-8B|cv|JEV|355/832|True|
|code|Llama-3.2-3B-Instruct / Seed-Coder-8B-Instruct|cv|JEV|359/832|True|
|code|Ministral-8B-Instruct-2410 / Mistral-Small-24B-Instruct-2501|cv|JEV|372/832|True|
|code|Ministral-8B-Instruct-2410 / Qwen2.5-Coder-7B-Instruct|cv|JEV|355/832|True|
|code|Ministral-8B-Instruct-2410 / Qwen3-32B|cv|JEV|366/832|False|
|code|Ministral-8B-Instruct-2410 / Qwen3-4B|cv|JEV|354/832|True|
|code|Ministral-8B-Instruct-2410 / Qwen3-8B|cv|JEV|364/832|False|
|code|Ministral-8B-Instruct-2410 / Seed-Coder-8B-Instruct|cv|JEV|372/832|False|
|code|Mistral-Small-24B-Instruct-2501 / Qwen3-32B|cv|JEV|380/832|True|
|code|Mistral-Small-24B-Instruct-2501 / Seed-Coder-8B-Instruct|cv|JEV|382/832|True|
|code|Ministral-8B-Instruct-2410 / Qwen2.5-3B-Instruct|cv|JEV|324/832|False|
|code|Mistral-Small-24B-Instruct-2501 / Qwen2.5-3B-Instruct|cv|JEV|368/832|False|
|code|Qwen2.5-3B-Instruct / Qwen2.5-7B-Instruct|cv|JEV|316/832|True|
|code|Qwen2.5-3B-Instruct / Qwen2.5-Coder-7B-Instruct|cv|JEV|345/832|False|
|code|Qwen2.5-3B-Instruct / Qwen3-32B|cv|JEV|371/832|False|
|code|Qwen2.5-3B-Instruct / Qwen3-4B|cv|JEV|341/832|False|
|code|Qwen2.5-3B-Instruct / Qwen3-8B|cv|JEV|355/832|False|
|code|Qwen2.5-3B-Instruct / Seed-Coder-8B-Instruct|cv|JEV|360/832|False|
|code|Ministral-8B-Instruct-2410 / Qwen2.5-7B-Instruct|cv|JEV|331/832|False|
|code|Mistral-Small-24B-Instruct-2501 / Qwen2.5-7B-Instruct|cv|JEV|370/832|False|
|code|Qwen2.5-7B-Instruct / Qwen2.5-Coder-7B-Instruct|cv|JEV|349/832|False|
|code|Qwen2.5-7B-Instruct / Qwen3-32B|cv|JEV|370/832|True|
|code|Qwen2.5-7B-Instruct / Qwen3-4B|cv|JEV|341/832|True|
|code|Qwen2.5-7B-Instruct / Qwen3-8B|cv|JEV|364/832|False|
|code|Qwen2.5-7B-Instruct / Seed-Coder-8B-Instruct|cv|JEV|368/832|False|
|code|Mistral-Small-24B-Instruct-2501 / Qwen2.5-Coder-7B-Instruct|cv|JEV|382/832|False|
|code|Qwen2.5-Coder-7B-Instruct / Qwen3-32B|cv|JEV|383/832|False|
|code|Qwen2.5-Coder-7B-Instruct / Qwen3-8B|cv|JEV|370/832|False|
|code|Qwen2.5-Coder-7B-Instruct / Seed-Coder-8B-Instruct|cv|JEV|382/832|False|
|code|Qwen3-32B / Seed-Coder-8B-Instruct|cv|JEV|380/832|False|
|code|Mistral-Small-24B-Instruct-2501 / Qwen3-4B|cv|JEV|366/832|False|
|code|Qwen2.5-Coder-7B-Instruct / Qwen3-4B|cv|JEV|357/832|False|
|code|Qwen3-32B / Qwen3-4B|cv|JEV|359/832|False|
|code|Qwen3-4B / Qwen3-8B|cv|JEV|372/832|False|
|code|Qwen3-4B / Seed-Coder-8B-Instruct|cv|JEV|373/832|False|
|code|Mistral-Small-24B-Instruct-2501 / Qwen3-8B|cv|JEV|371/832|True|
|code|Qwen3-32B / Qwen3-8B|cv|JEV|376/832|False|
|code|Qwen3-8B / Seed-Coder-8B-Instruct|cv|JEV|384/832|False|
|nl_reasoning|Llama-3.2-3B-Instruct / Qwen3-4B|cv|JEV|639/901|True|
|nl_reasoning|Llama-3.2-3B-Instruct / Qwen2.5-3B-Instruct|cv|JEV|568/901|True|
|nl_reasoning|Llama-3.2-3B-Instruct / Qwen3-32B|cv|JEV|713/901|True|
|nl_reasoning|Qwen3-32B / Qwen3-8B|cv|JEV|729/901|True|
|nl_reasoning|Qwen3-4B / Qwen3-8B|cv|JEV|690/901|True|
|nl_reasoning|Mistral-Small-24B-Instruct-2501 / Qwen3-4B|cv|JEV|710/901|True|
|nl_reasoning|Qwen3-32B / Qwen3-4B|cv|JEV|724/901|True|
|nl_reasoning|Qwen2.5-3B-Instruct / Qwen2.5-7B-Instruct|cv|JEV|626/901|True|
|nl_reasoning|Mistral-Small-24B-Instruct-2501 / Qwen2.5-3B-Instruct|cv|JEV|689/901|True|
|nl_reasoning|Qwen2.5-7B-Instruct / Qwen3-4B|cv|JEV|661/901|True|
|nl_reasoning|Ministral-8B-Instruct-2410 / Qwen2.5-3B-Instruct|cv|JEV|567/901|True|
|nl_reasoning|Mistral-Small-24B-Instruct-2501 / Qwen3-8B|cv|JEV|725/901|True|
