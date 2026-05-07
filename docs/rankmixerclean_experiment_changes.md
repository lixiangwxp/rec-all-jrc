# rankmixerclean 实验运行变更说明

更新时间：2026-05-04

## 这次要解决什么

目标是在 `rankmixerclean.ipynb` 里跑完三组实验，并且让结果直接写回原 notebook 的三个结果单元格：

1. `old_ge + two_logit + diff`，作为 baseline
2. `query_softmax_ce + single_logit + scalar`
3. `listwise_bpr_bce + single_logit + scalar`

## 修改了什么

### `rankmixerclean.ipynb`

- 默认主线从 `jrc_bpr` 改为 `old_ge`。
- `DEFAULT_SCENARIOS` 只保留计划要跑的三组：
  - `old_ge`
  - `query_softmax_ce`
  - `listwise_bpr_bce`
- 删除计划外分支：
  - `balanced_ge`
  - `jrc_bpr`
  - `BalancedGELoss`
  - `JRCBPRLoss`
  - `jrc_bpr_weight`
- 将最后训练区改成三个独立结果单元格：
  - cell 41: `old_ge_result = run_experiment("old_ge")`
  - cell 43: `query_softmax_ce_result = run_experiment("query_softmax_ce")`
  - cell 45: `listwise_bpr_bce_result = run_experiment("listwise_bpr_bce")`

每个结果单元格设计为显示该实验的 summary、epoch history 和 artifact 路径。

为了让完整实验能稳定跑完，训练配置也做了几处收紧：

- train 仍使用 full `top64`，不抽样训练集。
- val 在准备 `prepared_data` 之前按 `user_id` 固定抽 2,000 个 query，三组实验共享同一份验证样本。
- batch size 从 128 调到 1024，减少每个 epoch 的 DataLoader 步数。
- 训练阶段不再计算各 loss 的 `val_loss`，只用同一份 sampled val 的排序指标选 best epoch。
- 默认只保存 `history.csv`、`summary.csv`、`best_state.pt`，不再保存大体量 prediction CSV。
- 每 1000 个 batch 打一条训练进度日志，方便后台确认进程仍在推进。

同时将 `QueryBatchSampler` 从 `defaultdict(list)` 改成 numpy 排序边界切片：只保存 query 排序后的 index 和每个 query 的 start/end 边界，避免为 700 万训练行创建大量 Python `list[int]` 对象。

### `scripts/run_rankmixerclean_experiments.py`

新增 lean runner，用于绕过 notebook 里的探索分析 cell，只执行实验必要路径：

- 配置、scenario、特征定义
- 数据读取与模型输入准备
- RankMixer 模型定义
- loss、评估、训练循环定义
- 三组实验

runner 每跑完一组实验，就会把该组输出写回 `rankmixerclean.ipynb` 对应结果单元格。

runner 后续被收紧为更简单的主流程：

- 强制 `os.chdir(PROJECT_ROOT)`，避免 `Path.cwd()` 指到错误目录。
- 训练输出同时写到日志和 notebook cell 缓存，便于后台监控 epoch 进度。
- 单个实验失败时，只把该实验的错误写回对应结果 cell，并继续尝试后续实验。

没有保留复杂的 cell 结构校验和额外防御逻辑，避免脚本变得难读。

## 遇到的问题

### 1. `nbconvert --execute --inplace` 没有及时产出结果

原因不是三组实验代码本身报错，而是 `nbconvert` 从头执行整本 notebook，会先跑到中间的特征探索、相关性分析、绘图等 cell。那些 cell 对最终三组实验不是必需的，但会拖慢甚至卡住执行，导致一直没进入结果单元格。

处理方式：停止整本 notebook 的 in-place 执行，改用 lean runner 跳过探索分析 cell。

### 2. 第一次 lean runner 在 `@dataclass` 处报错

报错原因：runner 用自定义 namespace 执行 notebook cell，但没有把该 namespace 注册进 `sys.modules`。Python 3.10 的 `dataclasses` 会检查类所在模块，因此报：

```text
AttributeError: 'NoneType' object has no attribute '__dict__'
```

处理方式：在 runner 中创建 `types.ModuleType("__rankmixerclean_runner__")`，并注册到 `sys.modules`，再用该 module 的 `__dict__` 作为 notebook 执行 namespace。

### 3. 日志停在 `model params` 后很久没有 epoch

这里没有明确的异常栈；更像是进入训练前的 DataLoader/sampler 构造和第一轮长训练阶段消耗太大。后来也发现有重复 runner 进程残留，容易把状态判断搞乱。

处理方式：

- 停掉残留 runner，只保留一个受控后台进程。
- 优化 `QueryBatchSampler` 的内存结构。
- 把验证集改成固定 query sample。
- 增加 batch 级进度日志。

## 当前运行方式

当前使用 `funrecjrc` 环境运行：

```bash
/opt/anaconda3/envs/funrecjrc/bin/python scripts/run_rankmixerclean_experiments.py
```

后台日志写在：

```text
outputs/notebook_runs/
```

最新运行信息写在：

```text
outputs/notebook_runs/latest_rankmixerclean_run.txt
```

## 跑完后怎么看

跑完后直接重新打开或 reload：

```text
rankmixerclean.ipynb
```

看最后三个结果单元格即可：

- 实验 1：baseline
- 实验 2：single-logit 对照 1
- 实验 3：single-logit 对照 2

同时，CSV 结果会保存到：

```text
outputs/old_ge_*.csv
outputs/query_softmax_ce_*.csv
outputs/listwise_bpr_bce_*.csv
```
