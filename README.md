# RankMixer recommendation project

`rankmixer_fair_ablation_v2_score.ipynb` 的工程化实现。代码按推荐系统项目常见分层组织，旧 notebook service 名称作为 CLI alias 保留。

## Project layout

```text
rec/
├── config.py              # 路径、数据、模型、loss、训练配置
├── features.py            # 特征定义、rank/score 特征、article SVD、token schema
├── data.py                # 数据读取、历史序列、采样 batch、DataLoader 输入
├── models/rankmixer.py    # RankMixerRanker 和模型组件
├── losses.py              # BalancedGE、OldGE、JRC-BPR、Listwise-BPR-BCE 等 loss class
├── evaluate.py            # AUC/NDCG/MRR/HitRate 和预测表
├── train.py               # RankMixerTrainer
├── pipeline.py            # 场景编排与 CLI
└── services/              # RankMixer CLI service 入口
```

## Data

原始数据放在：

```text
data/raw/news_recommendation/
```

至少需要：

- `articles.csv`
- `articles_emb.csv`
- `train_click_log.csv`
- `testA_click_log.csv`

RankMixer 训练读取 `data/processed/temp_results/` 下的候选特征表，例如：

- `trn_user_item_feats_df_all_rankmixer_v2_top64.csv`
- `val_user_item_feats_df_all.csv`
- `tst_user_item_feats_df_all.csv`
- `click_hist_all.csv`

## Usage

查看可用入口：

```bash
python -m rec.app.main --list
```

预览工程结构和实验场景：

```bash
python -m rec.app.main rankmixer_score --preview
```

训练一个小样本 smoke：

```bash
python -m rec.app.main rankmixer_score --train --scenario jrc_bpr --limit-rows 1024 --epochs 1 --batch-size 128 --no-article-svd
```

正式训练：

```bash
python -m rec.app.main rankmixer_score --train --scenario jrc_bpr --train-variant top64 --epochs 5 --batch-size 128
```

### wandb

训练可通过 `--wandb` 开启 wandb 记录，常用参数包括：

- wandb 配置：`--wandb-project`、`--wandb-entity`、`--wandb-run-name`、`--wandb-mode`、`--wandb-group`、`--wandb-tags`、`--wandb-log-every-n-steps`、`--wandb-watch`、`--wandb-watch-log`、`--wandb-watch-log-freq`、`--wandb-log-model`、`--no-wandb-prediction-artifacts`
- 训练超参：`--weight-decay`、`--grad-clip-norm`、`--topk`

wandb 会记录 git commit/dirty status、模型结构 artifact、超参数、batch size、lr、`scheduler=none`、loss、step 级 train loss/lr/grad_norm/update_norm、epoch 级 val loss/MRR/NDCG、预测表 artifact、运行环境/设备/依赖版本。

示例：

```bash
python -m rec.app.main rankmixer_score --train --scenario jrc_bpr --train-variant top64 --epochs 5 --batch-size 128 --wandb --wandb-project funrec-rankmixer --wandb-run-name jrc-bpr-top64 --weight-decay 0.01 --grad-clip-norm 1.0 --topk 20
```

输出写入 `outputs/`。
