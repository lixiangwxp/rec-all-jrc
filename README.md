# chapter_5_projects_all

一个可独立抽出的新闻推荐实验项目，包含：

- `4.recall_all.ipynb`：多路召回
- `5.feature_engineering_all.ipynb`：特征工程
- `jrc-ranking_all.ipynb`：PyTorch 版 JRC 排序

## 目录结构

```text
chapter_5_projects_all/
├── 4.recall_all.ipynb
├── 5.feature_engineering_all.ipynb
├── jrc-ranking_all.ipynb
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   │   └── news_recommendation/
│   └── processed/
│       └── temp_results/
└── outputs/
```

## 原始数据放置位置

把原始数据文件放到：

`data/raw/news_recommendation/`

至少需要这些文件：

- `articles.csv`
- `articles_emb.csv`
- `train_click_log.csv`
- `testA_click_log.csv`

## 安装依赖

在项目根目录执行：

```bash
pip install -r requirements.txt
```

## 运行顺序

请先进入项目根目录，再打开 notebook：

```bash
cd chapter_5_projects_all
```

按顺序运行：

1. `4.recall_all.ipynb`
2. `5.feature_engineering_all.ipynb`
3. `jrc-ranking_all.ipynb`

## 文件输出位置

- 中间结果：`data/processed/temp_results/`
- 导出结果与提交文件：`outputs/`

## 说明

- 本项目默认使用纯相对路径，不依赖 `.env` 或 `FUNREC_*` 环境变量。
- 原始数据和中间结果默认不提交到 Git。
