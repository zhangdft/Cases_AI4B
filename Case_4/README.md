# Case 4：Fingerprint + VAE（面向学生）

本案例用一个最小可运行的 **VAE（变分自编码器）** 来学习电解液溶剂分子的 **连续隐空间**。你将用它完成三件事：

- 把分子表示成 Morgan/ECFP 指纹（fingerprint）
- 训练一个 VAE，让模型“压缩并重构”指纹
- 用隐空间做可视化、最近邻搜索、插值探索

## 1. 你将学到什么

- 什么是 `encoder / decoder / latent z / reconstruction`
- `AE` 与 `VAE` 的区别（VAE 多了 KL 约束，让隐空间更连续可采样）
- 为什么 `PCA`、`t-SNE` 适合做可视化，但不是“生成式隐空间”
- 如何用隐空间做：相似分子搜索、插值探索

## 2. 文件说明

- 数据：`dataset/electrolyte_solvent_library_s.json`（包含 `name / smiles / family` 等字段）
- 训练脚本：`01_fingerprint_vae.py`（Fingerprint + VAE）
- 分析脚本：`02_basic_latent_analysis.py`（PCA/t-SNE/VAE 对比、最近邻、插值，并画分子结构图）

## 3. Python 环境

本课程使用已有 conda 环境 `qimeng`：

```bash
conda activate qimeng
```

或直接用指定解释器运行：

```bash
/data/home/5240019/.conda/envs/qimeng/bin/python -V
```

环境中需要这些包（你不需要手动写代码安装，若缺失再联系老师/助教）：\
`pytorch`、`rdkit`、`numpy`、`pandas`、`scikit-learn`、`matplotlib`

## 4. 运行流程（建议按顺序做）

所有命令都在 `Case_4` 目录下执行。

### 4.1 训练 Fingerprint-VAE

最简运行（自动选择 cpu/cuda）：

```bash
/data/home/5240019/.conda/envs/qimeng/bin/python 01_fingerprint_vae.py \
  --data dataset/electrolyte_solvent_library_s.json \
  --out-dir runs/basic_fingerprint_vae
```

强制使用 GPU 或 CPU：

```bash
# GPU
python 01_fingerprint_vae.py   --data dataset/electrolyte_solvent_library_s.json   --out-dir runs/basic_fingerprint_vae   --epochs 400   --latent-dim
 8   --hidden-dims 1024,512   --dropout 0.0   --beta 1.0 --device cuda ...
# CPU
/data/home/5240019/.conda/envs/qimeng/bin/python 01_fingerprint_vae.py --device cpu ...
```

训练输出（在 `--out-dir` 目录）：

- `basic_vae_best.pt`：验证集最优模型
- `basic_vae_metrics.json`：关键指标（含 Tanimoto）
- `basic_vae_training.png`：训练曲线（total / recon / kld）
- `basic_vae_latent.csv`：每个分子的 latent（mu）坐标
- `basic_vae_latent_2d.png`：当 `--latent-dim 2` 时按 `family` 着色的散点图

你可以先重点看：

- `basic_vae_metrics.json` 里的 `test_mean_recon_tanimoto`（越大越好）
- `basic_vae_training.png` 中 recon 与 kld 是否都在下降/趋稳

### 4.2 画图与探索（PCA / t-SNE / 最近邻 / 插值）

```bash
python 02_basic_latent_analysis.py \
  --data dataset/electrolyte_solvent_library_s.json \
  --ckpt runs/basic_fingerprint_vae/basic_vae_best.pt \
  --out-dir runs/basic_fingerprint_vae/analysis
```

该脚本会生成：

- `compare_pca_tsne_vae.png`：PCA / t-SNE / VAE latent 三图对比（按 `family` 着色）
- `nearest_neighbors_latent.csv`：VAE latent 空间最近邻（默认查询 `EC,DME`）
- `nearest_neighbors_EC.png`、`nearest_neighbors_DME.png`：查询分子 + Top-K 最近邻的结构网格图
- `interp_DMC_DEC.csv` + `interp_DMC_DEC.png`：插值路径上 Tanimoto 变化
- `interp_DMC_DEC_mols.png`：插值路径上“最近邻真实分子”的结构演变网格图

常用自定义参数：

```bash
# 自定义最近邻查询对象
--queries EC,DME,THF,PC

# 自定义插值端点（A:B 或 A->B）
--pairs DMC:DEC,EC:PC,THF:DME

# 插值点数（越大越平滑）
--n-steps 21
```

## 5. 你需要写在报告/作业里的内容（建议）

- 贴出 `compare_pca_tsne_vae.png`，并回答：三种方法的差别是什么？哪个更适合“生成式隐空间”？为什么？
- 贴出 `nearest_neighbors_EC.png` 或 `nearest_neighbors_DME.png`，并解释：最近邻是否符合你对化学相似性的直觉？
- 贴出任意一组插值图（`interp_*_mols.png` + `interp_* .png`），描述插值路径上“分子家族/结构”如何变化

## 6. 常见问题

- `JSONDecodeError: Unexpected UTF-8 BOM`：已在脚本中用 `utf-8-sig` 兼容，不需要你处理。
- `RDKit DEPRECATION WARNING: please use MorganGenerator`：脚本已使用新接口 MorganGenerator。
- 训练较慢：优先尝试 `--device cuda`（如果机器支持 GPU），或把 `--epochs` 调小做快速验证。
- dataset下有更大的数据集，尝试构建更先进的GNN-VAE架构

