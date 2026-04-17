# Case 3: 基于图神经网络 (GNN) 的晶体剪切模量预测

本案例旨在介绍图神经网络 (Graph Neural Networks, GNN) 在材料性质预测中的应用。相比于 Case 2 中的传统机器学习方法，GNN 能够直接从晶体结构图中学习特征，而不需要手动构建复杂的描述符。

## 1. 环境配置 (Prerequisites)

本案例依赖于 `pytorch` 环境。建议使用 `conda` 进行管理。

### 1.1 创建 Conda 环境
如果尚未配置环境，请参考以下命令创建并安装必要的依赖库：
```bash
# 创建环境
# conda create -n pytorch python=3.10
conda activate pytorch

# 安装核心依赖
#conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install pymatgen matminer scikit-learn matplotlib pandas numpy 
```

### 1.2 依赖模型 (CGCNN)
方案 1 依赖于 [CGCNN (Crystal Graph Convolutional Neural Networks)](https://github.com/txie-93/cgcnn) 开源代码库。
- **引用文献**: Xie, T., & Grossman, J. C. (2018). Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties. *Physical Review Letters*, 120(14), 145301.
- **本地路径**: 本案例假设 CGCNN 代码存储在 `../cgcnn` 目录下。

## 2. 目录结构

- `00_prepare_data.py`: 数据准备。从 `matminer` 导出 `elastic_tensor_2015` 数据集为 CIF 格式及 `id_prop.csv`。
- `01_finetune_gpu.py`: **方案 1 核心**。支持单卡 GPU 的 CGCNN 微调脚本，加入了正则化与动态学习率优化。
- `01_finetune_cpu_parallel.sh`: 方案 1 的 CPU 并行版运行脚本。
- `02_prepare_jarvis_data.py`: 为方案 2 准备 `jarvis_dft_3d` 数据集。
- `03_simple_gnn.py`: **方案 2 核心**。从零构建一个精简的晶体 GNN 模型，包含 Dropout 和早停逻辑。
- `04_plot_results.py`: 针对方案 1 的可视化脚本。
- `05_plot_simple_gnn.py`: 针对方案 2 的可视化脚本。
- `data/`: 自动生成的数据存储目录。

## 3. 方案 1：简化版 (迁移学习/微调)

利用 CGCNN 在大型数据库（如 Materials Project）上预训练的权重，在较小的 `elastic_tensor_2015` 数据集上进行微调。

### 3.1 数据转换
```bash
python 00_prepare_data.py
```

### 3.2 GPU 微调 (推荐)
```bash
# --pretrained 指定预训练权重路径
python 01_finetune_gpu.py data/elastic_tensor_2015 --pretrained ../cgcnn/pre-trained/shear-moduli.pth.tar --epochs 30 --batch-size 32
```

### 3.3 CPU 并行设置
在 `01_finetune_cpu_parallel.sh` 中，可以通过以下参数优化 CPU 性能：
- `--workers`: 数据读取进程数。
- `--cpu-threads`: 内部矩阵计算线程数。

### 3.4 可视化分析
```bash
python 04_plot_results.py
```

## 4. 方案 2：全流程版 (从头构建 GNN)

使用 `jarvis_dft_3d` 数据库，深入理解 GNN 的底层实现，包括原子嵌入、消息传递和池化。

### 4.1 数据准备
```bash
python 02_prepare_jarvis_data.py
```

### 4.2 模型训练与优化
本案例中的 GNN 实现了以下优化以防止过拟合：
- **Dropout (p=0.3)**: 随机失活神经元。
- **Weight Decay (1e-5)**: L2 正则化。
- **LR Scheduler**: 验证集误差停滞时自动减小学习率。

```bash
python 03_simple_gnn.py > simple_gnn.out
```

### 4.3 结果展示
```bash
python 05_plot_simple_gnn.py
```

## 5. 核心概念 (Core Concepts)

1. **晶体图表示 (Crystal Graph)**:
   - 节点 $v_i$: 原子，初始特征由 `atom_init.json` 中的向量表示。
   - 边 $e_{ij}$: 原子间距，通过高斯距离扩展为向量特征。
2. **消息传递 (Message Passing)**:
   - 每一个卷积层，原子 $i$ 会聚合其邻居 $j$ 的特征：$h_i^{(l+1)} = f(h_i^{(l)}, h_j^{(l)}, e_{ij})$。
3. **全局池化 (Global Pooling)**:
   - 将晶体中所有原子的局部特征求平均，得到整个晶体的结构特征。

## 6. 思考题 (Discussion)

1. **迁移学习**: 为什么使用预训练模型（方案 1）比从头训练（方案 2）在小样本上效果更好？
2. **过拟合**: 观察 `simple_gnn_performance.png`，如果验证集误差在后期上升，我们应该采取哪些措施？
3. **结构敏感性**: GNN 捕捉的是晶体的哪些物理信息（如配位数、键长等）？
