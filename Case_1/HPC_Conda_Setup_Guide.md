# AI4Bat 启蒙计划：开发环境搭建与配置指南

本指南旨在协助大家在学校 HPC 平台上构建 AI 开发环境，配置 Conda 环境并安装必要的 Python 包。

---

## 第一部分：HPC 平台登录与实例创建

### 1. 登录 HPC 平台
首先，使用学校提供的账号登录 HPC 平台门户界面。
![登录 HPC](resources/conda_env/1-enterHPC.jpg)

### 2. 创建新环境/实例
在开发中心中选择“新建开发环境”。
![新建环境](resources/conda_env/2-New_env.jpg)

### 3. 选择环境类型
根据课程需求选择合适的运行环境。
![选择类型](resources/conda_env/3-Type.jpg)

### 4. 配置资源
选择所需的计算资源，如 14核和32G内存。
![配置资源](resources/conda_env/4-resources.jpg)

### 5. 挂载存储
确保挂载了必要的数据存储卷，采用默认挂载即可。
![挂载存储](resources/conda_env/5-mount.jpg)

### 6. 构建并启动实例
确认配置无误后，点击“创建”。
![构建实例](resources/conda_env/6-builde.jpg)

---

## 第二部分：进入开发桌面与终端准备

### 7. 加载并打开环境
等待实例构建完成，状态显示为“运行中”后，点击进入“桌面”模式。
![加载环境](resources/conda_env/7-load_env.jpg)
![打开桌面](resources/conda_env/7-open_env.jpg)

### 8. Ubuntu 桌面环境
进入后你将看到熟悉的 Ubuntu 图形化桌面。
![Ubuntu 桌面](resources/conda_env/8-Ubuntu-desktop.jpg)

### 9. 打开终端
右键找到并打开“终端 (Terminal)”。
![打开终端](resources/conda_env/9-open_terminal.jpg)
![进入终端](resources/conda_env/10-new_terminal.jpg)

---

## 第三部分：Conda 环境配置

### 10. 检查 Conda 状态
在终端中输入以下命令查看当前的 Conda 环境列表：
```bash
conda envs list 
```
![检查环境](resources/conda_env/11_check_conda_env.jpg)

### 11. 初始化 Conda
如果 Conda 尚未在当前 shell 中初始化，请执行：
```bash
conda init bash
```
然后重启终端或执行 `source ~/.bashrc` 使其生效。
![Conda 初始化](resources/conda_env/12_conda_init.jpg)

### 12. 配置 Conda 镜像源
为了加快下载速度，建议添加国内镜像源（如清华源）：
```bash
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
```
![配置镜像源 1](resources/conda_env/13_conda_channels-1.jpg)
![配置镜像源 2](resources/conda_env/13_conda_channels-2.jpg)

### 13. 创建 Conda 环境
创建一个名为 `qimeng` 的环境，并指定 Python 版本为 3.10：
```bash
conda create -n qimeng python=3.10 -y
```
![创建环境](resources/conda_env/13_create_env.jpg)

### 14. 激活环境
```bash
conda activate qimeng
```
![激活环境](resources/conda_env/13_activate_env.jpg)

---

## 第四部分：安装包与 PyCharm 配置

### 15. 安装 Python 包
在激活的 `qimeng` 环境中安装必要的科学计算和深度学习包（如 torch, numpy, pandas 等）：
```bash
conda install numpy pandas scipy matplotlib seaborn scikit-learn xgboost
```
![安装包 1](resources/conda_env/14_install_packages.jpg)
![安装包 2](resources/conda_env/14_install_packages-2.jpg)

### 16. 启动 PyCharm
在应用程序菜单中打开 PyCharm。
![打开 PyCharm](resources/conda_env/15_open_pycharm.jpg)

### 17. 新建项目并配置解释器
新建项目时，在 Interpreter 设置中选择 `Conda Environment`，并指向你刚才创建的 `qimeng` 环境。
![新建项目](resources/conda_env/16_new_project.jpg)

---

祝你在 AI4Bat 启蒙计划中学习顺利！
