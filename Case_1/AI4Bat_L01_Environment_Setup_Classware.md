---
marp: true
theme: default
paginate: true
header: "AI4Bat 启蒙计划 - 第一课"
footer: "材料与能源学院 - AI4Bat 课程组"
---

# AI4Bat 启蒙计划：第一课
## 开发环境搭建与配置指南

---

# 课程目标

1. **登录学校 HPC 平台**：熟悉高性能计算环境的使用。
2. **实例创建与资源配置**：学会根据 AI 开发需求申请 GPU/CPU 资源。
3. **Conda 环境配置**：掌握 Python 虚拟环境的管理与镜像源配置。
4. **PyCharm 配置**：将本地 IDE 与远程 Conda 环境关联。

---

# 第一部分：HPC 平台登录与实例创建

---

# 1. 登录 HPC 平台
- 使用学校统一认证账号登录。
- 进入 HPC 控制台。

![HPC Login](resources/conda_env/1-enterHPC.jpg)

---

# 2. 创建新环境/实例
- 在开发中心选择“新建开发环境”。
- 这是你进行 AI 实验的“云端工作台”。

![New Env](resources/conda_env/2-New_env.jpg)

---

# 3. 配置环境与资源
- **类型**：根据课程需求选择合适的运行环境。
- **资源**：选择所需的计算资源，如 14 核和 32G 内存。

![Type](resources/conda_env/3-Type.jpg)
![Resources](resources/conda_env/4-resources.jpg)

---

# 4. 挂载存储与启动
- **挂载存储**：采用默认挂载即可。
- **创建实例**：点击“创建”，等待系统就绪。

![Mount](resources/conda_env/5-mount.jpg)
![Build](resources/conda_env/6-builde.jpg)

---

# 第二部分：进入开发桌面与终端准备

---

# 5. 打开远程桌面
- 实例状态变为“运行中”后，点击进入“桌面”模式。

![Open Env](resources/conda_env/7-open_env.jpg)
![Load Env](resources/conda_env/7-load_env.jpg)

---

# 6. Ubuntu 桌面与终端
- 右键找到并打开 **Terminal** 图标。
- 这是我们后续所有配置的核心入口。

![Ubuntu Desktop](resources/conda_env/8-Ubuntu-desktop.jpg)
![Open Terminal](resources/conda_env/9-open_terminal.jpg)

---

# 第三部分：Conda 环境配置 (核心步骤)

---

# 7. 检查与初始化 Conda
- 查看现有环境：`conda envs list`
- 初始化 Shell：`conda init bash`

![Check Conda](resources/conda_env/11_check_conda_env.jpg)
![Conda Init](resources/conda_env/12_conda_init.jpg)

---

# 8. 配置镜像源加速 (清华源)
- 解决包下载慢的问题。
```bash
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
```

![Channels 1](resources/conda_env/13_conda_channels-1.jpg)
![Channels 2](resources/conda_env/13_conda_channels-2.jpg)

---

# 9. 创建并激活 qimeng 环境
- 环境名称统一采用：`qimeng`
- 指定 Python 版本：`3.10`

```bash
conda create -n qimeng python=3.10 -y
conda activate qimeng
```

![Create Env](resources/conda_env/13_create_env.jpg)
![Activate Env](resources/conda_env/13_activate_env.jpg)

---

# 第四部分：安装包与 PyCharm 配置

---

# 10. 安装 Python 包
- 在激活的 `qimeng` 环境中执行：
```bash
conda install numpy pandas scipy matplotlib seaborn scikit-learn xgboost
```

![Install Packages](resources/conda_env/14_install_packages.jpg)

---

# 11. PyCharm 配置
- 打开 PyCharm，在项目设置中选择 `Conda Environment`。
- 指向 `/home/user/anaconda3/envs/qimeng/bin/python`。

![Open PyCharm](resources/conda_env/15_open_pycharm.jpg)
![New Project](resources/conda_env/16_new_project.jpg)

---

# 恭喜你！
## 你的 AI4Bat 开发环境已准备就绪。

- **详细指南**：查看 `HPC_Conda_Setup_Guide.md`
- **下一步**：尝试运行第一个 Python 脚本。
