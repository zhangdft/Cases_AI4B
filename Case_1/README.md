# AI4Bat 启蒙计划：机器学习入门 —— 薪资预测实战 (Salary Prediction)

本教程旨在通过一个真实的数据集（Kaggle 薪资预测数据集），带领大家走入机器学习的大门。我们将从数据探索开始，一步步构建、优化并解释我们的第一个回归模型。

---

## 课程目标

1. **数据探索**：了解数据的分布与基本统计信息。
2. **特征分析**：识别特征之间的关联性与冗余。
3. **模型复杂度理解**：通过多项式回归直观感受欠拟合与过拟合。
4. **多模型对比**：学习如何使用交叉验证 (K-fold CV) 选择最佳模型。
5. **模型解释**：分析哪些因素对薪资影响最大。

---

## 实验环境准备

请确保已按照之前的指南配置好 `qimeng` 环境，并安装了以下包：
```bash
conda install numpy pandas scipy matplotlib seaborn scikit-learn xgboost
```

---

## 第一阶段：数据探索 (Data Exploration)
**对应脚本：`01_data_exploration.py`**

在开始建模之前，我们必须先“看一眼”数据。
- **关键点**：检查缺失值、异常值以及目标变量（Salary）的分布。
- **可视化**：直观查看 `Experience Years`（工龄）与 `Salary`（薪资）的关系。

![Salary Distribution](salary_distribution.png)
*图 1：薪资分布直方图，帮助我们理解薪资的整体范围。*

![Experience vs Salary](experience_vs_salary.png)
*图 2：工龄与薪资的散点图，观察两者之间的初步线性趋势。*

---

## 第二阶段：特征关联分析 (Feature Analysis)
**对应脚本：`02_feature_analysis.py`**

特征（Features）是模型的输入。我们需要分析：
- 哪些特征与薪资最相关？
- 特征之间是否存在冗余？（例如两个特征高度线性相关，通常只需保留其一）。
- **关键技术**：相关系数矩阵 (Correlation Matrix) 与热力图 (Heatmap)。

![Correlation Heatmap](correlation_heatmap.png)
*图 3：热力图展示了各数值特征之间的相关性。*

![Education vs Salary](education_vs_salary.png)
*图 4：不同学历水平下的薪资分布，观察类别特征对目标的影响。*

---

## 第三阶段：欠拟合与过拟合 (Underfitting vs Overfitting)
**对应脚本：`03_underfitting_overfitting.py`**

这是机器学习中最核心的概念之一。在这一阶段，我们不再只看单一特征，而是利用**所有特征**进行建模。
- **数据处理**：
  - **缺失值补齐**：使用 `SimpleImputer` 自动填充缺失数据。
  - **特征编码**：
    - **序数编码 (Ordinal Encoding)**：针对有等级关系的特征（如 `Education Level`, `Company Size`）。
    - **独热编码 (One-Hot Encoding)**：针对无等级关系的分类特征（如 `Job Title`, `Industry`）。
- **欠拟合 (Underfitting)**：模型太简单（如 Degree 1），无法充分捕捉所有特征与薪资的复杂关系。
- **过拟合 (Overfitting)**：模型太复杂（如 Degree 3），虽然在训练集上误差极低，但在测试集上误差激增。
- **关键技术**：多项式特征 (Polynomial Features) 扩展全特征空间。

![Error Curve All Features](error_curve_all_features.png)
*图 5：全特征下的误差随复杂度变化的曲线，展示了偏差与方差的权衡。*

---

## 第四阶段：模型对比与选择 (Model Comparison)
**对应脚本：`04_model_comparison.py`**

现实中没有一个模型是通用的。我们需要对比多种算法：
1. **线性回归 (Linear Regression)**：基础回归模型。
2. **Lasso 回归**：带正则化的线性模型，可自动选择特征。
3. **决策树 (Decision Tree)**：非线性模型，易于理解。
4. **随机森林 (Random Forest)**：集成模型，稳定性好。
5. **XGBoost**：目前竞赛中最流行的梯度提升树模型。

- **关键技术**：K 折交叉验证 (K-fold Cross-Validation)，确保评估结果的稳定性。

![Model Comparison R2](model_comparison_r2_all_features.png)
*图 6：各模型在全特征交叉验证下的 R² 得分对比。*

![Actual vs Predicted](actual_vs_predicted_all_features.png)
*图 7：最佳模型（XGBoost）的预测值与真实值对比图。*

---

## 第五阶段：模型结果分析与解释 (Model Interpretation)
**对应脚本：`05_model_interpretation.py`**

模型不仅要预测准，还要告诉我们“为什么”。
- **关键技术**：特征重要性 (Feature Importance)。
- **分析点**：在我们的预测中，是 `Experience Years` 更重要，还是某个特定的 `Job Title` 影响更大？

![Feature Importance](feature_importance_all_features.png)
*图 8：排名前 20 的详细特征（含独热编码后）重要性分布。*

![Aggregated Feature Importance](aggregated_feature_importance.png)
*图 9：聚合后的原始特征重要性分布，直观展示核心影响因素。*

---

## 总结与思考

通过本次实验，我们完成了一个完整的机器学习闭环：
**数据导入 -> 特征工程 -> 模型训练 -> 调优验证 -> 结果解释**

**课后思考：**
1. 如果我们把 `location`（地点）这个特征去掉，模型的准确率会下降多少？
2. 为什么 Degree 3 的模型在训练集上表现接近完美，但在测试集上表现却变差了？
3. 对于材料学背景的同学，如果我们要预测“某种合金的硬度”，这些步骤是否同样适用？

---
祝你在 AI4Bat 的学习中收获满满！
