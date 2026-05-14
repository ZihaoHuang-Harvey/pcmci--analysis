# 时序拟合工具使用说明

## 1. 工具简介

时序拟合工具用于基于不连续的时间序列数据，训练回归模型补全缺失时间点的数值。主要功能包括：

- 支持多种拟合方法：Spline、CubicSpline、RandomForest、GradientBoosting、XGBoost、CatBoost、PLS、SVR
- 自动处理多个流域的数据
- 时序数据内插与补全
- 结果可视化与导出

## 2. 启动方式

在 PCMCI 主界面中：

- 点击菜单 **工具 → 时序拟合工具**
- 或独立运行：`python app/tools/time_series_fitting_tool.py`

## 3. 界面功能

### 3.1 控制面板

| 功能 | 说明 |
|-----|------|
| 加载数据 | 选择 Excel 或 CSV 文件（需包含 watershed、date 及各指标列） |
| 清除数据 | 清除当前加载的数据 |
| 拟合方法 | 选择模型：Spline、CubicSpline、RandomForest、GradientBoosting、PLS、SVR、XGBoost、CatBoost |
| 时间粒度 | 选择输出粒度：daily、quarterly、yearly、monthly、monthly_4 |
| 平滑程度 | 滑动条调整平滑因子（0-100%） |
| 参数设置 | 打开各模型的参数配置对话框 |
| 全时序拟合 | 开始执行时序拟合 |
| 停止 | 停止当前运行的任务 |
| 导出结果 | 将拟合结果导出为 Excel 文件 |

### 3.2 数据要求

输入数据需满足以下格式：

- 必须包含 `watershed` 列：流域名称
- 必须包含 `date` 列：日期（格式为 YYYYMMDD）
- 数值列：需要拟合的景观指标（如 NDVI_mean、CONTAG、SIDI 等）

数据会自动进行以下预处理：

- 提取年、月、日信息
- 计算季节特征（sin/cos）
- 数据范围过滤（如 NDVI 在 0-1 之间）

### 3.3 结果查看

工具包含多个结果标签页：

- **拟合结果**：查看每个流域的拟合效果
- **原始数据**：浏览原始输入数据
- **DOY 拟合**：按年积日(Day of Year)进行拟合

## 4. 拟合方法说明

| 方法 | 适用场景 | 特点 |
|-----|---------|------|
| Spline | 数据点较少 | 样条插值，平滑通过 |
| CubicSpline | 数据点较少 | 三次样条，更平滑 |
| RandomForest | 数据量较大 | 随机森林，非线性关系 |
| GradientBoosting | 数据量较大 | 梯度提升，精度较高 |
| XGBoost | 数据量较大 | XGBoost（需安装） |
| CatBoost | 数据量较大 | CatBoost（需安装） |
| PLS | 多变量 | 偏最小二乘降维 |
| SVR | 数据量中等 | 支持向量回归 |

## 5. 参数设置

点击"参数设置"按钮可以配置各模型的超参数：

- **RandomForest**: 树数量、最大深度、最小分割样本数等
- **GradientBoosting**: 迭代次数、学习率、深度等
- **XGBoost**: 迭代次数、学习率、子采样比例等
- **CatBoost**: 迭代次数、深度、学习率等
- **SVR**: C正则化、epsilon、gamma核系数等
- **PLS**: 主成分数量

## 6. 导出结果

拟合完成后可以导出：

- 完整的拟合结果数据（包含原始值和预测值）
- 按流域、年份分组的子表
- 模型评估指标（R²、MSE）

## 7. 常见问题

### 7.1 数据点不足

如果某个流域的数据点少于 5 个，系统会自动跳过该流域。

### 7.2 XGBoost/CatBoost 未安装

如未安装 XGBoost 或 CatBoost，对应选项将不可用。可以使用 pip 安装：

```powershell
pip install xgboost catboost
```

### 7.3 拟合效果不佳

可以尝试：

- 调整平滑程度参数
- 更换拟合方法
- 检查数据质量，清除异常值
