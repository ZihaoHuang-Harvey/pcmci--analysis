# Causal analysis tool

多方法交互因果分析平台 - 基于 PCMCI+ 和转移熵(Transfer Entropy)的因果关系发现与可视化工具

## 项目简介

本项目是一个功能完善的时间序列因果分析平台，集成了两种主流因果发现方法：

| 方法 | 描述 | 适用场景 |
|------|------|----------|
| **PCMCI+** | 基于条件独立性检验的因果发现算法 | 多元时间序列因果结构学习 |
| **TE (转移熵)** | 基于信息论的信息流动方向分析 | 量化变量间的信息传递强度 |

平台提供交互式 GUI 界面，支持数据导入、变量角色配置、参数调优、因果图可视化和结果导出。

## 功能特性

### 核心分析功能
- **PCMCI+ 因果发现**：发现时间序列变量间的因果关系网络
- **转移熵分析**：量化信息流动方向和强度
- **变量角色配置**：支持时间驱动、地形驱动、普通变量、排除变量四种角色
- **灵活约束引擎**：支持边约束规则，融入领域先验知识

### 可视化功能
- 因果图网络可视化
- 时间序列因果演化图
- 目标变量上游影响链路图
- MCI/TE 矩阵热力图
- 显著变量对分析

### 结果导出
- MCI 矩阵导出 (Excel/CSV)
- TE/NDTE 矩阵导出
- 显著变量对清单

## 安装

### 环境要求
- Python 3.8+
- PyQt5
- numpy, pandas, matplotlib
- scikit-learn
- tigramite
- pyinform (可选，用于加速 TE 计算)

### 安装依赖

```powershell
pip install pyqt5 pandas numpy matplotlib tigramite scikit-learn openpyxl pyyaml pyinform
```

> **注意**：`pyinform` 是 TE 分析的可选加速后端。如不安装，程序会自动使用内置离散 TE 估算。

## 快速开始

### 启动 GUI

```powershell
cd "Causal analysis tool"
python run_gui.py
```

或使用模块方式运行：

```powershell
python -m app.main
```

### 分析流程

1. **数据导入**：加载 Excel 或 CSV 文件，选择要排除的列
2. **变量角色**：配置变量的因果角色（时间驱动/地形驱动/普通变量）
3. **参数设置**：设置 tau_min、tau_max、pc_alpha 等参数
4. **运行分析**：点击运行全部分析
5. **查看结果**：在结果页查看因果图和矩阵

## 项目结构

```
Causal analysis tool/
├── run_gui.py              # GUI 启动入口
├── MCI_TE.py               # 脚本模式示例
├── app/
│   ├── main.py             # PyQt5 应用主入口
│   ├── config/
│   │   ├── app_settings.yaml   # 默认配置
│   │   └── settings.py         # 配置加载器
│   ├── core/
│   │   ├── analysis_service.py      # PCMCI+ 分析服务
│   │   ├── analysis_service_te.py   # TE 分析服务
│   │   ├── constraint_engine.py     # 约束引擎
│   │   └── models.py                # 数据模型
│   ├── gui/
│   │   ├── main_window.py     # 主窗口
│   │   ├── canvas.py           # 画布组件
│   │   ├── result_plotting.py  # 结果绘图
│   │   └── analysis_worker.py  # 异步分析 worker
│   └── tools/
│       └── time_series_fitting_tool.py  # 时间序列拟合工具
├── tests/                  # 单元测试
├── docs/                   # 文档
└── outputs/               # 结果输出目录
```

## 核心概念

### PCMCI+

PCMCI+ 是一种专门针对时间序列数据的因果发现算法，适合处理：
- 多元时间序列变量间的因果关系
- 滞后因果效应（τ > 0）
- 同时刻因果关系（τ = 0）

### 变量角色

| 角色 | 说明 | 入边 | 出边 |
|------|------|------|------|
| 时间驱动 | date_sin, date_cos 等 | ❌ 禁止 | ✅ 允许 |
| 地形驱动 | elevation_mean, slope_mean 等 | ❌ 禁止 | ✅ 允许 |
| 普通变量 | 其他变量 | ✅ 允许 | ✅ 允许 |
| 排除变量 | 不参与分析 | - | - |

### 转移熵 (TE)

转移熵衡量从变量 X 到变量 Y 的信息流动：
- `TE(X→Y)`：X 对 Y 的转移熵
- `NDTE = TE(X→Y) / (TE(X→Y) + TE(Y→X))`：归一化方向性指标
  - 接近 1：信息主要从 X 流向 Y
  - 接近 0：信息主要从 Y 流向 X

## 配置说明

编辑 `app/config/app_settings.yaml` 可修改默认参数：

```yaml
defaults:
  tau_min: 0
  tau_max: 3
  pc_alpha: 0.05
  quantile_range: [25, 75]
```

## 常见问题

### Q: 为什么图中没有某条边？
可能原因：统计上不显著（未通过 pc_alpha 阈值），或被变量角色约束屏蔽。

### Q: TE 分析失败怎么办？
- 检查是否安装了 pyinform：`pip install pyinform`
- 确保数据无过多 NaN 值
- 尝试减小分箱数（默认 10）

### Q: 如何选择 PCMCI+ 和 TE？
- **PCMCI+**：适合发现因果图结构，输出明确的因果方向
- **TE**：适合量化信息流动强度，对非线性关系敏感

建议同时运行两种方法，综合对比结果。

## 相关文档

- [中文用户指南](docs/user_guide_zh.md)
- [时间序列拟合工具指南](docs/ts_fitting_tool_guide.md)

## 致谢

本项目基于以下开源库：

- [tigramite](https://github.com/jakob runge/tigramite) - 时间序列因果发现
- [pyinform](https://github.com/elife-asymp-bayes/pyinform) - 信息熵计算
- [PyQt5](https://riverbankcomputing.com/software/pyqt/) - GUI 框架
