# Hydrological Models / 水文模型教学库

[English](#english) | [中文](#中文)

-----

<a name="english"></a>

## English

### Overview for Students

This repository is designed as a teaching tool for 3rd-year hydraulic engineering students. It provides Python implementations of classic hydrological models, focusing on bridging the gap between theory and practical application. Here, you will not only learn the mathematical principles behind these models but also understand how they are calibrated and validated using real-world data.

### Core Concepts Covered

* **Rainfall-Runoff Process**: Understand how different models conceptualize the transformation of precipitation into streamflow.
* **Model Structure**: Compare lumped vs. semi-distributed concepts and saturation excess vs. infiltration excess mechanisms.
* **Parameter Sensitivity**: Learn how model parameters influence simulation results.
* **Model Calibration & Validation**: Master the essential workflow of hydrological modeling: using historical data to optimize parameters and validating the model on an independent dataset.
* **Performance Metrics**: Learn to evaluate model performance using standard metrics like Nash-Sutcliffe Efficiency (NSE).

### Implemented Models

#### 1. **Xinanjiang Model (新安江模型)**

- **Type**: Conceptual, saturation excess mechanism.
- **Best for**: Humid and semi-humid regions.

#### 2. **Tank Model (タンクモデル)**

- **Type**: Conceptual, multiple reservoirs.
- **Best for**: Versatile for various catchment types.

#### 3. **GR4J Model**

- **Type**: Lumped, conceptual, and parsimonious (only 4 parameters).
- **Best for**: Daily streamflow simulation.

#### 4. **Sacramento Model (SAC-SMA)**

- **Type**: Continuous soil moisture accounting.
- **Best for**: Detailed operational river forecasting.

### Installation

```bash
# Clone the repository
git clone https://github.com/licm13/Hydrological-model.git
cd Hydrological-model

# Install required packages
pip install -r requirements.txt
```

### Quick Start: Understanding the Basics

First, run the examples with synthetic data to understand model behavior.

```bash
# This script compares all models under various scenarios
python examples.py
```

This will demonstrate model comparison, sensitivity analysis, and storm event simulation.

### Advanced Example: Calibration and Validation

This is the core of practical hydrological modeling. We will use a sample dataset to calibrate the GR4J model.

1. **Explore the data**: Open `data/sample_data.csv` to see the structure: Date, Precipitation, Evapotranspiration, Observed_Flow.
2. **Run the calibration**:
   ```bash
   python calibration_example.py
   ```

This script will:

* Load real-world data.
* Split the data into calibration and validation periods.
* Use an optimization algorithm to find the best parameters for the GR4J model by maximizing the NSE.
* Generate a plot comparing simulated and observed streamflow.

### Project Structure

```
Hydrological-model/
├── data/
│   └── sample_data.csv       # Sample real-world data
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── xinanjiang_model.py       # Xinanjiang model
├── tank_model.py             # Tank model
├── gr4j_model.py             # GR4J model
├── sacramento_model.py       # Sacramento model
├── examples.py               # Examples with synthetic data
└── calibration_example.py    # Calibration with real data
```

### For Your Assignment

1. **Try calibrating other models**: Modify `calibration_example.py` to calibrate the Xinanjiang or Tank model. Compare which model performs better on the sample dataset.
2. **Adjust the calibration period**: Change the split date between calibration and validation. How does this affect the results?
3. **Experiment with parameters**: Manually change the parameters of a model and observe how the simulated hydrograph changes.

-----

<a name="中文"></a>

## 中文

### 教学概述

本代码库专为水利工程大三学生设计，是一个连接水文模型理论与实践的教学工具。在这里，您不仅能学习经典模型的数学原理，还将掌握如何使用真实数据对模型进行**参数率定**与**验证**，这是水文模型应用的核心技能。

### 覆盖的核心概念

* **降雨径流过程**: 理解不同模型如何将降雨转化为径流。
* **模型结构**: 比较集总式与半分布式、蓄满产流与超渗产流等不同机制。
* **参数敏感性**: 学习模型参数如何影响模拟结果。
* **参数率定与验证**: 掌握水文建模的完整工作流——使用历史数据优化参数，并在独立的数据集上检验模型效果。
* **模型评价指标**: 学习使用纳什效率系数 (NSE) 等国际通用指标来评价模型表现。

### 已实现的模型

#### 1. **新安江模型**

- **类型**: 概念性，蓄满产流机制。
- **适用于**: 湿润和半湿润地区。

#### 2. **Tank模型**

- **类型**: 概念性，多水库结构。
- **适用于**: 各种不同产流特性的流域。

#### 3. **GR4J模型**

- **类型**: 集总式概念性模型，仅4个参数，非常简约。
- **适用于**: 日径流模拟。

#### 4. **Sacramento模型 (SAC-SMA)**

- **类型**: 连续土壤水分核算模型。
- **适用于**: 精细化的业务化洪水预报。

### 安装

```bash
# 克隆仓库
git clone https://github.com/licm13/Hydrological-model.git
cd Hydrological-model

# 安装依赖包
pip install -r requirements.txt
```

### 快速入门：理解模型基础

首先，运行基于虚拟数据的示例，以理解各个模型的基本行为。

```bash
# 该脚本在多种情境下对比所有模型
python examples.py
```

这将为您展示模型对比、参数敏感性分析和暴雨洪水模拟等功能。

### 进阶案例：参数率定与验证

这是应用水文模型最核心的环节。我们将使用一个案例数据集来率定GR4J模型。

1. **探索数据**: 打开`data/sample_data.csv`文件，查看数据结构：日期、降雨量、蒸发量、实测流量。
2. **运行率定脚本**:
   ```bash
   python calibration_example.py
   ```

该脚本将自动完成以下任务：

* 加载真实流域数据。
* 将数据分为"率定期"和"验证期"。
* 通过优化算法，以NSE最大化为目标，寻找GR4J模型的最优参数组合。
* 生成一张对比模拟流量与实测流量过程线的图片。

### 项目结构

```
Hydrological-model/
├── data/
│   └── sample_data.csv       # 真实数据样例
├── README.md                 # 本文件
├── requirements.txt          # Python依赖
├── xinanjiang_model.py       # 新安江模型
├── tank_model.py             # Tank模型
├── gr4j_model.py             # GR4J模型
├── sacramento_model.py       # Sacramento模型
├── examples.py               # 基于虚拟数据的示例
└── calibration_example.py    # 基于真实数据的率定与验证
```

### 课后思考与练习

1. **率定其他模型**: 修改`calibration_example.py`，尝试率定新安江模型或Tank模型，并比较哪个模型在该数据集上表现更好。
2. **调整率定周期**: 更改率定期与验证期的分割点，观察结果有何变化？
3. **手动调整参数**: 在不使用优化算法的情况下，手动修改某个模型的参数，观察流量过程线如何响应。

