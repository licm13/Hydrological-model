# Hydrological Models / 水文模型教学库

[English](#english) | [中文](#中文)

-----

<a name="english"></a>

## English

### 🎓 NEW: Teaching Materials Available! / 新增教学材料！

We have added comprehensive teaching materials to help students and instructors:
我们添加了综合教学材料来帮助学生和教师：

- 📊 **[PowerPoint Presentation](docs/HydroLearn_Teaching_Presentation.pptx)** - 18 slides covering all models / 18页PPT涵盖所有模型
- 📖 **[Teaching Guide](docs/TEACHING_GUIDE.md)** - Complete semester curriculum with exercises / 完整学期课程含练习
- 🎨 **[Model Flowcharts](figures/)** - Visual diagrams for HBV, Xinanjiang, and more / 模型流程图
  - `hbv_model_flowchart.png` - HBV model structure / HBV模型结构
  - `xinanjiang_model_flowchart.png` - Xinanjiang model structure / 新安江模型结构
  - `water_cycle_diagram.png` - Hydrological cycle concept / 水文循环概念
  - `model_comparison_table.png` - Model comparison / 模型对比表

**Generate Teaching Materials:**
```bash
# Generate flowcharts and diagrams
python docs/model_flowcharts.py

# Create PowerPoint presentation
python docs/create_presentation.py
```

---

### Overview for Students

This repository is designed as a teaching tool for 3rd-year hydraulic engineering students. It provides Python implementations of classic hydrological models, focusing on bridging the gap between theory and practical application. Here, you will not only learn the mathematical principles behind these models but also understand how they are calibrated and validated using real-world data.

### Core Concepts Covered

* **Rainfall-Runoff Process**: Understand how different models conceptualize the transformation of precipitation into streamflow.
* **Model Structure**: Compare lumped vs. semi-distributed concepts and saturation excess vs. infiltration excess mechanisms.
* **Parameter Sensitivity**: Learn how model parameters influence simulation results.
* **Model Calibration & Validation**: Master the essential workflow of hydrological modeling: using historical data to optimize parameters and validating the model on an independent dataset.
* **Performance Metrics**: Learn to evaluate model performance using standard metrics like Nash-Sutcliffe Efficiency (NSE).

### Implemented Models / 已实现的模型

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

#### 5. **HBV Model (Hydrologiska Byråns Vattenbalansavdelning)**

- **Type**: Conceptual, snow and soil moisture accounting with temperature-driven processes.
- **Best for**: Cold and temperate regions with snow accumulation and melt.
- **Features**: Snow routine (degree-day method), soil moisture accounting, three-component runoff generation.

#### 6. **SCS-CN + Unit Hydrograph (Event Model)**

- **Type**: Event-based runoff estimation and routing.
- **Best for**: Storm event analysis and design flood estimation.
- **Features**: SCS Curve Number method for runoff estimation, unit hydrograph for routing, antecedent moisture condition adjustments.

#### 7. **TOPMODEL-inspired Benchmark**

- **Type**: Conceptual, terrain-index-driven saturation excess.
- **Best for**: Demonstrating the role of topography in runoff production.
- **Features**: Spatially distributed saturation deficit, exponential transmissivity decay, complementary perspective to soil-moisture-based models.

#### 8. **Random Forest ML Baseline**

- **Type**: Data-driven ensemble regression.
- **Best for**: Benchmarking against conceptual models using the same forcings.
- **Features**: Uses rainfall and antecedent flow lags as predictors, reports NSE/RMSE/PBIAS alongside conceptual model outputs.

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
# This script compares all continuous models under various scenarios
python examples.py

# Try alternative catchment stories
python - <<'PY'
from examples import compare_all_models

# Humid vs. arid climates
compare_all_models(scenario="humid")
compare_all_models(scenario="arid")

# Drought followed by extreme rainfall with ML benchmark disabled
compare_all_models(scenario="extreme_event", include_ml=False, reservoir_residence_time=5.0)
PY
```

This will demonstrate model comparison, sensitivity analysis, and storm event simulation.

**Running Individual Models:**

Each model can be run independently for detailed demonstrations:

```bash
# Run HBV model (includes temperature-based processes)
python hbv_model.py

# Run SCS-CN + Unit Hydrograph event model
python event_model_scs_uh.py

# Run other continuous models
python xinanjiang_model.py
python tank_model.py
python gr4j_model.py
python sacramento_model.py
```

Each model generates comprehensive visualizations in the `figures/` directory.

### Interactive Teaching Notebook

For an interactive learning experience, explore the Jupyter notebook:

```bash
# Open the teaching quickstart notebook
jupyter notebook notebooks/teaching_quickstart.ipynb
```

This notebook provides:
* Step-by-step demonstrations of HBV and SCS-CN+UH models
* Visualizations of model behavior and outputs
* Land use scenario comparisons
* Bilingual explanations (English/中文)

The notebook uses the teaching dataset (`data/example_teaching_dataset.csv`) which contains 98 days of hydrological data including precipitation, temperature, PET, and observed flow.

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

### Project Structure / 项目结构

```
Hydrological-model/
├── data/
│   ├── sample_data.csv               # Sample real-world data / 真实数据样例
│   ├── hourly_forcings.csv           # Hourly meteorological data / 小时气象数据
│   └── example_teaching_dataset.csv  # Teaching dataset (NEW) / 教学数据集(新增)
├── notebooks/
│   └── teaching_quickstart.ipynb     # Teaching notebook (NEW) / 教学笔记本(新增)
├── docs/                              # Teaching materials (NEW) / 教学材料(新增)
│   ├── TEACHING_GUIDE.md             # Complete teaching guide / 完整教学指南
│   ├── model_flowcharts.py           # Generate flowcharts / 生成流程图
│   ├── create_presentation.py        # Generate PPT / 生成演示文稿
│   └── HydroLearn_Teaching_Presentation.pptx  # PowerPoint slides / PPT幻灯片
├── figures/                           # Output figures and diagrams / 输出图表和图解
│   ├── hbv_model_flowchart.png       # HBV model flowchart / HBV模型流程图
│   ├── xinanjiang_model_flowchart.png # Xinanjiang flowchart / 新安江流程图
│   ├── water_cycle_diagram.png       # Water cycle diagram / 水文循环图
│   └── model_comparison_table.png    # Model comparison / 模型对比表
├── README.md                         # This file / 本文件
├── requirements.txt                  # Python dependencies / Python依赖
├── xinanjiang_model.py               # Xinanjiang model / 新安江模型
├── tank_model.py                     # Tank model / Tank模型
├── gr4j_model.py                     # GR4J model / GR4J模型
├── sacramento_model.py               # Sacramento model / Sacramento模型
├── hbv_model.py                      # HBV model (NEW) / HBV模型(新增)
├── event_model_scs_uh.py             # SCS-CN + UH event model (NEW) / SCS-CN + 单位线事件模型(新增)
├── examples.py                       # Examples with synthetic data / 基于虚拟数据的示例
└── calibration_example.py            # Calibration with real data / 基于真实数据的率定与验证
```

### For Your Assignment

1. **Try calibrating other models**: Modify `calibration_example.py` to calibrate the Xinanjiang or Tank model. Compare which model performs better on the sample dataset.
2. **Adjust the calibration period**: Change the split date between calibration and validation. How does this affect the results?
3. **Experiment with parameters**: Manually change the parameters of a model and observe how the simulated hydrograph changes.

-----

<a name="中文"></a>

## 中文

### 🎓 新增：教学材料现已提供！

我们添加了全面的教学材料来帮助学生和教师：

- 📊 **[PowerPoint演示文稿](docs/HydroLearn_Teaching_Presentation.pptx)** - 18页幻灯片涵盖所有模型
- 📖 **[教学指南](docs/TEACHING_GUIDE.md)** - 完整学期课程含实践练习
- 🎨 **[模型流程图](figures/)** - HBV、新安江等模型的可视化图表
  - `hbv_model_flowchart.png` - HBV模型结构流程图
  - `xinanjiang_model_flowchart.png` - 新安江模型结构流程图
  - `water_cycle_diagram.png` - 水文循环概念图
  - `model_comparison_table.png` - 模型对比表

**生成教学材料：**
```bash
# 生成流程图和图表
python docs/model_flowcharts.py

# 创建PowerPoint演示文稿
python docs/create_presentation.py
```

---

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

#### 5. **HBV模型 (瑞典水文局水量平衡模型)**

- **类型**: 概念性模型，具有积雪和土壤水分核算及温度驱动过程。
- **适用于**: 有积雪累积和融化的寒冷和温带地区。
- **特点**: 积雪模块(度日法)、土壤水分核算、三成分径流产生。

#### 6. **SCS-CN + 单位线模型 (事件模型)**

- **类型**: 基于事件的径流估算和汇流模型。
- **适用于**: 暴雨事件分析和设计洪水估算。
- **特点**: SCS曲线数法进行径流估算、单位线汇流、前期湿度条件调整。

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
# 该脚本在多种情境下对比所有连续型模型
python examples.py
```

这将为您展示模型对比、参数敏感性分析和暴雨洪水模拟等功能。

**运行单个模型:**

每个模型都可以独立运行以获得详细演示:

```bash
# 运行HBV模型(包括基于温度的过程)
python hbv_model.py

# 运行SCS-CN + 单位线事件模型
python event_model_scs_uh.py

# 运行其他连续型模型
python xinanjiang_model.py
python tank_model.py
python gr4j_model.py
python sacramento_model.py
```

每个模型在`figures/`目录中生成综合可视化图表。

### 交互式教学笔记本

获得交互式学习体验，请探索Jupyter笔记本：

```bash
# 打开教学快速入门笔记本
jupyter notebook notebooks/teaching_quickstart.ipynb
```

此笔记本提供：
* HBV和SCS-CN+UH模型的分步演示
* 模型行为和输出的可视化
* 土地利用情景比较
* 双语解释（英文/中文）

笔记本使用教学数据集（`data/example_teaching_dataset.csv`），其中包含98天的水文数据，包括降水、温度、PET和观测流量。

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
│   ├── sample_data.csv               # 真实数据样例
│   ├── hourly_forcings.csv           # 小时气象数据
│   └── example_teaching_dataset.csv  # 教学数据集(新增)
├── notebooks/
│   └── teaching_quickstart.ipynb     # 教学笔记本(新增)
├── docs/                              # 教学材料(新增)
│   ├── TEACHING_GUIDE.md             # 完整教学指南
│   ├── model_flowcharts.py           # 生成流程图
│   ├── create_presentation.py        # 生成演示文稿
│   └── HydroLearn_Teaching_Presentation.pptx  # PowerPoint幻灯片
├── figures/                           # 输出图表和图解
│   ├── hbv_model_flowchart.png       # HBV模型流程图
│   ├── xinanjiang_model_flowchart.png # 新安江流程图
│   ├── water_cycle_diagram.png       # 水文循环图
│   └── model_comparison_table.png    # 模型对比表
├── README.md                         # 本文件
├── requirements.txt                  # Python依赖
├── xinanjiang_model.py               # 新安江模型
├── tank_model.py                     # Tank模型
├── gr4j_model.py                     # GR4J模型
├── sacramento_model.py               # Sacramento模型
├── hbv_model.py                      # HBV模型(新增)
├── event_model_scs_uh.py             # SCS-CN + 单位线事件模型(新增)
├── examples.py                       # 基于虚拟数据的示例
└── calibration_example.py            # 基于真实数据的率定与验证
```
├── gr4j_model.py                     # GR4J模型
├── sacramento_model.py               # Sacramento模型
├── hbv_model.py                      # HBV模型(新增)
├── event_model_scs_uh.py             # SCS-CN + 单位线事件模型(新增)
├── examples.py                       # 基于虚拟数据的示例
└── calibration_example.py            # 基于真实数据的率定与验证
```

### 课后思考与练习

1. **率定其他模型**: 修改`calibration_example.py`，尝试率定新安江模型或Tank模型，并比较哪个模型在该数据集上表现更好。
2. **调整率定周期**: 更改率定期与验证期的分割点，观察结果有何变化？
3. **手动调整参数**: 在不使用优化算法的情况下，手动修改某个模型的参数，观察流量过程线如何响应。

