"""
Model Flowcharts and Conceptual Diagrams Generator
模型流程图和概念图生成器

This script generates comprehensive flowcharts and diagrams for each hydrological model
to aid in teaching and understanding the model structures.

这个脚本为每个水文模型生成综合流程图和图表，以辅助教学和理解模型结构。

Author: HydroLearn Teaching Team
Date: 2024
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Polygon
from matplotlib.patches import ConnectionPatch
from matplotlib.lines import Line2D
import numpy as np
import os
import matplotlib.font_manager as fm

def _pick_cjk_font():
        """Pick a best-available CJK-capable sans-serif font installed on the system."""
        preferred = [
                # Simplified Chinese first
                'Microsoft YaHei', 'Microsoft YaHei UI', 'SimHei',
                'Noto Sans CJK SC', 'Source Han Sans CN', 'WenQuanYi Zen Hei', 'Sarasa UI SC',
                # Japanese as fallback for mixed content like タンク
                'Noto Sans CJK JP', 'Meiryo', 'Yu Gothic', 'MS Gothic',
                # Pan-Unicode fallbacks
                'Arial Unicode MS', 'DejaVu Sans'
        ]
        available = {f.name for f in fm.fontManager.ttflist}
        for name in preferred:
                if name in available:
                        return name
        return 'DejaVu Sans'


def configure_cjk_fonts():
    """Configure matplotlib to render Chinese/Japanese text and minus signs correctly."""
    chosen = _pick_cjk_font()
    # Put chosen first, then add symbol-capable fallbacks to cover checkmarks, etc.
    plt.rcParams['font.sans-serif'] = [
        chosen,
        'Segoe UI Symbol',
        'Noto Sans Symbols',
        'Noto Sans CJK SC',
        'SimHei',
        'Arial Unicode MS',
        'DejaVu Sans'
    ]
    # Ensure minus signs display properly when using CJK fonts
    plt.rcParams['axes.unicode_minus'] = False


# Apply font configuration early
configure_cjk_fonts()


def create_hbv_flowchart(save_dir='figures'):
    """
    Create a comprehensive flowchart for the HBV model structure.
    创建HBV模型结构的综合流程图。
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 16))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis('off')
    
    # Title
    ax.text(5, 19, 'HBV Model Structure Flowchart\nHBV模型结构流程图', 
            ha='center', va='top', fontsize=18, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', edgecolor='navy', linewidth=2))
    
    # Input section
    ax.text(5, 17.5, 'Inputs / 输入数据', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Precipitation and Temperature boxes
    precip_box = FancyBboxPatch((1, 16.5), 2, 0.6, boxstyle="round,pad=0.1", 
                                 edgecolor='blue', facecolor='lightblue', linewidth=2)
    ax.add_patch(precip_box)
    ax.text(2, 16.8, 'P (降水)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    temp_box = FancyBboxPatch((4, 16.5), 2, 0.6, boxstyle="round,pad=0.1",
                               edgecolor='red', facecolor='lightyellow', linewidth=2)
    ax.add_patch(temp_box)
    ax.text(5, 16.8, 'T (气温)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    pet_box = FancyBboxPatch((7, 16.5), 2, 0.6, boxstyle="round,pad=0.1",
                              edgecolor='orange', facecolor='wheat', linewidth=2)
    ax.add_patch(pet_box)
    ax.text(8, 16.8, 'PET (蒸散发)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Snow Routine Module
    snow_module = FancyBboxPatch((1.5, 14), 7, 2, boxstyle="round,pad=0.15",
                                  edgecolor='darkblue', facecolor='aliceblue', linewidth=3)
    ax.add_patch(snow_module)
    ax.text(5, 15.5, 'Snow Routine / 积雪模块', ha='center', va='top', 
            fontsize=11, fontweight='bold', color='darkblue')
    
    # Snow accumulation and melt process
    ax.text(3, 15, 'if T < TT:', ha='center', va='center', fontsize=9)
    ax.text(3, 14.7, 'Snow = Snow + P', ha='center', va='center', fontsize=9, style='italic')
    ax.text(7, 15, 'if T > TT:', ha='center', va='center', fontsize=9)
    ax.text(7, 14.7, 'Melt = CFMAX×(T-TT)', ha='center', va='center', fontsize=9, style='italic')
    ax.text(5, 14.3, 'Liquid Water = Rain + Snowmelt', ha='center', va='center', fontsize=9)
    
    # Arrow from inputs to snow module
    arrow1 = FancyArrowPatch((2, 16.5), (3, 16), arrowstyle='->', mutation_scale=20, 
                             linewidth=2, color='blue')
    ax.add_patch(arrow1)
    arrow2 = FancyArrowPatch((5, 16.5), (5, 16), arrowstyle='->', mutation_scale=20,
                             linewidth=2, color='red')
    ax.add_patch(arrow2)
    
    # Soil Routine Module
    soil_module = FancyBboxPatch((1.5, 10.5), 7, 3, boxstyle="round,pad=0.15",
                                  edgecolor='brown', facecolor='wheat', linewidth=3)
    ax.add_patch(soil_module)
    ax.text(5, 13.2, 'Soil Moisture Routine / 土壤水分模块', ha='center', va='top',
            fontsize=11, fontweight='bold', color='brown')
    
    # Soil moisture processes
    ax.text(5, 12.5, 'Actual ET / 实际蒸散发:', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(5, 12.2, 'EA = PET × min(SM/(LP×FC), 1)', ha='center', va='center', fontsize=9, style='italic')
    ax.text(5, 11.7, 'Recharge / 补给:', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(5, 11.4, 'R = (SM/FC)^BETA × (Rain - EA)', ha='center', va='center', fontsize=9, style='italic')
    ax.text(5, 11, 'SM: Soil Moisture / 土壤水分', ha='center', va='center', fontsize=8)
    
    # Arrow from snow to soil
    arrow3 = FancyArrowPatch((5, 14), (5, 13.5), arrowstyle='->', mutation_scale=20,
                             linewidth=2.5, color='navy')
    ax.add_patch(arrow3)
    ax.text(5.5, 13.7, 'Liquid\nWater', ha='left', va='center', fontsize=8, color='navy')
    
    # Arrow from PET to soil
    arrow_pet = FancyArrowPatch((8, 16.5), (7.5, 12), arrowstyle='->', mutation_scale=15,
                                linewidth=1.5, color='orange', linestyle='--')
    ax.add_patch(arrow_pet)
    
    # Response Routine Module - Three Zones
    response_module = FancyBboxPatch((1.5, 4.5), 7, 5.5, boxstyle="round,pad=0.15",
                                      edgecolor='darkgreen', facecolor='lightgreen', linewidth=3)
    ax.add_patch(response_module)
    ax.text(5, 9.7, 'Response Routine / 径流响应模块', ha='center', va='top',
            fontsize=11, fontweight='bold', color='darkgreen')
    
    # Upper Zone
    upper_zone = Rectangle((2, 7.5), 6, 1.8, edgecolor='red', facecolor='lightyellow', linewidth=2)
    ax.add_patch(upper_zone)
    ax.text(5, 9, 'Upper Zone / 上层蓄水 (SUZ)', ha='center', va='center',
            fontsize=10, fontweight='bold')
    ax.text(5, 8.6, 'Q0 = K0 × max(SUZ - UZL, 0)  [快速径流]', ha='center', va='center', fontsize=8)
    ax.text(5, 8.3, 'Q1 = K1 × SUZ  [壤中流]', ha='center', va='center', fontsize=8)
    ax.text(5, 8, 'Percolation ↓', ha='center', va='center', fontsize=8, style='italic')
    
    # Lower Zone
    lower_zone = Rectangle((2, 5.5), 6, 1.5, edgecolor='purple', facecolor='lavender', linewidth=2)
    ax.add_patch(lower_zone)
    ax.text(5, 6.5, 'Lower Zone / 下层蓄水 (SLZ)', ha='center', va='center',
            fontsize=10, fontweight='bold')
    ax.text(5, 6.1, 'Q2 = K2 × SLZ  [基流]', ha='center', va='center', fontsize=8)
    
    # Arrow from soil to response
    arrow4 = FancyArrowPatch((5, 10.5), (5, 9.3), arrowstyle='->', mutation_scale=20,
                             linewidth=2.5, color='brown')
    ax.add_patch(arrow4)
    ax.text(5.5, 10, 'Recharge', ha='left', va='center', fontsize=8, color='brown')
    
    # Percolation arrow
    arrow_perc = FancyArrowPatch((5, 7.5), (5, 7), arrowstyle='->', mutation_scale=15,
                                 linewidth=2, color='blue')
    ax.add_patch(arrow_perc)
    
    # Runoff outputs
    arrow_q0 = FancyArrowPatch((8, 8.8), (9, 8.8), arrowstyle='->', mutation_scale=15,
                               linewidth=2, color='red')
    ax.add_patch(arrow_q0)
    ax.text(9.2, 8.8, 'Q0', ha='left', va='center', fontsize=9, fontweight='bold', color='red')
    
    arrow_q1 = FancyArrowPatch((8, 8.3), (9, 8.3), arrowstyle='->', mutation_scale=15,
                               linewidth=2, color='green')
    ax.add_patch(arrow_q1)
    ax.text(9.2, 8.3, 'Q1', ha='left', va='center', fontsize=9, fontweight='bold', color='green')
    
    arrow_q2 = FancyArrowPatch((8, 6.1), (9, 6.1), arrowstyle='->', mutation_scale=15,
                               linewidth=2, color='purple')
    ax.add_patch(arrow_q2)
    ax.text(9.2, 6.1, 'Q2', ha='left', va='center', fontsize=9, fontweight='bold', color='purple')
    
    # Routing Module
    routing_box = FancyBboxPatch((1.5, 2), 7, 2, boxstyle="round,pad=0.15",
                                  edgecolor='navy', facecolor='lightcyan', linewidth=3)
    ax.add_patch(routing_box)
    ax.text(5, 3.6, 'Routing / 汇流模块', ha='center', va='top',
            fontsize=11, fontweight='bold', color='navy')
    ax.text(5, 3.1, 'Triangular Weighting Function', ha='center', va='center', fontsize=9)
    ax.text(5, 2.7, 'Q(t) = Σ [Q_gen × weight_function]', ha='center', va='center', 
            fontsize=9, style='italic')
    ax.text(5, 2.3, 'MAXBAS: 汇流参数', ha='center', va='center', fontsize=8)
    
    # Arrow to routing
    arrow5 = FancyArrowPatch((5, 4.5), (5, 4), arrowstyle='->', mutation_scale=20,
                             linewidth=2.5, color='darkgreen')
    ax.add_patch(arrow5)
    ax.text(5.5, 4.2, 'Q_gen', ha='left', va='center', fontsize=8, color='darkgreen')
    
    # Final Output
    output_box = FancyBboxPatch((3.5, 0.5), 3, 1, boxstyle="round,pad=0.1",
                                 edgecolor='navy', facecolor='gold', linewidth=3)
    ax.add_patch(output_box)
    ax.text(5, 1, 'Total Discharge / 总径流\nQ = Q0 + Q1 + Q2', ha='center', va='center',
            fontsize=11, fontweight='bold', color='navy')
    
    # Arrow to output
    arrow6 = FancyArrowPatch((5, 2), (5, 1.5), arrowstyle='->', mutation_scale=25,
                             linewidth=3, color='navy')
    ax.add_patch(arrow6)
    
    # Key Parameters Box
    param_box = FancyBboxPatch((0.2, 0.2), 2.5, 3.5, boxstyle="round,pad=0.1",
                                edgecolor='black', facecolor='lightyellow', linewidth=2)
    ax.add_patch(param_box)
    ax.text(1.45, 3.5, 'Key Parameters\n关键参数', ha='center', va='top',
            fontsize=9, fontweight='bold')

    params_text = """Snow / 积雪:
TT, CFMAX, CWH

Soil / 土壤:
FC, LP, BETA

Response / 响应:
PERC, UZL
K0, K1, K2

Routing / 汇流:
MAXBAS"""

    # Avoid forcing monospace which often lacks CJK glyphs
    ax.text(1.45, 3, params_text, ha='center', va='top', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'hbv_model_flowchart.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Created HBV model flowchart: {save_dir}/hbv_model_flowchart.png")


def create_xinanjiang_flowchart(save_dir='figures'):
    """
    Create a comprehensive flowchart for the Xinanjiang model structure.
    创建新安江模型结构的综合流程图。
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 16))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis('off')
    
    # Title
    ax.text(5, 19, 'Xinanjiang Model Structure Flowchart\n新安江模型结构流程图', 
            ha='center', va='top', fontsize=18, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', edgecolor='darkgreen', linewidth=2))
    
    # Input section
    ax.text(5, 17.5, 'Inputs / 输入数据', ha='center', va='center', fontsize=12, fontweight='bold')
    
    precip_box = FancyBboxPatch((2, 16.5), 2.5, 0.6, boxstyle="round,pad=0.1",
                                 edgecolor='blue', facecolor='lightblue', linewidth=2)
    ax.add_patch(precip_box)
    ax.text(3.25, 16.8, 'P (降水)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    pet_box = FancyBboxPatch((5.5, 16.5), 2.5, 0.6, boxstyle="round,pad=0.1",
                              edgecolor='orange', facecolor='wheat', linewidth=2)
    ax.add_patch(pet_box)
    ax.text(6.75, 16.8, 'EP (蒸散发)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Three-Layer Evapotranspiration
    et_module = FancyBboxPatch((1.5, 13.5), 7, 2.5, boxstyle="round,pad=0.15",
                                edgecolor='orange', facecolor='moccasin', linewidth=3)
    ax.add_patch(et_module)
    ax.text(5, 15.7, 'Three-Layer Evapotranspiration / 三层蒸散发', ha='center', va='top',
            fontsize=11, fontweight='bold', color='darkorange')
    
    # Three layers visualization
    upper_layer = Rectangle((2, 14.8), 2, 0.5, edgecolor='brown', facecolor='wheat', linewidth=2)
    ax.add_patch(upper_layer)
    ax.text(3, 15.05, 'Upper / 上层\nEU = EP×(W/WUM)', ha='center', va='center', fontsize=8)
    
    lower_layer = Rectangle((4.5, 14.8), 2, 0.5, edgecolor='brown', facecolor='tan', linewidth=2)
    ax.add_patch(lower_layer)
    ax.text(5.5, 15.05, 'Lower / 下层\nEL', ha='center', va='center', fontsize=8)
    
    deep_layer = Rectangle((7, 14.8), 1.5, 0.5, edgecolor='brown', facecolor='sienna', linewidth=2)
    ax.add_patch(deep_layer)
    ax.text(7.75, 15.05, 'Deep / 深层\nED', ha='center', va='center', fontsize=8, color='white')
    
    ax.text(5, 14.3, 'E = EU + EL + ED', ha='center', va='center', fontsize=9, style='italic')
    ax.text(5, 13.9, 'W: Total Soil Moisture / 土壤总水量', ha='center', va='center', fontsize=8)
    
    # Arrows from inputs to ET module
    arrow1 = FancyArrowPatch((3.25, 16.5), (4, 16), arrowstyle='->', mutation_scale=20,
                             linewidth=2, color='blue')
    ax.add_patch(arrow1)
    arrow2 = FancyArrowPatch((6.75, 16.5), (6, 16), arrowstyle='->', mutation_scale=20,
                             linewidth=2, color='orange')
    ax.add_patch(arrow2)
    
    # Runoff Generation Module (Saturation Excess)
    runoff_gen = FancyBboxPatch((1.5, 10), 7, 3, boxstyle="round,pad=0.15",
                                 edgecolor='blue', facecolor='lightblue', linewidth=3)
    ax.add_patch(runoff_gen)
    ax.text(5, 12.7, 'Runoff Generation / 产流计算 (Saturation Excess / 蓄满产流)', 
            ha='center', va='top', fontsize=11, fontweight='bold', color='darkblue')
    
    # Parabolic curve illustration
    ax.text(5, 12, 'Parabolic Distribution Curve / 抛物线蓄水容量曲线:', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(5, 11.5, 'f = SM × (1 - (1 - A)^(1/(1+B)))', ha='center', va='center',
            fontsize=9, style='italic')
    ax.text(5, 11.1, 'if PE + A < SM:', ha='center', va='center', fontsize=8)
    ax.text(5, 10.7, 'R = PE + A - SM + SM×(1 - (PE+A)/SM)^(1+B)', 
            ha='center', va='center', fontsize=8, style='italic')
    ax.text(5, 10.3, 'B: 蓄水容量曲线指数参数', ha='center', va='center', fontsize=7)
    
    # Arrow from ET to Runoff Gen
    arrow3 = FancyArrowPatch((5, 13.5), (5, 13), arrowstyle='->', mutation_scale=20,
                             linewidth=2.5, color='navy')
    ax.add_patch(arrow3)
    ax.text(5.5, 13.2, 'PE=P-E', ha='left', va='center', fontsize=8, color='navy')
    
    # Runoff Separation Module
    separation = FancyBboxPatch((1.5, 6.5), 7, 3, boxstyle="round,pad=0.15",
                                 edgecolor='green', facecolor='lightgreen', linewidth=3)
    ax.add_patch(separation)
    ax.text(5, 9.2, 'Runoff Separation / 水源划分', ha='center', va='top',
            fontsize=11, fontweight='bold', color='darkgreen')
    
    # Three components
    ax.text(2.5, 8.5, 'Surface Runoff\n地表径流\nRS', ha='center', va='center',
            fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
    
    ax.text(5, 8.5, 'Interflow\n壤中流\nRI = KI × RSS', ha='center', va='center',
            fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))
    
    ax.text(7.5, 8.5, 'Groundwater\n地下水\nRG = (1-KI) × RSS', ha='center', va='center',
            fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='purple', alpha=0.3))
    
    ax.text(5, 7.5, 'Free Water Capacity Curve / 自由水蓄水容量曲线', 
            ha='center', va='center', fontsize=8, style='italic')
    ax.text(5, 7.1, 'EX: 自由水蓄水容量曲线指数', ha='center', va='center', fontsize=7)
    
    # Arrow from Runoff Gen to Separation
    arrow4 = FancyArrowPatch((5, 10), (5, 9.5), arrowstyle='->', mutation_scale=20,
                             linewidth=2.5, color='blue')
    ax.add_patch(arrow4)
    ax.text(5.5, 9.7, 'R', ha='left', va='center', fontsize=8, color='blue')
    
    # Flow Routing Module
    routing = FancyBboxPatch((1.5, 3), 7, 3, boxstyle="round,pad=0.15",
                              edgecolor='purple', facecolor='lavender', linewidth=3)
    ax.add_patch(routing)
    ax.text(5, 5.7, 'Flow Routing / 汇流计算', ha='center', va='top',
            fontsize=11, fontweight='bold', color='purple')
    
    # Routing components
    ax.text(2.5, 5, 'Surface\nNo Routing\nQS = RS', ha='center', va='center',
            fontsize=8, bbox=dict(boxstyle='round', facecolor='red', alpha=0.2))
    
    ax.text(5, 5, 'Interflow\nLinear Reservoir\nSI = CI×SI + RI\nQI = (1-CI)×SI', 
            ha='center', va='center', fontsize=7.5,
            bbox=dict(boxstyle='round', facecolor='green', alpha=0.2))
    
    ax.text(7.5, 5, 'Groundwater\nLinear Reservoir\nSG = CG×SG + RG\nQG = (1-CG)×SG',
            ha='center', va='center', fontsize=7.5,
            bbox=dict(boxstyle='round', facecolor='purple', alpha=0.2))
    
    ax.text(5, 3.5, 'CI, CG: Recession Constants / 消退系数', ha='center', va='center', fontsize=7)
    
    # Arrows from Separation to Routing
    arrow5 = FancyArrowPatch((2.5, 6.5), (2.5, 5.5), arrowstyle='->', mutation_scale=15,
                             linewidth=2, color='red')
    ax.add_patch(arrow5)
    arrow6 = FancyArrowPatch((5, 6.5), (5, 5.5), arrowstyle='->', mutation_scale=15,
                             linewidth=2, color='green')
    ax.add_patch(arrow6)
    arrow7 = FancyArrowPatch((7.5, 6.5), (7.5, 5.5), arrowstyle='->', mutation_scale=15,
                             linewidth=2, color='purple')
    ax.add_patch(arrow7)
    
    # Final Output
    output_box = FancyBboxPatch((3, 1), 4, 1.5, boxstyle="round,pad=0.1",
                                 edgecolor='navy', facecolor='gold', linewidth=3)
    ax.add_patch(output_box)
    ax.text(5, 1.75, 'Total Discharge / 总径流', ha='center', va='center',
            fontsize=11, fontweight='bold', color='navy')
    ax.text(5, 1.3, 'Q = QS + QI + QG', ha='center', va='center',
            fontsize=10, fontweight='bold', style='italic')
    
    # Arrows to output
    arrow8 = FancyArrowPatch((5, 3), (5, 2.5), arrowstyle='->', mutation_scale=25,
                             linewidth=3, color='navy')
    ax.add_patch(arrow8)
    
    # Key Parameters Box
    param_box = FancyBboxPatch((0.2, 0.2), 2, 4, boxstyle="round,pad=0.1",
                                edgecolor='black', facecolor='lightyellow', linewidth=2)
    ax.add_patch(param_box)
    ax.text(1.2, 4, 'Key Parameters\n关键参数', ha='center', va='top',
            fontsize=9, fontweight='bold')

    params_text = """Evaporation / 蒸发:
K, C

Soil / 土壤:
WM, WUM, WLM
B (curve shape)

Runoff / 产流:
SM, EX, IMP

Routing / 汇流:
KI, KG
CI, CG"""

    # Avoid forcing monospace which often lacks CJK glyphs
    ax.text(1.2, 3.5, params_text, ha='center', va='top', fontsize=7)

    # Model Feature Box
    feature_box = FancyBboxPatch((8.2, 0.2), 1.6, 3, boxstyle="round,pad=0.1",
                                  edgecolor='darkgreen', facecolor='lightgreen', linewidth=2)
    ax.add_patch(feature_box)
    ax.text(9, 3, 'Model Type\n模型类型', ha='center', va='top',
            fontsize=9, fontweight='bold')

    features_text = """• Saturation
  Excess / 蓄满
  产流

• Humid
  Regions / 湿
  润地区

• Conceptual
  概念性"""

    ax.text(9, 2.5, features_text, ha='center', va='top', fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'xinanjiang_model_flowchart.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Created Xinanjiang model flowchart: {save_dir}/xinanjiang_model_flowchart.png")


def create_water_cycle_diagram(save_dir='figures'):
    """
    Create a conceptual water cycle diagram for hydrological modeling.
    创建水文模拟的水循环概念图。
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(7, 9.5, 'Hydrological Water Cycle / 水文循环概念图', 
            ha='center', va='top', fontsize=16, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', edgecolor='navy', linewidth=2))
    
    # Atmosphere (clouds)
    cloud1 = mpatches.Ellipse((3, 7.5), 2, 0.8, edgecolor='gray', facecolor='lightgray', linewidth=2)
    ax.add_patch(cloud1)
    cloud2 = mpatches.Ellipse((11, 7.5), 2, 0.8, edgecolor='gray', facecolor='lightgray', linewidth=2)
    ax.add_patch(cloud2)
    
    ax.text(7, 7.5, 'Atmosphere / 大气', ha='center', va='center',
            fontsize=12, fontweight='bold')
    
    # Precipitation arrows
    for x in [2.5, 3.5, 10.5, 11.5]:
        for i in range(3):
            arrow = FancyArrowPatch((x, 6.8 - i*0.3), (x, 6.5 - i*0.3), 
                                   arrowstyle='->', mutation_scale=15,
                                   linewidth=1.5, color='blue', alpha=0.6)
            ax.add_patch(arrow)
    
    ax.text(3, 6.2, 'Precipitation\n降水 (P)', ha='center', va='top',
            fontsize=10, fontweight='bold', color='blue')
    ax.text(11, 6.2, 'Precipitation\n降水 (P)', ha='center', va='top',
            fontsize=10, fontweight='bold', color='blue')

    # Land surface
    land_line = Line2D([0.5, 13.5], [5, 5], color='brown', linewidth=3)
    ax.add_line(land_line)
    
    # Vegetation/Interception
    tree_trunk = Rectangle((2.8, 5), 0.4, 0.8, edgecolor='brown', facecolor='saddlebrown')
    ax.add_patch(tree_trunk)
    tree_crown = mpatches.Circle((3, 6.2), 0.5, edgecolor='darkgreen', facecolor='green')
    ax.add_patch(tree_crown)
    ax.text(3, 4.5, 'Vegetation\n植被', ha='center', va='top', fontsize=8)
    
    # Evapotranspiration arrows
    for x in [2.7, 3, 3.3]:
        arrow_et = FancyArrowPatch((x, 6.5), (x, 7.2), 
                                  arrowstyle='->', mutation_scale=12,
                                  linewidth=1.5, color='orange', linestyle='--')
        ax.add_patch(arrow_et)
    
    ax.text(4.5, 7, 'Evapotranspiration\n蒸散发 (ET)', ha='center', va='center',
            fontsize=9, fontweight='bold', color='orange')
    
    # Soil layers
    soil_upper = Rectangle((5, 3.5), 4, 1.5, edgecolor='brown', facecolor='wheat', linewidth=2)
    ax.add_patch(soil_upper)
    ax.text(7, 4.5, 'Soil Zone / 土壤层', ha='center', va='center',
            fontsize=10, fontweight='bold')
    ax.text(7, 4.1, 'Soil Moisture / 土壤水分', ha='center', va='center', fontsize=8)
    
    # Infiltration
    for x in [6, 7, 8]:
        arrow_inf = FancyArrowPatch((x, 5), (x, 4.5), 
                                   arrowstyle='->', mutation_scale=15,
                                   linewidth=2, color='blue')
        ax.add_patch(arrow_inf)
    
    ax.text(9.5, 4.7, 'Infiltration\n下渗', ha='center', va='center',
            fontsize=8, color='blue')
    
    # Unsaturated zone
    unsat_zone = Rectangle((5, 2), 4, 1.5, edgecolor='steelblue', facecolor='lightblue', linewidth=2)
    ax.add_patch(unsat_zone)
    ax.text(7, 2.8, 'Unsaturated Zone\n非饱和带', ha='center', va='center',
            fontsize=9, fontweight='bold')
    
    # Groundwater zone
    gw_zone = Rectangle((5, 0.5), 4, 1.5, edgecolor='darkblue', facecolor='cyan', linewidth=2)
    ax.add_patch(gw_zone)
    ax.text(7, 1.3, 'Groundwater Zone\n地下水带', ha='center', va='center',
            fontsize=9, fontweight='bold', color='darkblue')
    
    # Percolation arrow
    arrow_perc = FancyArrowPatch((7, 2), (7, 1.5), 
                                arrowstyle='->', mutation_scale=20,
                                linewidth=2.5, color='blue')
    ax.add_patch(arrow_perc)
    ax.text(7.7, 1.7, 'Percolation\n渗漏', ha='left', va='center',
            fontsize=8, color='blue')
    
    # Surface runoff
    for i in range(4):
        x_start = 10.5 - i*0.3
        arrow_surf = FancyArrowPatch((x_start, 5), (x_start - 0.3, 4.7), 
                                    arrowstyle='->', mutation_scale=12,
                                    linewidth=2, color='red')
        ax.add_patch(arrow_surf)
    
    ax.text(11, 4.3, 'Surface Runoff\n地表径流', ha='center', va='center',
            fontsize=9, fontweight='bold', color='red')
    
    # Interflow
    arrow_inter = FancyArrowPatch((9, 4), (10.5, 3.2), 
                                 arrowstyle='->', mutation_scale=20,
                                 linewidth=2.5, color='green')
    ax.add_patch(arrow_inter)
    ax.text(10, 3.5, 'Interflow\n壤中流', ha='center', va='center',
            fontsize=9, fontweight='bold', color='green')
    
    # Baseflow
    arrow_base = FancyArrowPatch((9, 1.3), (11, 2), 
                                arrowstyle='->', mutation_scale=20,
                                linewidth=2.5, color='purple')
    ax.add_patch(arrow_base)
    ax.text(10.5, 1.5, 'Baseflow\n基流', ha='center', va='center',
            fontsize=9, fontweight='bold', color='purple')
    
    # River/Stream
    river = Rectangle((11.5, 1.5), 1.5, 2.5, edgecolor='navy', facecolor='lightblue', linewidth=3)
    ax.add_patch(river)
    ax.text(12.25, 2.7, 'River\n河流', ha='center', va='center',
            fontsize=10, fontweight='bold', color='navy')
    ax.text(12.25, 2.2, 'Total\nDischarge\n总径流', ha='center', va='center',
            fontsize=8, color='navy')
    
    # Evaporation from river
    for x in [11.8, 12.25, 12.7]:
        arrow_ev = FancyArrowPatch((x, 4), (x, 6.5), 
                                  arrowstyle='->', mutation_scale=10,
                                  linewidth=1.2, color='orange', linestyle='--')
        ax.add_patch(arrow_ev)
    
    ax.text(12.25, 6.7, 'Evaporation\n蒸发', ha='center', va='bottom',
            fontsize=8, color='orange')
    
    # Water balance equation box
    balance_box = FancyBboxPatch((0.5, 0.2), 4.5, 1.2, boxstyle="round,pad=0.1",
                                  edgecolor='navy', facecolor='lightyellow', linewidth=2)
    ax.add_patch(balance_box)
    ax.text(2.75, 1.1, 'Water Balance Equation / 水量平衡方程', ha='center', va='top',
            fontsize=10, fontweight='bold', color='navy')
    ax.text(2.75, 0.7, 'P = ET + Q + ΔS', ha='center', va='center',
            fontsize=11, fontweight='bold', style='italic')
    ax.text(2.75, 0.35, 'P: Precipitation, ET: Evapotranspiration', ha='center', va='center',
            fontsize=7)
    ax.text(2.75, 0.15, 'Q: Runoff, ΔS: Storage Change', ha='center', va='center',
            fontsize=7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'water_cycle_diagram.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Created water cycle diagram: {save_dir}/water_cycle_diagram.png")


def create_model_comparison_table(save_dir='figures'):
    """
    Create a visual comparison table of all models.
    创建所有模型的对比表格图。
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.axis('off')
    
    # Title
    fig.suptitle('Hydrological Models Comparison Table\n水文模型对比表', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Table data
    models = ['Xinanjiang\n新安江', 'Tank\nタンク', 'GR4J', 'Sacramento\nSAC-SMA', 
              'HBV', 'SCS-CN+UH']
    
    criteria = [
        'Type / 类型',
        'Mechanism / 机制',
        'Parameters / 参数数量',
        'Time Step / 时间步长',
        'Best For / 适用区域',
        'Complexity / 复杂度',
        'Snow Module / 积雪模块'
    ]
    
    data = [
        ['Conceptual\n概念性', 'Conceptual\n概念性', 'Conceptual\n概念性', 
         'Conceptual\n概念性', 'Conceptual\n概念性', 'Event-based\n事件型'],
        ['Saturation Excess\n蓄满产流', 'Multi-reservoir\n多水库', 'Lumped\n集总式',
         'Continuous SMA\n连续土壤水', 'Snow+Soil+Response\n雪+土+响应', 'Infiltration Excess\n超渗产流'],
        ['13', '10-15', '4', '11-17', '13', '2-3'],
        ['Daily\n日', 'Hourly/Daily\n小时/日', 'Daily\n日', 'Hourly/Daily\n小时/日',
         'Daily\n日', 'Event\n事件'],
        ['Humid regions\n湿润地区', 'Various\n多种', 'General\n通用',
         'Operational forecasting\n业务预报', 'Cold/Temperate\n寒冷/温带', 'Storm events\n暴雨事件'],
        ['Medium\n中等', 'Medium\n中等', 'Low\n低', 'High\n高',
         'Medium\n中等', 'Low\n低'],
        ['No\n无', 'No\n无', 'No\n无', 'Optional\n可选',
         'Yes\n有', 'No\n无']
    ]
    
    # Colors for each model
    colors = ['lightgreen', 'lightblue', 'lightyellow', 'lightcoral', 'lightcyan', 'wheat']
    
    # Create table
    table = ax.table(cellText=data, rowLabels=criteria, colLabels=models,
                     cellLoc='center', loc='center',
                     colWidths=[0.16]*6, rowLoc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 3)
    
    # Style the table
    for i, color in enumerate(colors):
        table[(0, i)].set_facecolor(color)
        table[(0, i)].set_text_props(weight='bold', fontsize=10)
    
    for i in range(len(criteria)):
        table[(i+1, -1)].set_facecolor('lightgray')
        table[(i+1, -1)].set_text_props(weight='bold', fontsize=9)
    
    # Add cell borders
    for key, cell in table.get_celld().items():
        cell.set_edgecolor('black')
        cell.set_linewidth(1.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison_table.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Created model comparison table: {save_dir}/model_comparison_table.png")


def main():
    """
    Generate all flowcharts and diagrams.
    生成所有流程图和图表。
    """
    print("=" * 80)
    print("Generating Teaching Flowcharts and Diagrams")
    print("生成教学流程图和图表")
    print("=" * 80)
    
    save_dir = 'figures'
    
    # Create all diagrams
    print("\nCreating HBV model flowchart...")
    create_hbv_flowchart(save_dir)
    
    print("\nCreating Xinanjiang model flowchart...")
    create_xinanjiang_flowchart(save_dir)
    
    print("\nCreating water cycle conceptual diagram...")
    create_water_cycle_diagram(save_dir)
    
    print("\nCreating model comparison table...")
    create_model_comparison_table(save_dir)
    
    print("\n" + "=" * 80)
    print("✓ All flowcharts and diagrams created successfully!")
    print("✓ 所有流程图和图表创建成功!")
    print(f"✓ Saved to: {save_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()

