"""
测试matplotlib中文字体显示
Test Chinese font display in matplotlib
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os

# 测试不同的字体配置
font_configs = [
    {
        'name': '配置1: SimHei, Microsoft YaHei',
        'fonts': ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    },
    {
        'name': '配置2: Microsoft YaHei, SimHei',
        'fonts': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    },
    {
        'name': '配置3: KaiTi, FangSong',
        'fonts': ['KaiTi', 'FangSong', 'SimHei', 'DejaVu Sans']
    },
    {
        'name': '配置4: STSong, STKaiti',
        'fonts': ['STSong', 'STKaiti', 'SimHei', 'DejaVu Sans']
    }
]

def test_font_config(config):
    """测试特定字体配置"""
    print(f"\n测试: {config['name']}")
    print(f"字体列表: {config['fonts']}")
    
    # 设置字体
    plt.rcParams['font.sans-serif'] = config['fonts']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建测试图形
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 测试数据
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    
    # 绘图
    ax.plot(x, y1, 'b-', linewidth=2, label='正弦波 Sine Wave')
    ax.plot(x, y2, 'r--', linewidth=2, label='余弦波 Cosine Wave')
    
    # 设置中文标题和标签
    ax.set_title('中文字体测试：降雨径流模拟\nChinese Font Test: Rainfall-Runoff Simulation', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('时间 (小时) Time (hours)', fontsize=12)
    ax.set_ylabel('流量 (mm/h) Discharge (mm/h)', fontsize=12)
    
    # 图例
    ax.legend(fontsize=11, loc='best')
    
    # 网格
    ax.grid(True, alpha=0.3)
    
    # 添加注释
    ax.annotate('峰值 Peak', xy=(np.pi/2, 1), xytext=(np.pi/2, 1.3),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # 添加文本框
    textstr = '模型参数 Model Parameters:\n'
    textstr += '• 曲线数 CN = 75\n'
    textstr += '• 前期湿度 AMC = II\n'
    textstr += '• 峰现时间 Tp = 3.0 h'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # 保存图形
    os.makedirs('figures', exist_ok=True)
    filename = f"figures/font_test_{config['fonts'][0].replace(' ', '_')}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ 保存到: {filename}")
    
    plt.close()
    
    return True

def main():
    print("=" * 70)
    print("matplotlib 中文字体显示测试")
    print("Matplotlib Chinese Font Display Test")
    print("=" * 70)
    
    print("\n系统中可用的中文字体:")
    fonts = sorted([f.name for f in fm.fontManager.ttflist])
    chinese_fonts = sorted(list(set([f for f in fonts if any(x in f for x in 
                          ['SimHei', 'YaHei', 'KaiTi', 'Song', 'FangSong', 'STXihei', 'STKaiti', 'STSong'])])))
    for font in chinese_fonts[:15]:  # 只显示前15个
        print(f"  • {font}")
    
    # 测试每个配置
    for config in font_configs:
        test_font_config(config)
    
    print("\n" + "=" * 70)
    print("测试完成！请检查 figures/ 目录下的图像文件")
    print("Test completed! Please check the image files in figures/ directory")
    print("=" * 70)
    
    # 推荐配置
    print("\n推荐的字体配置:")
    print("plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']")
    print("plt.rcParams['axes.unicode_minus'] = False")

if __name__ == "__main__":
    main()
