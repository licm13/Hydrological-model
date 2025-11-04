"""
测试中文字体显示
Test Chinese Font Display in Matplotlib
"""

import matplotlib.pyplot as plt
import numpy as np

# Configure matplotlib for Chinese font display / 配置matplotlib以显示中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display / 修复负号显示

def test_chinese_display():
    """测试中文字体显示是否正常"""
    
    # 创建测试数据
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制曲线
    ax.plot(x, y, 'b-', linewidth=2, label='正弦曲线')
    
    # 设置标题和标签
    ax.set_title('中文字体测试 / Chinese Font Test', fontsize=16, fontweight='bold')
    ax.set_xlabel('时间 (小时) / Time (hours)', fontsize=12)
    ax.set_ylabel('数值 / Value', fontsize=12)
    
    # 添加图例
    ax.legend(fontsize=11)
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 添加文本注释
    ax.text(5, 0.5, '这是中文文本测试\nChinese text display test', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=10, ha='center')
    
    plt.tight_layout()
    
    # 保存图形
    plt.savefig('figures/chinese_font_test.png', dpi=150, bbox_inches='tight')
    print("✓ 测试图形已保存到 'figures/chinese_font_test.png'")
    print("✓ Test figure saved to 'figures/chinese_font_test.png'")
    
    plt.show()
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("中文字体显示测试")
    print("Chinese Font Display Test")
    print("=" * 60)
    
    test_chinese_display()
    
    print("\n如果图形中的中文显示正常，说明字体配置成功！")
    print("If Chinese characters display correctly, font configuration is successful!")
