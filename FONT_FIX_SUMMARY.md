# 中文字体显示修复总结 / Chinese Font Display Fix Summary

## 修改日期 / Modification Date
2025-11-04 (Updated)

## 问题描述 / Problem Description

在matplotlib图形中，中文字符无法正常显示，显示为方框或乱码。特别是在使用`plt.style.use()`设置样式后，字体配置会被重置。

Chinese characters in matplotlib plots were not displaying correctly, showing as boxes or garbled text. Especially after using `plt.style.use()`, font configuration would be reset.

## 解决方案 / Solution

### 1. 在文件开头添加全局字体配置

在所有包含matplotlib绘图代码的Python文件中，在导入语句后添加：

```python
# Configure matplotlib for Chinese font display / 配置matplotlib以显示中文
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'STSong', 'KaiTi', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display / 修复负号显示
```

### 2. 在plt.style.use()后重新配置字体

**关键修复**：由于`plt.style.use()`会重置字体设置，需要在每次调用后重新配置：

```python
# Set style
plt.style.use('seaborn-v0_8-darkgrid')

# Re-configure Chinese font after style setting / 样式设置后重新配置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'STSong', 'KaiTi', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
```

### 字体优先级 / Font Priority
1. **Microsoft YaHei (微软雅黑)** - 最佳中文显示效果，无字形警告
2. **SimHei (黑体)** - 备选，可能有部分字形缺失
3. **STSong (华文宋体)** - 备选宋体
4. **KaiTi (楷体)** - 备选楷体  
5. **DejaVu Sans** - 后备英文字体

如果系统中没有前两个字体，会自动使用第三个后备字体。
If the first two fonts are not available, the fallback font will be used automatically.

## 修改的文件列表 / Modified Files

以下8个Python文件已被修改：

1. ✅ **event_model_scs_uh.py** - SCS-CN + 单位线事件模型
2. ✅ **xinanjiang_model.py** - 新安江模型
3. ✅ **tank_model.py** - Tank模型
4. ✅ **sacramento_model.py** - Sacramento SAC-SMA模型
5. ✅ **hbv_model.py** - HBV模型
6. ✅ **gr4j_model.py** - GR4J模型
7. ✅ **examples.py** - 综合示例脚本
8. ✅ **calibration_example.py** - 校准示例脚本

## 新增文件 / New Files

- **test_chinese_font.py** - 中文字体显示测试脚本

## 使用方法 / Usage

### 1. 测试中文字体显示 / Test Chinese Font Display

运行测试脚本：
```bash
python test_chinese_font.py
```

这将生成一个测试图形文件：`figures/chinese_font_test.png`

### 2. 运行原有脚本 / Run Original Scripts

所有原有脚本现在都能正确显示中文：

```bash
# 运行事件模型
python event_model_scs_uh.py

# 运行示例脚本
python examples.py

# 运行其他模型脚本
python xinanjiang_model.py
python hbv_model.py
python gr4j_model.py
# ... 等等
```

## 技术细节 / Technical Details

### 配置项说明 / Configuration Explanation

1. **plt.rcParams['font.sans-serif']** 
   - 设置sans-serif字体族的字体列表
   - 按优先级顺序尝试使用列表中的字体
   
2. **plt.rcParams['axes.unicode_minus']** 
   - 设置为False可以正确显示负号
   - 避免负号显示为方框

### 兼容性 / Compatibility

- ✅ Windows 系统（推荐）
- ✅ macOS（需要手动安装相应中文字体）
- ✅ Linux（需要手动安装相应中文字体）

### macOS/Linux 字体安装 / Font Installation for macOS/Linux

如果在macOS或Linux上运行，可能需要安装中文字体：

**macOS:**
```bash
# SimHei字体通常需要手动下载安装
# 或使用系统内置的其他中文字体，如 STHeiti, PingFang SC
```

**Linux:**
```bash
# Ubuntu/Debian
sudo apt-get install fonts-wqy-microhei fonts-wqy-zenhei

# Fedora/CentOS
sudo yum install wqy-microhei-fonts wqy-zenhei-fonts
```

也可以修改字体列表为系统已安装的字体：
```python
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']  # Linux
plt.rcParams['font.sans-serif'] = ['STHeiti', 'PingFang SC', 'DejaVu Sans']  # macOS
```

## 验证 / Verification

运行任何模型脚本后，检查生成的图形文件（在`figures/`目录下），确认：

1. ✅ 中文字符正常显示
2. ✅ 英文字符正常显示
3. ✅ 负号正常显示
4. ✅ 图形标题、坐标轴标签、图例等所有文本清晰可读

## 常见问题 / FAQ

### Q1: 图形中仍然显示方框？
A: 可能是系统中没有安装SimHei或Microsoft YaHei字体。请：
   1. 检查系统已安装的字体
   2. 修改字体配置列表为系统中存在的中文字体
   3. 或者下载安装相应字体

### Q2: 如何查看系统可用字体？
A: 运行以下Python代码：
```python
import matplotlib.font_manager as fm
fonts = [f.name for f in fm.fontManager.ttflist]
chinese_fonts = [f for f in fonts if 'Hei' in f or 'Yahei' in f or 'Song' in f]
print(chinese_fonts)
```

### Q3: 能否使用其他中文字体？
A: 可以！修改配置为你喜欢的字体：
```python
plt.rcParams['font.sans-serif'] = ['你的字体名称', 'DejaVu Sans']
```

## 联系方式 / Contact

如有问题，请联系开发团队或提交Issue。

---

**最后更新 / Last Updated:** 2025-11-04
**维护者 / Maintainer:** HydroLearn Team
