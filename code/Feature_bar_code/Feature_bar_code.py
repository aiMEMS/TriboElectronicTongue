# 导入必要的库 / Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable  # 坐标轴分割工具 / Axis divider tools

# 设置全局字体为Arial，基础字号14 / Set global font to Arial, base size 14
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14  # 基础字号 / Base font size
plt.rcParams['axes.titlesize'] = 14  # 坐标轴标题字号 / Axis title size
plt.rcParams['axes.labelsize'] = 14  # 坐标轴标签字号 / Axis label size

# 使用Matplotlib内置的Plasma调色板 / Use built-in Plasma colormap
cmap = plt.get_cmap('plasma')

# 读取Excel文件 / Read Excel file
df = pd.read_excel("feature_information.xlsx")

# 获取特征列（B-L列，即第2到第12列） / Get feature columns (B-L columns)
feature_columns = df.columns[1:12]
# 获取类型名称列（A列） / Get type names column (A column)
types = df.iloc[:, 0]

# 提取特征数据为numpy数组 / Extract feature data as numpy array
data = df[feature_columns].values

# 创建8x8英寸的画布和轴对象 / Create 8x8 inch figure and axis
fig, ax = plt.subplots(figsize=(8, 8))

# 设置四个边框的线宽和颜色（黑色，1.5磅） / Set border properties (black, 1.5pt width)
for spine in ax.spines.values():
    spine.set_linewidth(1.5)
    spine.set_edgecolor('black')

# 生成y轴位置数组（0到类型数量-1） / Generate y-axis positions array
y_positions = np.arange(len(types))

# 设置特征之间的水平间距 / Set horizontal spacing between features
feature_spacing = 0.105

# 遍历每个特征列进行绘制 / Loop through each feature column for plotting
for i, feature in enumerate(feature_columns):
    # 获取当前特征列的所有数据 / Get current feature's data
    feature_data = data[:, i]
    # 计算矩形宽度（使用对数缩放） / Calculate rectangle widths (logarithmic scaling)
    widths = 0.1 * np.log1p(feature_data * len(types)) / np.log1p(len(types))

    # 计算当前特征的水平位置 / Calculate horizontal position for current feature
    x_position = i * feature_spacing

    # 遍历每个数据点绘制矩形 / Plot rectangles for each data point
    for j, (value, width) in enumerate(zip(feature_data, widths)):
        # 根据数值映射颜色 / Map value to color
        color = cmap(value)
        # 创建矩形对象（去除边框，调整位置居中） / Create rectangle object (no border, centered)
        rect = plt.Rectangle(
            (x_position - width / 2, y_positions[j] - 0.5),  # 左下角坐标 / Bottom-left coordinates
            width,  # 宽度 / Width
            0.8,  # 高度固定为0.8 / Fixed height 0.8
            facecolor=color,
            edgecolor='none'
        )
        ax.add_patch(rect)

# 设置坐标轴范围 / Set axis limits
ax.set_xlim(-0.105, (len(feature_columns) - 1) * feature_spacing + 0.105)
ax.set_ylim(-0.5, len(types) - 0.5)

# 设置x轴刻度位置和标签（旋转45度，右对齐） / Set x-ticks and labels (45° rotation, right-aligned)
ax.set_xticks([i * feature_spacing for i in range(len(feature_columns))])
ax.set_xticklabels(
    feature_columns,
    rotation=45,
    ha='right',  # 水平对齐方式 / Horizontal alignment
    va='top'  # 垂直对齐方式 / Vertical alignment
)

# 调整x轴标签位置（向右微调） / Adjust x-label positions (shift right)
for tick in ax.get_xticklabels():
    tick.set_position((tick.get_position()[0] + 10, tick.get_position()[1]))

# 设置y轴刻度和标签 / Set y-ticks and labels
ax.set_yticks(y_positions)
ax.set_yticklabels(types)
ax.invert_yaxis()  # 反转y轴使标签从上到下排列 / Invert y-axis for top-to-bottom labeling

# 创建颜色条 / Create colorbar
divider = make_axes_locatable(ax)  # 创建可分割的轴 / Create divisible axis
cax = divider.append_axes("right", size="3%", pad=0.1)  # 右侧添加颜色条轴 / Add colorbar axis on right
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))  # 创建可映射对象 / Create mappable object
cbar = plt.colorbar(sm, cax=cax)  # 绘制颜色条 / Draw colorbar

# 设置颜色条标签（旋转270度，设置字体大小） / Set colorbar label (270° rotation, fontsize)
cbar.set_label('Relative feature intensity',
               labelpad=20, rotation=270, fontsize=18)

# 设置颜色条刻度样式 / Set colorbar tick style
cbar.ax.tick_params(width=0.5, labelsize=12)

# 设置颜色条边框样式 / Set colorbar border style
for spine in cax.spines.values():
    spine.set_edgecolor('black')  # 边框颜色 / Border color
    spine.set_linewidth(1.5)  # 边框线宽 / Border width

# 调整布局使元素不重叠 / Adjust layout to prevent overlapping
plt.tight_layout()

# 保存为PDF格式，300dpi分辨率，紧密边框 / Save as PDF with 300dpi and tight bounding box
plt.savefig('Feature_bar_codes_of_14_beverages.pdf', dpi=300, bbox_inches='tight')