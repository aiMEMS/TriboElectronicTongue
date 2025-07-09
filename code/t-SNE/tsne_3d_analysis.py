import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns
import pandas as pd
import time
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# 数据集路径
data_dir = 'Unknown liquid'
class_names = [
    'Pure water',    # 水
    'mineral water', # 矿泉水
    'Alkaline Soda', # 碱性苏打
    'Glucose Drink', # 葡萄糖饮料
    'Peach Water',   # 蜜桃水
    'Lemon Soda',    # 柠檬苏打水
    'Jinro Soju',    # 真露烧酒
    'Cocktail ',     # RIO鸡尾酒
    'Sprite',        # 雪碧
    'Vodka',         # 小鸟伏特加
    'Pulse Drink',   # 脉动
    'Chinese Baijiu',
    'Dongpeng Electrolyte Water', # 补水啦
    'Jianlibao Electrolyte Water'      # 健力宝
]

class XLSDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        self.file_list = []
        self.transform = transform
        self.scaler = StandardScaler()
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for file_name in os.listdir(class_dir):
                if file_name.endswith('.data'):
                    self.file_list.append((os.path.join(class_dir, file_name), class_idx))
        all_data = []
        for file_path, _ in self.file_list:
            data = self.read_data_file(file_path)
            all_data.extend(data)
        self.scaler.fit(np.array(all_data).reshape(-1, 1))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path, label = self.file_list[idx]
        data = self.read_data_file(file_path)
        data = np.nan_to_num(data, nan=0.0)
        length = 300
        if len(data) < length:
            data = np.pad(data, (0, length - len(data)), 'constant')
        else:
            data = data[:length]
        data = self.scaler.transform(data.reshape(-1, 1)).flatten()
        if self.transform:
            data = self.transform(data)
        data = np.ascontiguousarray(data)
        data = torch.FloatTensor(data)
        return data, label

    def read_data_file(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            data_start_index = lines.index('***End_of_Header***\n') + 1
            data = []
            for line in lines[data_start_index:]:
                try:
                    voltage = float(line.strip().split('\t')[1])
                    data.append(voltage)
                except ValueError:
                    continue
        return np.array(data)

class MultiStepEIFNode(nn.Module):
    def __init__(self, threshold=1.0, rest_potential=0.0, tau=1.0, delta_T=1.0, theta_rh=0.8):
        super(MultiStepEIFNode, self).__init__()
        self.threshold = nn.Parameter(torch.tensor(threshold))
        self.rest_potential = nn.Parameter(torch.tensor(rest_potential))
        self.tau = nn.Parameter(torch.tensor(tau))
        self.delta_T = nn.Parameter(torch.tensor(delta_T))
        self.theta_rh = nn.Parameter(torch.tensor(theta_rh))
    def forward(self, x):
        device = x.device
        self.mem = torch.zeros_like(x, device=device)
        spike = torch.zeros_like(x, device=device)

        self.mem = self.mem + (1.0 / self.tau) * (
                x - (self.mem - self.rest_potential) +
                self.delta_T * torch.exp((self.mem - self.theta_rh) / self.delta_T)
        )
        spike = (self.mem >= self.threshold).float()
        self.mem = (1 - spike) * self.mem + spike * self.rest_potential
        return spike


class ConvSAB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, reduction_ratio=16):
        super(ConvSAB, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.sn1 = MultiStepEIFNode()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.sn2 = MultiStepEIFNode()
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(out_channels)
            )
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(out_channels, out_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels // reduction_ratio, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv1d(out_channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        self.channel_gate = nn.Parameter(torch.zeros(1))
        self.spatial_gate = nn.Parameter(torch.zeros(1))
        self.features = None
        self.gradients = None
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def save_gradient(self, grad):
        self.gradients = grad.clone()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.sn2(out)
        channel_att = self.channel_attention(out)
        out_channel = out * channel_att
        spatial_att = self.spatial_attention(out)
        out_spatial = out * spatial_att
        out_att = self.channel_gate * out_channel + self.spatial_gate * out_spatial
        self.features = out_att.clone()
        if self.training:
            out_att.register_hook(self.save_gradient)
        out = out_att + residual
        return out

    def get_attention_maps(self):
        return {
            'channel_attention': self.channel_attention(self.features),
            'spatial_attention': self.spatial_attention(self.features)
        }

class TCSN(nn.Module):
    def __init__(self, num_classes=14,input_size=300,dropout_rate=0.1):
        super(TCSN, self).__init__()
        self.conv1 = nn.Conv1d(1, 128, kernel_size=16, stride=4, padding=6)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=16, stride=4, padding=6)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.srb1_1 = ConvSAB(128, 128)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(128, num_classes)
        self.intermediate_features = {}

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.srb1_1(x)
        self.intermediate_features['srb1_1_output'] = x.detach().clone()
        x = self.avgpool(x)
        x = x.squeeze(2)
        x = self.dropout(x)
        x = self.fc(x)
        return x



# 加载预训练模型 / Load pre-trained model
model = TCSN(num_classes=14, input_size=300)
model.load_state_dict(torch.load('E:/111DL/12未知溶液tsne/best_model_Unknownliquid.pth', weights_only=True))
model.eval()  # 设置为评估模式 / Set to evaluation mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建测试数据集和数据加载器 / Create test dataset and data loader
test_dataset = XLSDataset(os.path.join(data_dir, 'Test'))
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
# 提取最后一层隐藏层的特征
features = []
labels = []

with torch.no_grad():  # 禁用梯度计算 / Disable gradient calculation
    for data, label in test_loader:
        data = data.to(device)
        # 前向传播获取模型输出（分类层前的特征） / Forward pass to get model outputs (features before classification layer)
        output = model(data)
        features.append(output.cpu().numpy())  # 转移数据到CPU并转numpy数组 / Transfer data to CPU and convert to numpy
        labels.append(label.numpy())  # 收集真实标签 / Collect true labels

# 合并批次数据 / Concatenate batch data
features = np.vstack(features)      # 垂直堆叠特征 / Stack features vertically (n_samples, feature_dim)
labels = np.concatenate(labels)     # 拼接标签 / Concatenate labels


# 使用t-SNE降维 / Apply t-SNE dimensionality reduction
tsne = TSNE(
    n_components=3,      # 输出3维 / Output 3 dimensions
    perplexity=75,      # 困惑度 / Perplexity
    learning_rate=100,  # 学习率 / Learning rate
    max_iter=5000       # 最大优化迭代次数 / Maximum optimization iterations
)
# 执行降维 / Perform dimensionality reduction
features_tsne = tsne.fit_transform(features)

# 数据归一化到[-1, 1]范围 / Normalize data to [-1, 1] range
min_val = features_tsne.min(axis=0)  # 各维度最小值 / Min values per dimension
max_val = features_tsne.max(axis=0)  # 各维度最大值 / Max values per dimension

# 归一化公式：将数据线性映射到[-1,1] / Normalization formula: linear mapping to [-1,1]
features_tsne_normalized = 2 * (features_tsne - min_val) / (max_val - min_val) - 1



# ==================== 可视化设置 Visualization Settings ====================
# 定义14种对比色 / Define 14 distinct colors
custom_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD',
                 '#D4A5A5', '#9B59B6', '#3498DB', '#2ECC71', '#FFA07A',
                 '#8A2BE2', '#FF1493', '#00CED1', '#FFD700']

# 创建颜色映射
num_classes = len(np.unique(labels))
custom_cmap = mcolors.ListedColormap(custom_colors[:num_classes])

# 定义14种标记符号 / Define 14 marker symbols
markers = ['o', 's', 'D', 'p', 'h', 'H', '*', 'X', 'P', 'd', 'v', '^', '<', '>']

# 创建3D画布 / Create 3D canvas
fig = plt.figure(figsize=(6, 4))  # 6英寸宽，4英寸高 / 6 inches width, 4 inches height
ax = fig.add_subplot(111, projection='3d')  # 添加3D子图 / Add 3D subplot


# 逐类别绘制数据点 / Plot data points per class
for i in range(len(class_names)):
    mask = labels == i  # 当前类别的布尔掩码 / Boolean mask for current class
    class_points = features_tsne_normalized[mask]  # 筛选当前类别的数据点 / Filter current class points
    # 绘制3D散点图 / Plot 3D scatter
    ax.scatter(
        class_points[:, 0],  # X轴：t-SNE第一维度 / X-axis: t-SNE 1st dimension
        class_points[:, 1],  # Y轴：t-SNE第二维度 / Y-axis: t-SNE 2nd dimension
        class_points[:, 2],  # Z轴：t-SNE第三维度 / Z-axis: t-SNE 3rd dimension
        c=[custom_colors[i]],  # 颜色映射 / Color mapping
        marker=markers[i],  # 标记形状 / Marker symbol
        alpha=0.8,  # 透明度（80%） / Transparency (80%)
        s=2,  # 点大小（2像素） / Marker size (2 pixels)
        label=f'Class {i}'  # 图例标签 / Legend label
    )


# 设置坐标轴背景 / Set axis background
ax.xaxis.pane.fill = True  # X轴背景填充 / X-axis background fill
ax.yaxis.pane.fill = True  # Y轴背景填充 / Y-axis background fill
ax.zaxis.pane.fill = True  # Z轴背景填充 / Z-axis background fill
ax.xaxis.pane.set_color((1.0, 1.0, 1.0, 1.0))
ax.yaxis.pane.set_color((1.0, 1.0, 1.0, 1.0))
ax.zaxis.pane.set_color((1.0, 1.0, 1.0, 1.0))

# 设置网格线样式 / Configure grid line style
ax.xaxis._axinfo["grid"].update({
    "linestyle": '--',  # 虚线 / Dashed line
    "alpha": 0.3        # 透明度30% / 30% transparency
})
ax.xaxis._axinfo["grid"].update({"linestyle": '--', "alpha": 0.3})
ax.yaxis._axinfo["grid"].update({"linestyle": '--', "alpha": 0.3})
ax.zaxis._axinfo["grid"].update({"linestyle": '--', "alpha": 0.3})

# 创建图例元素 / Create legend elements
legend_elements = [
    # 每个类别对应图例项 / Legend item for each class
    Line2D([0], [0],
           marker=markers[i],        # 标记形状 / Marker symbol
           color='w',               # 线条颜色（白色） / Line color (white)
           markerfacecolor=custom_colors[i],  # 标记填充色 / Marker fill color
           label=class_names[i],     # 类别名称 / Class name
           markersize=8             # 图例标记大小 / Legend marker size
    ) for i in range(len(class_names))
]

# 设置全局字体 / Configure global font
plt.rcParams['font.family'] = 'Arial'  # 使用Arial字体 / Use Arial font

# 配置坐标轴标签 / Configure axis labels
ax.set_xlabel('')  # 隐藏X轴标签 / Hide X-axis label
ax.set_ylabel('')  # 隐藏Y轴标签 / Hide Y-axis label
ax.set_zlabel('')  # 隐藏Z轴标签 / Hide Z-axis label

# 设置刻度标签字体 / Set tick label font
ax.tick_params(axis='x', which='major', labelsize=8)
ax.tick_params(axis='y', which='major', labelsize=8)
ax.tick_params(axis='z', which='major', labelsize=8)
for text in ax.get_xticklabels():
    text.set_fontname('Arial')
for text in ax.get_yticklabels():
    text.set_fontname('Arial')
for text in ax.get_zticklabels():
    text.set_fontname('Arial')

# 添加图例 / Add legend
ax.legend(
    handles=legend_elements,          # 图例元素 / Legend items
    loc='center left',               # 左侧居中定位 / Left-center position
    bbox_to_anchor=(1.05, 0.5),      # 图例框外定位 / Position outside plot
    title='Signal Types',            # 图例标题 / Legend title
    title_fontsize=4,                # 标题字体大小4pt / Title font size 4pt
    fontsize=3,                      # 标签字体大小3pt / Label font size 3pt
    frameon=True,                    # 显示图例边框 / Show legend frame
    edgecolor='black',               # 边框颜色黑色 / Frame color black
    fancybox=False,                  # 禁用圆角边框 / Disable rounded corners
    prop={'family': 'Arial'}         # 图例字体 / Legend font
)

# 设置坐标轴范围 / Set axis limits
ax.set_xlim([-1, 1])  # X轴范围-1到1 / X-axis range -1 to 1
ax.set_ylim([-1, 1])  # Y轴范围-1到1 / Y-axis range -1 to 1
ax.set_zlim([-1, 1])  # Z轴范围-1到1 / Z-axis range -1 to 1

# 配置刻度位置 / Configure tick positions
ax.set_xticks([-1, -0.5, 0, 0.5, 1])  # X轴刻度位置 / X-axis ticks
ax.set_yticks([-1, -0.5, 0, 0.5, 1])  # Y轴刻度位置 / Y-axis ticks
ax.set_zticks([-1, -0.5, 0, 0.5, 1])  # Z轴刻度位置 / Z-axis ticks


# 隐藏首尾刻度标签 / Hide first and last tick labels
ax.set_xticklabels(['', '-0.5', '0', '0.5', ''])  # X轴标签 / X labels
ax.set_yticklabels(['', '-0.5', '0', '0.5', ''])  # Y轴标签 / Y labels
ax.set_zticklabels(['', '-0.5', '0', '0.5', ''])  # Z轴标签 / Z labels


# 调整3D视角 / Adjust 3D view angle
ax.view_init(elev=20,   # 仰角20度 / Elevation angle 20 degrees
            azim=45)   # 方位角45度 / Azimuth angle 45 degrees


# 保持物理尺寸调整布局 / Adjust layout while maintaining physical size
plt.gcf().set_size_inches(6, 4)  # 强制画布尺寸 / Enforce canvas size
plt.tight_layout()  # 紧凑布局 / Compact layout


# 保存矢量图 / Save vector graphic
plt.savefig(
    'tsne_visualization.pdf',  # 文件名 / Filename
    facecolor='white',         # 背景白色 / White background
    edgecolor='none',          # 无边框 / No border
    pad_inches=0.1           # 内边距0.1英寸 / 0.1 inch padding
)

# 显示图像 / Display plot
plt.show()


