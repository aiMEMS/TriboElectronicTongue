# 导入必要的库 / Import required libraries
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import torch
from sklearn.metrics import (f1_score, accuracy_score, roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score)
from sklearn.metrics import recall_score
import joblib
from sklearn.preprocessing import StandardScaler
from scipy import signal


# 模型超参数设置 / Model hyperparameters
num_epochs = 50
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 数据集配置 / Dataset configuration

# class_names = ['35° 5cm', '35° 10cm', '35° 15cm',
#                '45° 5cm', '45° 10cm', '45° 15cm',
#                '55° 5cm', '55° 10cm', '55° 15cm']
# data_dir = r'd:\2413\111DL\5高度角度\Singledatahigh'
# MODEL_NAME = "height&angle"

# class_names = [
#     'Pure water',    
#     'Mineral water', 
#     'Alkaline Soda', 
#     'Glucose Drink', 
#     'Peach Water',   
#     'Lemon Soda',    
#     'Jinro Soju',    
#     'Cocktail ',     
#     'Sprite',        
#     'Vodka',         
#     'Pulse Drink',   
#     'Chinese Baijiu',      
#     'DEW', 
#     'JEW'      
# ]
# data_dir = r'd:\2413\111DL\5高度角度\Unknownliquid'
# MODEL_NAME = "Unknownliquid"  

# data_dir = r'd:\2413\111DL\5高度角度\Unknownliquid-NaCl'
# class_names = ['Pure water', '2.5% NaCl', '5% NaCl','7.5% NaCl', '10% NaCl']
# MODEL_NAME = "NaCl"  

# data_dir = r'd:\2413\111DL\5高度角度\Unknownliquid-Glucose'
# class_names = ['Pure water', '5% Glucose', '10% Glucose','15% Glucose', '20% Glucose']
# MODEL_NAME = "Glucose"  

data_dir = r'd:\2413\111DL\5高度角度\Unknownliquid-Ethanol'
class_names = ['Pure water', '10% Ethanol', '20% Ethanol','30% Ethanol', '40% Ethanol']
MODEL_NAME = "Ethanol"  

best_model_path = f'best_model_{MODEL_NAME}.pth'
print(os.listdir(data_dir))


# 自定义数据集类 / Custom dataset class
class XLSDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        self.file_list = []
        self.transform = transform
        self.scaler = StandardScaler()# 数据标准化器 / Data normalizer

        # 遍历数据文件 / Traverse data files
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for file_name in os.listdir(class_dir):
                if file_name.endswith('.data'):
                    self.file_list.append((os.path.join(class_dir, file_name), class_idx))

        # 数据预处理 / Data preprocessing
        all_data = []
        for file_path, _ in self.file_list:
            data = self.read_data_file(file_path)
            all_data.extend(data)
        self.scaler.fit(np.array(all_data).reshape(-1, 1))
        joblib.dump(self.scaler, 'scaler.pkl')  # 保存标准化参数 / Save scaler parameters
        print("StandardScaler 已保存到 scaler.pkl")
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, idx):
        file_path, label = self.file_list[idx]
        data = self.read_data_file(file_path)
        data = np.nan_to_num(data, nan=0.0)# 处理缺失值 / Handle missing values

        # 数据长度标准化 / Standardize data length
        length = 300
        if len(data) < length:
            data = np.pad(data, (0, length - len(data)), 'constant')
        else:
            data = data[:length]# 截断过长序列 / Truncate long sequences

        # 数据标准化 / Data normalization
        data = self.scaler.transform(data.reshape(-1, 1)).flatten()
        if self.transform:
            data = self.transform(data)
        data = np.ascontiguousarray(data)
        data = torch.FloatTensor(data)
        return data, label
    def read_data_file(self, file_path):
        # 读取数据文件 / Read data file
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





# 数据增强函数 / Data augmentation functions
def add_brownian_noise(data, noise_level_min=0.001, noise_level_max=0.01):
    noise_level = np.random.uniform(noise_level_min, noise_level_max)
    brownian = np.cumsum(np.random.randn(len(data)))  # 生成布朗运动路径 / Generate Brownian path
    brownian = brownian - brownian.mean()  # 中心化处理 / Centering
    brownian = brownian * (data.std() / brownian.std())  # 标准化缩放 / Normalization scaling
    brownian = brownian * noise_level  # 应用噪声水平 / Apply noise level
    noisy_data = data + brownian
    return noisy_data


def add_adaptive_low_frequency_noise(data, base_amplitude=0.01, signal_amplitude=0.1):
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data.copy())  # 转换为Tensor / Convert to tensor
    data_min = data.min()
    data_max = data.max()
    threshold = (data_max + data_min) * 1 / 2  # 计算自适应阈值 / Calculate adaptive threshold

    # 生成低频噪声 / Generate low-frequency noise
    noise = torch.cumsum(torch.randn_like(data), dim=0)  # 累积随机数生成低频噪声 / Cumulative random numbers for low-freq
    noise = noise - noise.mean()  # 去均值 / Remove mean
    noise = noise / (noise.abs().max())  # 归一化 / Normalize

    # 创建概率掩码 / Create probability mask
    prob_mask = torch.rand_like(data) < 0.5  # 50%概率选择区域 / 50% probability selection
    signal_mask = data > threshold  # 高信号区域 / High-signal regions
    final_mask = prob_mask & signal_mask  # 组合掩码 / Combined mask

    # 自适应调整噪声幅度 / Adaptive noise scaling
    adaptive_amplitude = torch.where(final_mask,
                                     signal_amplitude * data,  # 高信号区增强噪声 / Boost noise in high-signal areas
                                     base_amplitude * data)  # 基础噪声 / Base noise
    noise = adaptive_amplitude * noise
    return (data + noise).numpy()



def temporal_shifting(data, max_right=10, max_left=5):
    if np.random.random() < 0.5:  # 50%概率左移 / 50% probability for left shift
        shift = -np.random.randint(0, max_left)
    else:  # 50%概率右移 / 50% probability for right shift
        shift = np.random.randint(0, max_right)

    result = np.zeros_like(data)
    if shift > 0:  # 右移处理 / Right shift handling
        result[shift:] = data[:-shift]
    elif shift < 0:  # 左移处理 / Left shift handling
        result[:shift] = data[-shift:]
    else:  # 无位移 / No shift
        result = data
    return result


def change_amplitude(data, factor_range=(0.9, 1.1)):
    factor = np.random.uniform(*factor_range)  # 随机缩放系数 / Random scaling factor
    return data * factor



class Compose:
    """组合增强方法 / Compose augmentation methods
    Args:
        transforms: 增强方法列表 / List of augmentation methods
    """
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, data):
        """顺序应用数据增强 / Apply augmentations sequentially"""
        for t in self.transforms:
            try:
                data = t(data)
            except Exception as e:
                print(f"Error applying transform {t.__name__}: {str(e)}")
                continue
        return data

# 创建数据增强流程 / Create augmentation pipeline
data_transform = Compose([
    # add_brownian_noise,
    add_adaptive_low_frequency_noise,
    temporal_shifting,
    change_amplitude
])


# 创建数据集 / Create datasets
train_dataset = XLSDataset(os.path.join(data_dir, 'train'), transform=data_transform)  # 训练集含增强 / Train set with augmentation
test_dataset = XLSDataset(os.path.join(data_dir, 'test'))  # 测试集无增强 / Test set without augmentation

# 创建数据加载器 / Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # 训练加载器 / Train loader
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)   # 测试加载器 / Test loader


# 改进的指数积分发放神经元 / Enhanced Exponential Integrate-and-Fire neuron
class MultiStepEIFNode(nn.Module):
    """多步指数积分发放神经元 / Multi-step EIF neuron
    Args:
        threshold: 发放阈值 / Firing threshold
        rest_potential: 静息电位 / Resting potential
        tau: 时间常数 / Time constant
        delta_T: 斜率因子 / Slope factor
        theta_rh: 阈值适应参数 / Threshold adaptation parameter
    """
    def __init__(self, threshold=1.0, rest_potential=0.0, tau=1.0, delta_T=1.0, theta_rh=0.8):
        super(MultiStepEIFNode, self).__init__()
        # 可学习参数 / Learnable parameters
        self.threshold = nn.Parameter(torch.tensor(threshold))
        self.rest_potential = nn.Parameter(torch.tensor(rest_potential))
        self.tau = nn.Parameter(torch.tensor(tau))
        self.delta_T = nn.Parameter(torch.tensor(delta_T))
        self.theta_rh = nn.Parameter(torch.tensor(theta_rh))
    def forward(self, x):
        device = x.device
        self.mem = torch.zeros_like(x, device=device)  # 膜电位初始化 / Membrane potential initialization
        spike = torch.zeros_like(x, device=device)  # 脉冲信号初始化 / Spike signal initialization
        # 膜电位更新公式 / Membrane potential update
        self.mem = self.mem + (1.0 / self.tau) * (
                x - (self.mem - self.rest_potential) +
                self.delta_T * torch.exp((self.mem - self.theta_rh) / self.delta_T)
        )
        # 脉冲发放条件 / Spike firing condition
        spike = (self.mem >= self.threshold).float()
        # 膜电位重置 / Membrane potential reset
        self.mem = (1 - spike) * self.mem + spike * self.rest_potential
        return spike


class ConvSAB(nn.Module):
    """尖峰注意力模块 / Spiking attention block
    Args:
        in_channels: 输入通道数 / Input channels
        out_channels: 输出通道数 / Output channels
        kernel_size: 卷积核大小 / Kernel size
        stride: 步长 / Stride
        padding: 填充 / Padding
        reduction_ratio: 通道压缩比例 / Channel reduction ratio
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, reduction_ratio=16):
        super(ConvSAB, self).__init__()
        # 双卷积层 / Double convolution layers
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.sn1 = MultiStepEIFNode()  # 脉冲神经元 / Spiking neuron
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.sn2 = MultiStepEIFNode()

        # 残差连接 / Residual connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(out_channels)
            )

        # 通道注意力机制 / Channel attention mechanism
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # 全局平均池化 / Global average pooling
            nn.Conv1d(out_channels, out_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels // reduction_ratio, out_channels, kernel_size=1),
            nn.Sigmoid()  # 激活函数 / Activation function
        )

        # 空间注意力机制 / Spatial attention mechanism
        self.spatial_attention = nn.Sequential(
            nn.Conv1d(out_channels, 1, kernel_size=7, padding=3),  # 空间编码 / Spatial encoding
            nn.Sigmoid()
        )

        # 注意力门控参数 / Attention gating parameters
        self.channel_gate = nn.Parameter(torch.zeros(1))  # 通道注意力权重 / Channel attention weight
        self.spatial_gate = nn.Parameter(torch.zeros(1))  # 空间注意力权重 / Spatial attention weight

        # 梯度保存相关 / Gradient saving
        self.features = None  # 特征缓存 / Feature buffer
        self.gradients = None  # 梯度缓存 / Gradient buffer

        self._initialize_weights()  # 参数初始化 / Parameter initialization

    def _initialize_weights(self):
        """参数初始化方法 / Weight initialization method"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # He初始化 / He initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def save_gradient(self, grad):
        """梯度保存钩子 / Gradient saving hook"""
        self.gradients = grad.clone()

    def forward(self, x):
        """前向传播过程 / Forward pass"""
        residual = self.shortcut(x)  # 残差连接 / Residual connection

        # 第一卷积层 / First convolution block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)

        # 第二卷积层 / Second convolution block
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.sn2(out)

        # 通道注意力计算 / Channel attention computation
        channel_att = self.channel_attention(out)
        out_channel = out * channel_att  # 通道注意力应用 / Apply channel attention

        # 空间注意力计算 / Spatial attention computation
        spatial_att = self.spatial_attention(out)
        out_spatial = out * spatial_att  # 空间注意力应用 / Apply spatial attention

        # 双注意力融合 / Dual attention fusion
        out_att = self.channel_gate * out_channel + self.spatial_gate * out_spatial
        self.features = out_att.clone()  # 保存特征供可视化 / Save features for visualization

        # 注册梯度钩子 / Register gradient hook
        if self.training:
            out_att.register_hook(self.save_gradient)

        # 残差连接 / Residual connection
        out = out_att + residual
        return out

    def get_attention_maps(self):
        """获取注意力图 / Get attention maps"""
        return {
            'channel_attention': self.channel_attention(self.features),
            'spatial_attention': self.spatial_attention(self.features)
        }


class TCSN(nn.Module):
    """全电流尖峰网络 / Total-Current Spiking Network（TCSN）
    Args:
        num_classes: 分类类别数 / Number of classes
        input_size: 输入尺寸 / Input size
        dropout_rate: Dropout概率 / Dropout probability
    """
    def __init__(self, num_classes=len(train_dataset.classes), input_size=300, dropout_rate=0.1):
        super(TCSN, self).__init__()
        # 特征提取层 / Feature extraction layers
        self.conv1 = nn.Conv1d(1, 128, kernel_size=16, stride=4, padding=6)  # 第一卷积层 / First conv layer
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(128, 128, kernel_size=16, stride=4, padding=6)  # 第二卷积层 / Second conv layer
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU(inplace=True)

        # 尖峰注意力模块 / Spking attention block
        self.srb1_1 = ConvSAB(128, 128)  # 卷积注意力块 / Convolutional attention block

        # 分类层 / Classification layers
        self.avgpool = nn.AdaptiveAvgPool1d(1)  # 全局平均池化 / Global average pooling
        self.dropout = nn.Dropout(dropout_rate)  # Dropout层 / Dropout layer
        self.fc = nn.Linear(128, num_classes)  # 全连接层 / Fully-connected layer

        # 中间特征存储 / Intermediate feature storage
        self.intermediate_features = {}

    def forward(self, x):
        """前向传播过程 / Forward pass"""
        x = x.unsqueeze(1)  # 增加通道维度 / Add channel dimension

        # 第一卷积块 / First convolution block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # 第二卷积块 / Second convolution block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        # 尖峰注意力模块 / Spking attention block
        x = self.srb1_1(x)
        self.intermediate_features['srb1_1_output'] = x.detach().clone()  # 保存中间特征 / Save intermediate features

        # 分类处理 / Classification processing
        x = self.avgpool(x)
        x = x.squeeze(2)  # 压缩维度 / Squeeze dimension
        x = self.dropout(x)  # 应用Dropout / Apply dropout
        x = self.fc(x)  # 全连接层 / Fully-connected layer
        return x


# 初始化模型 / Initialize model
model = TCSN(num_classes=len(train_dataset.classes), input_size=300)
model = model.to(device)
criterion = nn.CrossEntropyLoss()  # 交叉熵损失 / Cross entropy loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam优化器 / Adam optimizer



import math
# 三阶段学习率调度器 / Three-stage learning rate scheduler
class ThreeStageScheduler:
    """三阶段学习率调度策略 / Three-phase learning rate scheduling
    Args:
        optimizer: 优化器对象 / Optimizer
        max_lr: 最大学习率 / Maximum learning rate
        epochs: 总训练轮数 / Total epochs
        steps_per_epoch: 每轮步数 / Steps per epoch
    """
    def __init__(self, optimizer, max_lr, epochs, steps_per_epoch):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.total_steps = epochs * steps_per_epoch  # 总训练步数 / Total training steps
        self.current_step = 0

        # 阶段划分 / Phase division
        self.stage1_steps = int(self.total_steps * 0.1)  # 第一阶段步数（预热） / Warmup phase steps
        self.stage2_end = self.stage1_steps + int(self.total_steps * 0.3)  # 第二阶段结束步数 / End of stable phase

    def step(self):
        """更新学习率 / Update learning rate"""
        self.current_step += 1

        # 第一阶段：线性预热 / Phase 1: Linear warmup
        if self.current_step <= self.stage1_steps:
            progress = self.current_step / self.stage1_steps
            lr = self.max_lr * (0.05 + 0.95 * progress)  # 从5%到100%线性增长 / Linear increase from 5% to 100%

        # 第二阶段：稳定学习率 / Phase 2: Stable learning rate
        elif self.current_step <= self.stage2_end:
            lr = self.max_lr  # 保持最大学习率 / Maintain max learning rate

        # 第三阶段：余弦退火衰减 / Phase 3: Cosine annealing decay
        else:
            progress = (self.current_step - self.stage2_end) / (self.total_steps - self.stage2_end)
            lr = self.max_lr * (
                        math.cos(progress * math.pi) * 0.499 + 0.5) + self.max_lr / 1000  # 余弦衰减公式 / Cosine decay

        # 更新优化器学习率 / Update optimizer's learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr


# 初始化学习率调度器 / Initialize scheduler
scheduler = ThreeStageScheduler(
    optimizer,
    max_lr=learning_rate,
    epochs=num_epochs,
    steps_per_epoch=len(train_loader)
)

# 训练指标记录 / Training metrics recording
train_accuracy_history = []  # 训练准确率历史 / Training accuracy history
test_accuracy_history = []   # 测试准确率历史 / Test accuracy history
class_accuracy_history = {i: [] for i in range(len(train_dataset.classes))}  # 各类别准确率历史 / Per-class accuracy history
epoch_times = []  # 每轮训练时间 / Epoch duration history
best_accuracy = 0.0


def evaluate_model(model, data_loader):
    """模型评估函数 / Model evaluation function
    Args:
        model: 待评估模型 / Model to evaluate
        data_loader: 数据加载器 / Data loader
    Returns:
        overall_accuracy: 整体准确率 / Overall accuracy
        class_accuracies: 各类别准确率 / Class-wise accuracy
        test_loss: 测试损失 / Test loss
    """
    model.eval()  # 设置模型为评估模式 / Set model to evaluation mode
    correct = 0
    total = 0
    running_loss = 0.0
    class_correct = {i: 0 for i in range(len(train_dataset.classes))}  # 各类正确计数 / Class correct counts
    class_total = {i: 0 for i in range(len(train_dataset.classes))}    # 各类总数 / Class total counts

    with torch.no_grad():  # 禁用梯度计算 / Disable gradient calculation
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)  # 获取预测结果 / Get predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 计算每个类别的准确率 / Calculate per-class accuracy
            for i in range(len(train_dataset.classes)):
                class_mask = (labels == i)
                class_correct[i] += ((predicted == labels) & class_mask).sum().item()
                class_total[i] += class_mask.sum().item()

    # 计算结果 / Compute metrics
    overall_accuracy = correct / total * 100
    test_loss = running_loss / len(data_loader)
    class_accuracies = {i: (class_correct[i] / class_total[i] * 100 if class_total[i] > 0 else 0)
                        for i in range(len(train_dataset.classes))}
    return overall_accuracy, class_accuracies, test_loss


def train_model(model, train_loader, test_loader, num_epochs, criterion, optimizer, scheduler, device, start_epoch=0):
    """模型训练函数 / Model training function
    Args:
        model: 待训练模型 / Model to train
        train_loader: 训练数据加载器 / Training data loader
        test_loader: 测试数据加载器 / Test data loader
        num_epochs: 训练轮数 / Number of epochs
        criterion: 损失函数 / Loss function
        optimizer: 优化器 / Optimizer
        scheduler: 学习率调度器 / Learning rate scheduler
        device: 计算设备 / Computing device
        start_epoch: 起始轮数 / Start epoch
    """
    global best_accuracy, train_accuracy_history, test_accuracy_history, class_accuracy_history
    train_loss_history = []
    test_loss_history = []
    base_path = os.path.splitext(best_model_path)[0]
    start_time = time.time()

    for epoch in range(start_epoch, start_epoch + num_epochs):
        epoch_start_time = time.time()
        model.train() # 设置模型为训练模式 / Set model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        # 训练阶段 / Training phase
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            data.requires_grad_(True)  # 启用梯度计算 / Enable gradient calculation

            optimizer.zero_grad()  # 梯度清零 / Zero gradients
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()  # 反向传播 / Backward pass
            optimizer.step()  # 参数更新 / Update parameters
            scheduler.step()  # 学习率调整 / Adjust learning rate

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # 计算训练指标 / Compute training metrics
        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total * 100
        test_accuracy, class_accuracies, test_loss = evaluate_model(model, test_loader)
        epoch_time = time.time() - epoch_start_time

        # 记录训练指标 / Record training metrics
        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f'Epoch [{epoch + 1}/{start_epoch + num_epochs}], '
              f'Train Loss: {train_loss:.4f}, '
              f'Test Loss: {test_loss:.4f}, '
              f'Train Accuracy: {train_accuracy:.2f}%, '
              f'Test Accuracy: {test_accuracy:.2f}%, '
              f'LR: {current_lr:.6f}')
        print(f'Time: {epoch_time:.2f}s')

        # 更新历史记录 / Update history records
        train_accuracy_history.append(train_accuracy)
        test_accuracy_history.append(test_accuracy)
        for i in range(len(train_dataset.classes)):
            class_accuracy_history[i].append(class_accuracies[i])

        # 保存最佳模型 / Save best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), best_model_path)
            # 保存特征和梯度 / Save features and gradients
            features_dict = {
                'features': model.srb1_1.features if hasattr(model.srb1_1, 'features') else None,
                'gradients': model.srb1_1.gradients if hasattr(model.srb1_1, 'gradients') else None,
                'epoch': epoch,
                'accuracy': test_accuracy
            }
            features_path = f"{base_path}_features.pt"
            torch.save(features_dict, features_path)
            print(f"Best model saved with accuracy: {best_accuracy:.2f}%")
            print(f"Features and gradients saved to: {features_path}")

    # 最终统计与输出 / Final statistics and output
    total_time = time.time() - start_time
    metrics = calculate_metrics(model, test_loader, device, len(train_dataset.classes))
    print_all_metrics(metrics)
    print(f"Best Model Accuracy: {best_accuracy:.2f}%")
    print(f"Total Training Time: {total_time:.2f} seconds")
    print(f"Average Time per Epoch: {total_time / num_epochs:.2f} seconds")
    return epoch + 1

def save_all_accuracies(train_accuracy_history, test_accuracy_history, class_accuracy_history, filename='accuracy.data'):
    """保存准确率数据 / Save accuracy data
    Args:
        train_accuracy_history: 训练准确率历史 / Training accuracy history
        test_accuracy_history: 测试准确率历史 / Test accuracy history
        class_accuracy_history: 类别准确率历史 / Class accuracy history
        filename: 保存文件名 / Save filename
    """
    with open(filename, 'w') as f:
        f.write('Epoch,Train Accuracy,Test Accuracy,Class 0,Class 1,Class 2,Class 3,Class 4,Class 5\n')
        for epoch in range(len(train_accuracy_history)):
            class_acc_str = ','.join(f'{class_accuracy_history[i][epoch]:.2f}' for i in range(len(train_dataset.classes)))
            f.write(f'{epoch+1},{train_accuracy_history[epoch]:.2f},{test_accuracy_history[epoch]:.2f},{class_acc_str}\n')

def calculate_metrics(model, test_loader, device, num_classes):
    """计算评估指标 / Calculate evaluation metrics
    Args:
        model: 待评估模型 / Model to evaluate
        test_loader: 测试数据加载器 / Test data loader
        device: 计算设备 / Computing device
        num_classes: 类别数量 / Number of classes
    Returns:
        metrics: 包含各项指标的字典 / Dictionary containing metrics
    """
    model.eval()
    all_labels = []
    all_predictions = []
    all_probabilities = []

    # 收集预测结果 / Collect predictions
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            outputs = model(data)
            probabilities = F.softmax(outputs, dim=1)  # 计算概率 / Compute probabilities
            predictions = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    # 转换格式 / Convert formats
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_labels_one_hot = label_binarize(all_labels, classes=range(num_classes))

    # 计算各项指标 / Compute metrics
    metrics = {}
    metrics['f1_per_class'] = f1_score(all_labels, all_predictions, average=None)
    metrics['macro_f1'] = f1_score(all_labels, all_predictions, average='macro')
    metrics['micro_f1'] = f1_score(all_labels, all_predictions, average='micro')

    # 计算ROC AUC / Compute ROC AUC
    roc_auc_per_class = []
    for i in range(num_classes):
        try:
            roc_auc = roc_auc_score(all_labels_one_hot[:, i], all_probabilities[:, i])
            roc_auc_per_class.append(roc_auc)
        except:
            roc_auc_per_class.append(0.0)
    metrics['macro_roc_auc'] = np.mean(roc_auc_per_class)

    # 计算micro ROC AUC / Compute micro ROC AUC
    fpr_micro, tpr_micro, _ = roc_curve(all_labels_one_hot.ravel(), all_probabilities.ravel())
    metrics['micro_roc_auc'] = auc(fpr_micro, tpr_micro)

    # 计算平均精确率 / Compute average precision
    metrics['macro_ap'] = average_precision_score(all_labels_one_hot, all_probabilities, average='macro')
    metrics['micro_ap'] = average_precision_score(all_labels_one_hot, all_probabilities, average='micro')

    # 计算其他指标 / Compute other metrics
    metrics['accuracy'] = accuracy_score(all_labels, all_predictions)
    metrics['macro_recall'] = recall_score(all_labels, all_predictions, average='macro')
    metrics['micro_recall'] = recall_score(all_labels, all_predictions, average='micro')

    # 保存原始数据 / Save raw data
    metrics['raw_data'] = {
        'all_labels': all_labels,
        'all_predictions': all_predictions,
        'all_probabilities': all_probabilities,
        'all_labels_one_hot': all_labels_one_hot
    }

    return metrics

def print_all_metrics(metrics):
    """打印评估指标 / Print evaluation metrics"""
    print("\n=== Model Performance Metrics ===")
    print("--------------------------------")
    print(f"\nF1 Scores:")
    print(f"Macro-F1: {metrics['macro_f1'] * 100:.2f}%")
    print(f"Micro-F1: {metrics['micro_f1'] * 100:.2f}%")
    print(f"\nROC-AUC Scores:")
    print(f"Macro-AUC: {metrics['macro_roc_auc'] * 100:.2f}%")
    print(f"Micro-AUC: {metrics['micro_roc_auc'] * 100:.2f}%")
    print(f"\nAverage Precision Scores:")
    print(f"Macro-AP: {metrics['macro_ap'] * 100:.2f}%")
    print(f"Micro-AP: {metrics['micro_ap'] * 100:.2f}%")
    print(f"\nRecall Scores:")
    print(f"Macro-Recall: {metrics['macro_recall'] * 100:.2f}%")
    print(f"Micro-Recall: {metrics['micro_recall'] * 100:.2f}%")
    print("\nPer-Class F1 Scores:")
    for i, f1 in enumerate(metrics['f1_per_class']):
        print(f"Class {i}: {f1 * 100:.2f}%")


def plot_accuracy_trends(train_accuracy_history, test_accuracy_history, class_accuracy_history):
    """绘制准确率趋势图 / Plot accuracy trends
       Args:
           train_accuracy_history: 训练准确率历史 / Training accuracy history
           test_accuracy_history: 测试准确率历史 / Test accuracy history
           class_accuracy_history: 类别准确率历史 / Class accuracy history
    """
    plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']
    plt.rcParams['font.size'] = 14
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    ax.plot(range(1, len(train_accuracy_history) + 1), train_accuracy_history, marker='o', markersize=3, linewidth=1.5, color='#1976D2', label='Train Accuracy', markerfacecolor='#1976D2', markeredgewidth=1.5)
    ax.plot(range(1, len(test_accuracy_history) + 1), test_accuracy_history, marker='s', markersize=3, linewidth=1.5, color='#D32F2F', label='Test Accuracy', markerfacecolor='#D32F2F', markeredgewidth=1.5)
    ax.set_title('Training and Testing Accuracy Trend',fontsize=16, fontweight='bold',fontfamily='Times New Roman',pad=22)
    ax.set_xlabel('Epoch',fontsize=20,fontfamily='Times New Roman',labelpad=10)
    ax.set_ylabel('Accuracy (%)',fontsize=20,fontfamily='Times New Roman',labelpad=10)
    ax.set_ylim(0, 100)
    epochs = len(train_accuracy_history)
    if epochs <= 10:
        ticks = list(range(1, epochs + 1))
    else:
        interval = ((epochs // 10) + 9) // 10 * 10
        ticks = [1] 
        current = interval
        while current < epochs:
            ticks.append(current)
            current += interval
        if epochs % 10 <= 5:
            last_tick = (epochs // 10) * 10
        else:
            last_tick = ((epochs + 9) // 10) * 10
        if last_tick not in ticks and last_tick <= epochs:
            ticks.append(last_tick)
        if epochs not in ticks:
            ticks.append(epochs)

    plt.xticks(ticks)
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.legend(fontsize=12, frameon=True, edgecolor='black', fancybox=False, loc='lower right') 
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.tick_params(axis='both', which='major', labelsize=16, pad=10)  
    plt.tight_layout()
    plt.savefig('train_test_accuracy_trend.png',dpi=300,bbox_inches='tight', pad_inches=0.1)
    plt.close()
    plt.figure(figsize=(15, 10))
    for i in range(len(train_dataset.classes)):
        plt.plot(range(1, len(class_accuracy_history[i]) + 1), class_accuracy_history[i], marker='o', label=f'Class {i}')
    plt.title('Class-Specific Accuracy Trends')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True)
    plt.savefig('class_specific_accuracy_trends.png')
    plt.close()
    
best_accuracy = 0.0
total_epochs = 0

CONTROL_TRAINING = False  
if CONTROL_TRAINING:
    while True:
        total_epochs = train_model(model, train_loader, test_loader, num_epochs, criterion, optimizer, scheduler, device, start_epoch=total_epochs)
        save_all_accuracies(train_accuracy_history, test_accuracy_history, class_accuracy_history)
        plot_accuracy_trends(train_accuracy_history, test_accuracy_history, class_accuracy_history)
        user_input = input("是否继续训练？(y/n): ").lower()
        if user_input != 'y':
            break
        num_epochs = int(input("请输入继续训练的轮数: "))
else:
    total_epochs = train_model(model, train_loader, test_loader, num_epochs, criterion, optimizer, scheduler, device)
    save_all_accuracies(train_accuracy_history, test_accuracy_history, class_accuracy_history)
    plot_accuracy_trends(train_accuracy_history, test_accuracy_history, class_accuracy_history)



# 加载最佳模型 / Load best model
model.load_state_dict(torch.load(best_model_path))
model.eval()
all_labels = []
all_predictions = []
all_probs = []
with torch.no_grad():
    for data, labels in test_loader:
        data, labels = data.to(device), labels.to(device)
        outputs = model(data)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs.data, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())


def plot_confusion_matrix(all_labels, all_predictions, class_names, save_prefix=''):
    """绘制混淆矩阵 / Plot confusion matrix
    Args:
        all_labels: 真实标签 / True labels
        all_predictions: 预测标签 / Predicted labels
        class_names: 类别名称 / Class names
        save_prefix: 保存文件名前缀 / Save filename prefix
    """
    num_classes = len(class_names)
    # 动态尺寸设置 / Dynamic size settings
    size_map = {
        4:  (3.6, 3.15),
        5:  (3.87, 3.42),
        6:  (4.14, 3.69),
        7:  (4.41, 3.96),
        8:  (4.68, 4.23),
        9:  (4.95, 4.5),
        10: (5.22, 4.77),
        11: (5.49, 5.04),
        12: (5.76, 5.31),
        13: (6.03, 5.58),
        14: (6.201, 5.76),
        15: (6.3, 5.85)
    }
    # 动态字体设置 / Dynamic font settings
    font_size_map = {
        4:  {'base': 10, 'annot': 9, 'title': 12},
        5:  {'base': 10, 'annot': 9, 'title': 12},
        6:  {'base': 9, 'annot': 8, 'title': 11},
        7:  {'base': 9, 'annot': 8, 'title': 11},
        8:  {'base': 9, 'annot': 8, 'title': 11},
        9:  {'base': 9, 'annot': 8, 'title': 11},
        10: {'base': 8, 'annot': 7, 'title': 10},
        11: {'base': 8, 'annot': 7, 'title': 10},
        12: {'base': 8, 'annot': 7, 'title': 10},
        13: {'base': 8, 'annot': 7, 'title': 10},
        14: {'base': 8, 'annot': 7, 'title': 10},
        15: {'base': 7, 'annot': 6, 'title': 9}
    }

    cm = confusion_matrix(all_labels, all_predictions)
    cm_percentages = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # 创建注释文本 / Create annotation text
    annot = np.empty_like(cm, dtype=object)
    for i in range(len(cm)):
        for j in range(len(cm)):
            if cm[i,j] == 0:
                # annot[i, j] = '0'
                annot[i, j] = ''
            else:
                # annot[i, j] = f'{cm[i,j]}\n({cm_percentages[i,j]:.1f}%)'
                annot[i, j] = f'{cm[i,j]}'

    # 设置绘图参数 / Set plotting parameters
    font_sizes = font_size_map[num_classes]
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': font_sizes['base'],
        'axes.labelsize': font_sizes['base'] + 1,
        'axes.titlesize': font_sizes['title']+2,
        'axes.titleweight': 'bold', 
        'xtick.labelsize': font_sizes['base'],
        'ytick.labelsize': font_sizes['base']
    })

    # 创建热力图 / Create heatmap
    fig, ax = plt.subplots(figsize=size_map[num_classes], dpi=600)
    avg_accuracy = np.mean(cm_percentages.diagonal())
    sns.heatmap(cm, annot=annot, fmt='', cmap='Greens', xticklabels=class_names, yticklabels=class_names, ax=ax, annot_kws={'size': font_sizes['annot'], 'weight': 'normal', 'family': 'Times New Roman'})

    # 设置图表格式 / Format plot
    ax.set_title(f'Avg Accuracy: {avg_accuracy:.2f}%', pad=8, fontweight='normal')
    ax.set_xlabel('Predicted Label', labelpad=10)
    ax.set_ylabel('True Label', labelpad=10)
    ax.tick_params(axis='both', which='major', pad=8)

    # 旋转x轴标签 / Rotate x-axis labels
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')

    # 美化边框 / Beautify borders
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    plt.tight_layout()
    save_name = f'confusion_matrix_{save_prefix}{num_classes}classes'
    plt.savefig(f'{save_name}.pdf', dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f'{save_name}.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"\nClassification Report ({num_classes} classes):")
    print(classification_report(all_labels, all_predictions, target_names=class_names))
plot_confusion_matrix(all_labels, all_predictions, class_names, save_prefix='method1_')





from sklearn.metrics import recall_score
def plot_metrics(model, test_loader, device, num_classes, class_names=None):
    """绘制综合评估指标图 / Plot comprehensive evaluation metrics
    Args:
        model: 已训练模型 / Trained model
        test_loader: 测试数据加载器 / Test data loader
        device: 计算设备 / Computing device
        num_classes: 类别数量 / Number of classes
        class_names: 类别名称 / Class names
    """
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.titlesize': 11
    })
    model.eval()
    all_labels = []
    all_predictions = []
    all_probabilities = []
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_labels_one_hot = label_binarize(all_labels, classes=range(num_classes))
    f1_per_class = f1_score(all_labels, all_predictions, average=None)
    macro_f1 = f1_score(all_labels, all_predictions, average='macro')
    micro_f1 = f1_score(all_labels, all_predictions, average='micro')

    plt.figure(figsize=(6.89, 5.17), dpi=600)
    x = np.arange(num_classes)
    plt.bar(x, f1_per_class, color='#4477AA', alpha=0.7)
    plt.axhline(y=macro_f1, color='#EE6677', linestyle='--', 
                label=f'Macro-F1: {macro_f1:.5f}')
    plt.axhline(y=micro_f1, color='#228833', linestyle='--', 
                label=f'Micro-F1: {micro_f1:.5f}')
    plt.title('F1 Scores Across Classes')
    plt.xlabel('Class')
    plt.ylabel('F1 Score')
    plt.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    plt.legend(frameon=True, edgecolor='black', fancybox=False)
    for spine in plt.gca().spines.values():
        spine.set_linewidth(0.5)
    plt.tight_layout()
    plt.savefig('f1_scores.pdf', dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.savefig('f1_scores.png', dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.close()


    plt.figure(figsize=(6.89, 5.17), dpi=600)
    all_fpr = np.unique(np.concatenate([roc_curve(all_labels_one_hot[:, i], 
                        all_probabilities[:, i])[0] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(all_labels_one_hot[:, i], all_probabilities[:, i])
        mean_tpr += np.interp(all_fpr, fpr, tpr)
    mean_tpr /= num_classes
    
    fpr_micro, tpr_micro, _ = roc_curve(all_labels_one_hot.ravel(), 
                                       all_probabilities.ravel())
    macro_roc_auc = auc(all_fpr, mean_tpr)
    micro_roc_auc = auc(fpr_micro, tpr_micro)

    plt.plot(all_fpr, mean_tpr, '#4477AA', lw=1.5,
            label=f'Macro-average ROC (AUC = {macro_roc_auc:.5f})')
    plt.plot(fpr_micro, tpr_micro, '#EE6677', lw=1.5,
            label=f'Micro-average ROC (AUC = {micro_roc_auc:.5f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=0.8)
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right", frameon=True, edgecolor='black', fancybox=False)
    plt.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    for spine in plt.gca().spines.values():
        spine.set_linewidth(0.5)
    plt.tight_layout()
    plt.savefig('roc_curves.pdf', dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.savefig('roc_curves.png', dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.close()


    plt.figure(figsize=(6.89, 5.17), dpi=600)
    precision = dict()
    recall = dict()
    for i in range(num_classes):
        precision[i], recall[i], _ = precision_recall_curve(
            all_labels_one_hot[:, i], all_probabilities[:, i])
    mean_recall = np.linspace(0, 1, 100)
    mean_precision = np.zeros_like(mean_recall)
    for i in range(num_classes):
        mean_precision += np.interp(mean_recall, recall[i][::-1], precision[i][::-1])
    mean_precision /= num_classes
    precision_micro, recall_micro, _ = precision_recall_curve(
        all_labels_one_hot.ravel(), all_probabilities.ravel())
    

    macro_ap = average_precision_score(all_labels_one_hot, all_probabilities, average='macro')
    micro_ap = average_precision_score(all_labels_one_hot, all_probabilities, average='micro')
    plt.plot(mean_recall, mean_precision, '#4477AA', lw=1.5,
            label=f'Macro-average PR (AP = {macro_ap:.5f})')
    plt.plot(recall_micro, precision_micro, '#EE6677', lw=1.5,
            label=f'Micro-average PR (AP = {micro_ap:.5f})')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower left", frameon=True, edgecolor='black', fancybox=False)
    plt.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    for spine in plt.gca().spines.values():
        spine.set_linewidth(0.5)
    plt.tight_layout()
    plt.savefig('precision_recall_curves.pdf', dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.savefig('precision_recall_curves.png', dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    return {
        'f1_per_class': f1_per_class,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'macro_roc_auc': macro_roc_auc,
        'micro_roc_auc': micro_roc_auc,
        'macro_ap': macro_ap,
        'micro_ap': micro_ap,
    }
metrics = plot_metrics(model, test_loader, device, len(train_dataset.classes), class_names)