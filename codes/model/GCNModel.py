import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

# 假设你已经有了 anchor, positive, nagivate 这三个张量

# 构建一个简单的GNN模型
class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, adjacency_matrix):
        adjacency_matrix = adjacency_matrix
        x = torch.relu(self.fc1(x))
        x = torch.matmul(adjacency_matrix, x)  # 使用邻接矩阵进行信息传播
        x = self.fc2(x)
        return x
        
class MLP(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim1=2, hidden_dim2=2, output_dim=1):
        super(MLP, self).__init__()
        # 第一个线性层
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        # 第二个线性层
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        # 输出层，保持与原始Linear层相同的输出维度
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        # 通过第一个线性层，然后是ReLU激活函数
        x = F.relu(self.fc1(x))
        # 通过第二个线性层，然后是ReLU激活函数
        x = F.relu(self.fc2(x))
        # 通过输出层
        x = self.fc3(x)
        return x

# 定义一个神经网络来合并两个子图的信息
class MergeNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MergeNetwork, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 使用平均池化
        # 定义第一个全连接层，输入维度乘以2
        self.fc1 = nn.Linear(input_dim * 2, hidden_dim)
        # 定义第二个全连接层，作为输出层
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, graph1_info, graph2_info):
        # 对每个子图的信息进行平均池化
        pooled_graph1 = self.avg_pool(graph1_info.permute(0, 2, 1)).squeeze(dim=2)
        pooled_graph2 = self.avg_pool(graph2_info.permute(0, 2, 1)).squeeze(dim=2)

        # 将两个子图的池化结果拼接起来
        combined_info = torch.cat((pooled_graph1, pooled_graph2), dim=1)

        # 通过第一个全连接层
        x = F.relu(self.fc1(combined_info))
        # 通过第二个全连接层进行信息融合
        x = self.fc2(x)

        return x
# 定义对比学习的损失函数
class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        euclidean_distance_positive = torch.norm(anchor - positive, p=2, dim=1)
        euclidean_distance_negative = torch.norm(anchor - negative, p=2, dim=1)
        loss_contrastive = torch.mean((euclidean_distance_positive - euclidean_distance_negative + self.margin).clamp(min=0))
        return loss_contrastive