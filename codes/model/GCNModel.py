import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn


class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, adjacency_matrix):
        adjacency_matrix = adjacency_matrix
        x = torch.relu(self.fc1(x))
        x = torch.matmul(adjacency_matrix, x) 
        x = self.fc2(x)
        return x
        
class MLP(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim1=2, output_dim=1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MergeNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MergeNetwork, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  
        self.fc1 = nn.Linear(input_dim * 2, output_dim)


    def forward(self, graph1_info, graph2_info):
        pooled_graph1 = self.avg_pool(graph1_info.permute(0, 2, 1)).squeeze(dim=2)
        pooled_graph2 = self.avg_pool(graph2_info.permute(0, 2, 1)).squeeze(dim=2)

        combined_info = torch.cat((pooled_graph1, pooled_graph2), dim=1)

        x = F.relu(self.fc1(combined_info))

        return x
class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        euclidean_distance_positive = torch.norm(anchor - positive, p=2, dim=1)
        euclidean_distance_negative = torch.norm(anchor - negative, p=2, dim=1)
        loss_contrastive = torch.mean((euclidean_distance_positive - euclidean_distance_negative + self.margin).clamp(min=0))
        return loss_contrastive