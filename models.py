"""
Define residential location choice models
"""
# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class MNL_Choice(nn.Module):
    """
    Multinomial Logit model
    """
    def __init__(self, num_features, num_comm):
        super(MNL_Choice, self).__init__()
        self.beta = nn.Parameter(torch.randn(num_features), requires_grad=True)
        self.asc = nn.Parameter(torch.zeros(num_comm - 1), requires_grad=True)

    def forward(self, data):
        # data: (batch_size, num_comm, num_features)
        x = data @ self.beta  # (batch_size, num_comm)
        x[:, 1:] += self.asc
        return F.log_softmax(x, dim=-1)  # (batch_size, num_comm)


class SCL_Choice(nn.Module):
    def __init__(self, num_features, edge_index):
        """
        Spatially correlated logit model
        Note the edge_index in (2, num_edge) must be sorted by the first row.
        """
        super(SCL_Choice, self).__init__()

        num_comm = edge_index.max().item() + 1
        num_edge = edge_index.size(1)
        allocation_matrix = torch.zeros(num_comm, num_comm, dtype=torch.float32)
        allocation_matrix[edge_index[0], edge_index[1]] = 1
        allocation_matrix = allocation_matrix / allocation_matrix.sum(
            dim=1, keepdim=True
        )
        allocation_matrix.requires_grad = False
        mapping_matrix = torch.zeros(
            num_edge, num_comm, dtype=torch.float32, requires_grad=False
        )
        mapping_matrix[torch.arange(num_edge), edge_index[0]] = 1

        # Constant buffers
        self.register_buffer("ei", edge_index)
        self.register_buffer("am", allocation_matrix)
        self.register_buffer("mm", mapping_matrix)

        # Define the dissimilarity parameter $\mu$ with the bounds (0, 1]
        self.mu_raw = nn.Parameter(torch.tensor(5.0), requires_grad=True)
        self.fc1 = nn.Linear(num_features, 1, bias=False)
        # Define alternative-specific constants
        self.asc = nn.Parameter(torch.zeros(num_comm - 1), requires_grad=True)

    def forward(self, comm_data):
        # Batch over the community graphs
        # (bs, num_comm, num_comm_features) -> (bs, num_comm)
        x = self.fc1(comm_data).squeeze(-1)  # (batch_size, num_comm)
        x[:, 1:] += self.asc
        mu = torch.sigmoid(self.mu_raw)

        n_start = (self.am[self.ei[0], self.ei[1]] * torch.exp(x[:, self.ei[0]])) ** (
            1 / mu
        )  # (batch_size, num_edge)

        n_end = (self.am[self.ei[1], self.ei[0]] * torch.exp(x[:, self.ei[1]])) ** (
            1 / mu
        )  # (batch_size, num_edge)

        exp_utility = (
            n_start * (n_start + n_end) ** (mu - 1)
        ) @ self.mm  # (batch_size, num_comm)

        p = exp_utility / exp_utility.sum(dim=1, keepdim=True)  # (batch_size, num_comm)
        return torch.log(p)  # (batch_size, num_comm)


class GAT(nn.Module):
    def __init__(self, num_features, num_hidden, dropout=0.1, heads=1, n_layer=2, **kwargs):
        super(GAT, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(num_features, num_hidden // heads, heads=heads, **kwargs))
        for _ in range(n_layer - 1):
            self.convs.append(
                GATConv(num_hidden, num_hidden // heads, heads=heads, **kwargs)
            )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
            x = self.drop(x)
        return x

    # def forward_with_weights(self, x, edge_index):
    #     x = self.conv1(x, edge_index).relu()
    #     x = self.drop(x)
    #     x, weights = self.conv2(x, edge_index, return_attention_weights=True)
    #     return x, weights


class GNNChoiceModel(nn.Module):
    def __init__(
        self, num_comm_features, num_hidden, edge_index, dropout=0.1, heads=1, **kwargs
    ):
        """
        GNN-based residential location choice model
        """
        super(GNNChoiceModel, self).__init__()
        self.gnn = GAT(num_comm_features, num_hidden, dropout, heads, **kwargs)
        self.register_buffer("edge_index", edge_index)  # Constant buffer
        self.num_hidden = num_hidden
        self.out_layer = nn.Linear(num_hidden, 1)

    def forward(self, comm_data):
        """
        hh_data: (bs, num_hh_features)
        comm_data: (bs, num_comm, num_comm_features)
        edge_index: (bs, 2, num_edges)
        """
        # Batch over the community graphs
        # (bs, num_comm, num_comm_features) -> (bs, num_comm, num_hidden)
        comm_embedding = torch.zeros(
            comm_data.size(0), comm_data.size(1), self.num_hidden
        ).to(comm_data.device)
        for i in range(comm_data.size(0)):
            comm_embedding[i] = self.gnn(comm_data[i], self.edge_index)

        out = self.out_layer(comm_embedding).squeeze()  # (bs, num_comm)

        return F.log_softmax(out, dim=-1)  # (batch_size, num_comm)


# class GAT(nn.Module):
#     def __init__(self, num_features, num_hidden, dropout=0.1, heads=1, **kwargs):
#         super(GAT, self).__init__()
#         self.conv1 = GATConv(num_features, num_hidden // heads, heads=heads, **kwargs)
#         self.conv2 = GATConv(num_hidden, num_hidden // heads, heads=heads, **kwargs)
#         self.drop = nn.Dropout(dropout)

#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index).relu()
#         x = self.drop(x)
#         x = self.conv2(x, edge_index)
#         return x

#     def forward_with_weights(self, x, edge_index):
#         x = self.conv1(x, edge_index).relu()
#         x = self.drop(x)
#         x, weights = self.conv2(x, edge_index, return_attention_weights=True)
#         return x, weights


# class GNNChoiceModel(nn.Module):
#     def __init__(
#         self, num_comm_features, num_hidden, edge_index, dropout=0.1, heads=1, **kwargs
#     ):
#         """
#         GNN-based residential location choice model
#         """
#         super(GNNChoiceModel, self).__init__()
#         self.gnn = GAT(num_comm_features, num_hidden, dropout, heads, **kwargs)
#         self.register_buffer("edge_index", edge_index)  # Constant buffer
#         self.num_hidden = num_hidden
#         self.out_layer = nn.Linear(num_hidden, 1)

#     def forward(self, comm_data):
#         """
#         hh_data: (bs, num_hh_features)
#         comm_data: (bs, num_comm, num_comm_features)
#         edge_index: (bs, 2, num_edges)
#         """
#         # Batch over the community graphs
#         # (bs, num_comm, num_comm_features) -> (bs, num_comm, num_hidden)
#         comm_embedding = torch.zeros(
#             comm_data.size(0), comm_data.size(1), self.num_hidden
#         ).to(comm_data.device)
#         for i in range(comm_data.size(0)):
#             comm_embedding[i] = self.gnn(comm_data[i], self.edge_index)

#         out = self.out_layer(comm_embedding.relu()).squeeze()  # (bs, num_comm)

#         return F.log_softmax(out, dim=-1)  # (batch_size, num_comm)


class MLP_Choice(nn.Module):
    def __init__(self, num_features, num_hidden, dropout=0.1):
        """
        Multilayer perceptron model for residential location choice.
        """
        super(MLP_Choice, self).__init__()
        self.num_hidden = num_hidden
        self.fc1 = nn.Linear(num_features, num_hidden)
        self.fc2 = nn.Linear(num_hidden, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        # data: (batch_size, num_comm, num_features)
        x = F.relu(self.fc1(data))  # (batch_size, num_comm, num_hidden)
        x = self.dropout(x)
        x = self.fc2(x).squeeze()  # (batch_size, num_comm, 1)
        return F.log_softmax(x, dim=-1)  # (batch_size, num_comm)
