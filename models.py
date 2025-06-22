"""
Define residential location choice models
"""
# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from scipy.stats import t
import pandas as pd


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

    def unpack_param(self, param_vector):
        """
        Unpack the model parameters from a vector.
        Args:
            param_vector: A 1D tensor containing the model parameters.
        Returns: A dictionary mapping parameter names to their values.
        """
        param_dict = {}
        idx = 0
        for name, p in self.named_parameters():
            num_param = p.numel()
            param_dict[name] = param_vector[idx : idx + num_param].view_as(p)
            idx += num_param
        return param_dict

    def get_param_std(self, x, y):
        """
        Calculate the parameter robust standard errors, using hessian and jacobian.

        Returns: a vector of standard errors for each parameter.
        """

        def loss(param, reduction="none"):
            """
            Calculate the negative log-likelihood loss.
            Args:
                param: A vector of model parameters.
                reduction: Specifies the reduction to apply to the output:
                    'none' | 'mean' | 'sum'.
            """
            param_dict = self.unpack_param(param)
            log_probs = torch.func.functional_call(self, param_dict, x)
            nll = F.nll_loss(log_probs, y, reduction=reduction)
            return nll

        def loss_each(param):
            # Calculate the negative log-likelihood loss for each sample.
            return loss(param, reduction="none")

        def loss_total(param):
            # Calculate the total negative log-likelihood loss.
            return loss(param, reduction="sum")

        # Get the model parameters as a vector
        param = torch.cat([p.view(-1) for p in self.parameters()])

        # Compute the Jacobian and Hessian
        jacobian = torch.autograd.functional.jacobian(loss_each, param)
        hessian = torch.autograd.functional.hessian(loss_total, param)

        # Compute the standard errors
        B = jacobian.T @ jacobian  # (n_params, n_params)
        Hinv = torch.linalg.pinv(hessian)
        COV = Hinv @ B @ Hinv
        std_err = torch.sqrt(torch.diag(COV))

        return std_err

    def significance_test(self, x, y):
        """
        Perform a significance test on the model parameters.
        Args:
            x: Input features.
            y: Target labels.
        Returns: A pandas DataFrame containing the parameter estimates, standard errors, t-values, and p-values.
        """
        std_err = self.get_param_std(x, y).detach().cpu().numpy()
        param_vector = (
            torch.cat([p.view(-1) for p in self.parameters()]).detach().cpu().numpy()
        )
        t_values = param_vector / std_err
        p_values = 2 * (
            1 - t.cdf(abs(t_values), df=len(x) - len(param_vector))
        )  # two-tailed test

        # organize the results in a pandas DataFrame, columns are params, std_err, t_values, p_values
        # Row names are the parameter names
        results = {
            "params": param_vector,
            "std_err": std_err,
            "t_values": t_values,
            "p_values": p_values,
        }

        param_names = []
        for name, value in self.named_parameters():
            param_names.extend([f"{name}_{i}" for i in range(value.numel())])

        results_df = pd.DataFrame(results, index=param_names)
        return results_df


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
        self, num_features, num_hidden, edge_index, dropout=0.1, heads=1, n_layer=2, **kwargs
    ):
        """
        GNN-based residential location choice model (specifically GAT).
        """
        super(GNNChoiceModel, self).__init__()
        self.gnn = GAT(num_features, num_hidden, dropout, heads, n_layer=n_layer, **kwargs)
        self.register_buffer("edge_index", edge_index)  # Constant buffer
        self.num_hidden = num_hidden
        self.out_layer = nn.Linear(num_hidden, 1)

    def forward(self, data):
        """
        data: (bs, num_class, num_features)
        edge_index: (bs, 2, num_edges)
        """
        # Batch over the community graphs
        # (bs, num_class, num_features) -> (bs, num_class, num_hidden)
        comm_embedding = torch.zeros(
            data.size(0), data.size(1), self.num_hidden
        ).to(data.device)
        for i in range(data.size(0)):
            comm_embedding[i] = self.gnn(data[i], self.edge_index)

        out = self.out_layer(comm_embedding).squeeze()  # (bs, num_class)

        return F.log_softmax(out, dim=-1)  # (batch_size, num_class)


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


# -----------------------------------------------------------------------------------------------------
# Graph Convolutional neural network model with different types of aggregation functions
# For historical reasons, the current GCNConv and GATConv have different interfaces. We may combine them in the future.
# -----------------------------------------------------------------------------------------------------
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.aggr import Aggregation


class LogSumExpAggregation(Aggregation):
    """An aggregation operator that computes the log-sum-exp (LSE) of features across a set of elements.
    LSE is not a standard aggregation function like sum or mean, so we implement it as a custom aggregation class.
    LSE is computed in a numerically stable way (log-sum-exp trick) to avoid overflow/underflow issues:

    .. math::
        \text{LSE}(x_1, \dots, x_n) = \max(x) + \log\left( \sum_i \exp(x_i - \max(x)) \right)
    """
    def forward(self, x, index, **kwargs):
        # x: [num_edges, F]
        # index: [num_edges] or None
        # The maximum value of each group
        out = self.reduce(x, index, reduce="max", **kwargs)
        shifted = x - out[index]
        # Sum exp(shifted)
        exp_sum = self.reduce(shifted.exp(), index, reduce="sum", **kwargs)
        # Final log-sum-exp
        lse = out + torch.log(exp_sum + 1e-10)
        return lse


class GCNConv(MessagePassing):
    def __init__(self, in_dims, out_dims, aggr="add", norm=False):
        """
        Graph Convolutional Network (GCN) layer with different aggregation functions.
        Args:
            aggr: Aggregation function to use: 'add', 'mean', 'max', 'lse'(Log-Sum-Exp).
            norm: Whether to normalize the node features.
        """
        # If using lse aggregation, we need to use the custom aggregation class.
        if aggr == "lse":
            aggr = LogSumExpAggregation()
        super().__init__(aggr=aggr)  # aggregation.
        self.lin = nn.Linear(in_dims, out_dims, bias=False)
        self.bias = nn.Parameter(torch.empty(out_dims))
        self.norm = norm  # Whether to normalize the node features.

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        # x has shape [N, in_dims]
        # edge_index has shape [2, E]

        # Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Linearly transform node feature matrix.
        x = self.lin(x)

        # If compute normalization.
        if self.norm:
            row, col = edge_index
            deg = degree(col, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        else:  # No normalization
            norm = torch.ones(edge_index.size(1), dtype=x.dtype, device=x.device)

        # Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)

        # Apply a final bias vector.
        out = out + self.bias

        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_dims]

        # Normalize node features.
        return norm.view(-1, 1) * x_j


class GCN(nn.Module):
    """
    Includes multiple GCNConv layers with ReLU activation and dropout.
    """
    def __init__(
        self, num_features, num_hidden, dropout=0.1, aggr="add", norm=False, n_layer=2
    ):
        super(GCN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(
            GCNConv(num_features, num_hidden, aggr, norm)
        )
        for _ in range(n_layer - 1):
            self.convs.append(
                GCNConv(num_hidden, num_hidden, aggr, norm)
            )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
            x = self.drop(x)
        return x


class GCNChoiceModel(nn.Module):
    def __init__(
        self, num_features, num_hidden, edge_index, dropout=0.1, aggr="add", n_layer=2, norm=False
    ):
        """
        GCN-based residential location choice model.
        """
        super(GCNChoiceModel, self).__init__()
        self.gnn = GCN(num_features, num_hidden, dropout, aggr=aggr, n_layer=n_layer, norm=norm)
        self.register_buffer("edge_index", edge_index)  # Constant buffer
        self.num_hidden = num_hidden
        self.out_layer = nn.Linear(num_hidden, 1)

    def forward(self, data):
        """
        data: (bs, num_class, num_features)
        edge_index: (bs, 2, num_edges)
        """
        # Batch over the community graphs
        # (bs, num_class, num_features) -> (bs, num_class, num_hidden)
        comm_embedding = torch.zeros(data.size(0), data.size(1), self.num_hidden).to(
            data.device
        )
        for i in range(data.size(0)):
            comm_embedding[i] = self.gnn(data[i], self.edge_index)

        out = self.out_layer(comm_embedding).squeeze()  # (bs, num_class)

        return F.log_softmax(out, dim=-1)  # (batch_size, num_class)