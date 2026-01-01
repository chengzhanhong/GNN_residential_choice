"""
Define residential location choice models
"""
# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
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


class GSCL_Choice(nn.Module):
    def __init__(self, num_features, num_comm, distance, edge_index=None):
        """
        Generalized Spatially correlated logit model
        The same to the original GSCL_Choice. When the graph are fully connected, the edge_index is not used as input.

        num_features: Number of features for each community.
        num_comm: Number of communities.
        distance: A tensor of shape (num_comm, num_comm) representing the distance between communities.
        edge_index: Optional, a tensor of shape (2, num_edges) representing the edges in the graph.
                    If not provided, a fully connected graph is assumed.
        """
        super(GSCL_Choice, self).__init__()
        if edge_index is None:
            edge_index = torch.tensor(
                [[i for i in range(num_comm) for _ in range(num_comm)],
                 [j for _ in range(num_comm) for j in range(num_comm)]],
                dtype=torch.long
            )
        # Delete self-loops
        edge_index = edge_index[:, edge_index[0] != edge_index[1]]
        num_edge = edge_index.size(1)
        self.num_comm = num_comm
        self.num_edge = num_edge

        mapping_matrix = torch.zeros(
            num_edge, num_comm, dtype=torch.float32, requires_grad=False
        )
        mapping_matrix[torch.arange(num_edge), edge_index[0]] = 1

        # Constant buffers
        self.register_buffer("ei", edge_index)
        self.register_buffer("dm", torch.tensor(distance, dtype=torch.float32))
        self.register_buffer("mm", mapping_matrix)

        # Define the parameter $\phi$ with the bounds (-inf, 0)
        self.phi_raw = nn.Parameter(torch.tensor(2.0), requires_grad=True)
        # Define the dissimilarity parameter $\mu$ with the bounds (0, 1]
        self.mu_raw = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.fc1 = nn.Linear(num_features, 1, bias=False)
        # Define alternative-specific constants
        self.asc = nn.Parameter(torch.zeros(num_comm - 1), requires_grad=True)

    def forward(self, comm_data):
        # Batch over the community graphs
        # (bs, num_comm, num_comm_features) -> (bs, num_comm)
        x = self.fc1(comm_data).squeeze(-1)  # (batch_size, num_comm)
        x[:, 1:] += self.asc
        mu = torch.sigmoid(self.mu_raw)  # (0, 1]
        phi = -F.softplus(self.phi_raw)  # Ensure phi is negative

        # Calculate the allocation matrix
        am = torch.zeros(self.num_comm, self.num_comm, dtype=torch.float32)  # (num_comm, num_comm)
        am[self.ei[0], self.ei[1]] = self.dm[self.ei[0], self.ei[1]] ** phi
        am = am / am.sum(dim=1, keepdim=True)  # Normalize allocation matrix

        n_start = (am[self.ei[0], self.ei[1]] * torch.exp(x[:, self.ei[0]])) ** (
            1 / mu
        )  # (batch_size, num_edge)

        n_end = (am[self.ei[1], self.ei[0]] * torch.exp(x[:, self.ei[1]])) ** (
            1 / mu
        )  # (batch_size, num_edge)

        exp_utility = (
            n_start * (n_start + n_end) ** (mu - 1)
        ) @ self.mm  # (batch_size, num_comm)

        p = exp_utility / exp_utility.sum(dim=1, keepdim=True)  # (batch_size, num_comm)
        return torch.log(p)  # (batch_size, num_comm)


# -----------------------------------------------------------------------------------------------
# Wrapper for Graph Attention neural network (GATConv)
# Allows for multiple layers, heads, hidden dimensions, whether to use residual connections, and edge features.
# -----------------------------------------------------------------------------------------------

class GAT(nn.Module):
    def __init__(
        self,
        num_hidden,
        dropout=0.1,
        heads=1,
        n_layer=2,
        edge_dim=None,
        residual=False,
        **kwargs,
    ):
        super(GAT, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(n_layer):
            self.convs.append(
                GATConv(
                    num_hidden,
                    num_hidden // heads,
                    heads=heads,
                    edge_dim=edge_dim,
                    residual=residual,
                    **kwargs,
                )
            )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr=None):
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr=edge_attr).relu()
            x = self.drop(x)
        return x

# -----------------------------------------------------------------------------------------------
# Wrapper for Graph Convolutional neural network (GCNConv)
# Allows for multiple layers, number of hidden dimensions, whether to use gated residual connections, and edge weights.
# -----------------------------------------------------------------------------------------------------
class GCN(nn.Module):
    def __init__(
        self, num_hidden, dropout=0.1, n_layer=2, residual=False, **kwargs
    ):
        super(GCN, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(n_layer):
            self.convs.append(
                GCNConv(num_hidden, num_hidden)
            )
        self.drop = nn.Dropout(dropout)

        # define gated residual connection
        if residual:
            self.res = nn.ModuleList()
            for _ in range(n_layer):
                self.res.append(
                    nn.Linear(num_hidden, num_hidden, bias=True)
                )
        else:
            self.register_parameter('res', None)

    def forward(self, x, edge_index, edge_weight=None):
        for i, conv in enumerate(self.convs):
            h_in = x  # For residual connection
            x = conv(x, edge_index, edge_weight=edge_weight)

            # Apply the gated residual connection
            if self.res is not None:
                g = torch.sigmoid(self.res[i](h_in))
                x = (1 - g) * h_in + g * x

            # Apply activation and dropout
            x = x.relu()
            x = self.drop(x)
        return x


# -----------------------------------------------------------------------------------------------------
# Message passing neural network model
# Allows for different aggregation functions such as 'add', 'mean', 'max', and 'lse' (Log-Sum-Exp).
# For historical reasons, the current gated residual connections are implemented at a different location compared to the GCNConv.
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


class MPConv(MessagePassing):
    def __init__(self, in_dims, out_dims, aggr="add", norm=False, residual=False, **kwargs):
        """
        Message Passing Network layer with different aggregation functions.
        Args:
            aggr: Aggregation function to use: 'add', 'mean', 'max', 'lse'(Log-Sum-Exp).
            norm: Whether to normalize the node features.
            residual: Whether to use residual connection.
        """
        # If using lse aggregation, we need to use the custom aggregation class.
        if aggr == "lse":
            aggr = LogSumExpAggregation()
        super().__init__(aggr=aggr)  # aggregation.
        self.lin = nn.Linear(in_dims, out_dims, bias=False)
        self.bias = nn.Parameter(torch.empty(out_dims))
        self.norm = norm  # Whether to normalize the node features.
        self.sigmoid = nn.Sigmoid()
        if residual:
            self.res = nn.Linear(
                in_dims,
                out_dims,
                bias=True,
            )
        else:
            self.register_parameter("res", None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()
        if self.res is not None:
            self.res.reset_parameters()

    def forward(self, x, edge_index, **kwargs):
        # x has shape [N, in_dims]
        # edge_index has shape [2, E]

        # Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        h_in = x  # For residual connection

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

        if self.res is not None:
            # Apply the g to weight the messages.
            g = torch.sigmoid(self.res(h_in))
            out = (1 - g) * h_in + g * out  # Gated residual connection.

        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_dims]
        # Normalize node features.
        return norm.view(-1, 1) * x_j


class MPNet(nn.Module):
    """
    Message Passing Network with multiple layers.
    """
    def __init__(
        self,
        num_hidden,
        dropout=0.1,
        aggr="add",
        norm=False,
        n_layer=2,
        residual=False,
        **kwargs,
    ):
        super(MPNet, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(n_layer):
            self.convs.append(
                MPConv(num_hidden, num_hidden, aggr, norm, residual=residual, **kwargs)
            )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index, **kwargs):
        for conv in self.convs:
            x = conv(x, edge_index, **kwargs).relu()
            x = self.drop(x)
        return x


class GNNChoiceModel(nn.Module):
    def __init__(
        self,
        num_features,
        num_hidden,
        edge_index,
        dropout=0.1,
        n_layer=2,
        edge_attr=None,
        model_name="GAT",
        **kwargs,
    ):
        """
        GNN-based residential location choice model.
        """
        model_dict = {"GCN": GCN, "GAT": GAT, "MPNet": MPNet}
        super(GNNChoiceModel, self).__init__()
        self.in_layer = nn.Linear(num_features, num_hidden)
        self.gnn = model_dict[model_name](
            num_hidden, dropout=dropout, n_layer=n_layer, **kwargs
        )
        self.register_buffer("edge_index", edge_index)  # Constant buffer

        if edge_attr is not None:
            self.register_buffer("edge_attr", edge_attr)
        else:
            self.edge_attr = None
        self.num_hidden = num_hidden
        self.out_layer = nn.Linear(num_hidden, 1)

    def forward(self, data):
        """
        data: (bs, num_class, num_features)
        edge_index: (bs, 2, num_edges)
        """
        # Batch over the community graphs
        # (bs, num_class, num_features) -> (bs, num_class, num_hidden)
        data = self.in_layer(data)  # (bs, num_class, num_hidden)
        comm_embedding = torch.zeros(
            data.size(0), data.size(1), self.num_hidden
        ).to(data.device)
        for i in range(data.size(0)):
            if self.edge_attr is not None:
                comm_embedding[i] = self.gnn(data[i], self.edge_index, self.edge_attr)
            else:
                comm_embedding[i] = self.gnn(data[i], self.edge_index)

        out = self.out_layer(comm_embedding).squeeze()  # (bs, num_class)

        return F.log_softmax(out, dim=-1)  # (batch_size, num_class)


# class MLP_Choice(nn.Module):
#     def __init__(self, num_features, num_hidden, dropout=0.1):
#         """
#         Multilayer perceptron model for residential location choice.
#         """
#         super(MLP_Choice, self).__init__()
#         self.num_hidden = num_hidden
#         self.fc1 = nn.Linear(num_features, num_hidden)
#         self.fc2 = nn.Linear(num_hidden, 1)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, data):
#         # data: (batch_size, num_comm, num_features)
#         x = F.relu(self.fc1(data))  # (batch_size, num_comm, num_hidden)
#         x = self.dropout(x)
#         x = self.fc2(x).squeeze()  # (batch_size, num_comm, 1)
#         return F.log_softmax(x, dim=-1)  # (batch_size, num_comm)


class MLP_Choice(nn.Module):
    def __init__(self, num_features, num_hidden, n_layer=2, dropout=0.1):
        """
        Multilayer perceptron model for residential location choice.
        """
        super(MLP_Choice, self).__init__()
        self.num_hidden = num_hidden

        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(num_features, num_hidden))
        for _ in range(n_layer - 1):
            self.fc_layers.append(nn.Linear(num_hidden, num_hidden))
        self.fc_layers.append(nn.Linear(num_hidden, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        # data: (batch_size, num_comm, num_features)
        x = data
        for fc in self.fc_layers[:-1]:
            x = F.relu(fc(x))
            x = self.dropout(x)
        x = self.fc_layers[-1](x)  # No activation here
        x = x.squeeze()  # (batch_size, num_comm, 1)
        return F.log_softmax(x, dim=-1)  # (batch_size, num_comm)
