# To test the impact of different number of layers, hidden dimensions, and gnn update functions.
# %% Initial setup
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from functions import *
from models import GNNChoiceModel
from data_process import load_data, spatial_choice_dataset_interact

comm, hh, edge_index, distance_to_work = load_data()

comm_features = [
    "pop_density",
    "white_prop",
    "black_prop",
    "single_res",
    "multi_res",
    "office",
    # "retail",
    "land_mix",
    "transit_a_scaled",
    "med_house_age_scaled",
    "med_value_scaled",
    "h_units_scaled",
    "median_inc_scaled",
]

hh_features = ["hh_income_scaled", "race_white", "race_black"]

# %% Set the training and evaluation procedure
device = torch.device("cpu")
config = Config()
config.bs = 32
config.num_hidden = 64
config.dropout = 0.05
config.optimizer = "adam"  # one of [adam, sgd]
config.lr = 0.01
config.lr_scheduler = "one_cycle"  # one of [step, one_cycle, exp, none]
config.n_epoch = 20
config.model = "MPNet"  # GAT, GCN, or MPNet (Message Passing Network)
config.aggr = "lse"  # Aggregation method for MPNet, 'add', 'mean', 'max', 'lse', for other models it is not used
config.heads = 8  # Number of attention heads for GAT
config.mode = "online"  # online or disabled
config.residual = True  # Whether to use residual connections in GNN
config.seed = 100
config.n_layer = 2  # Number of layers for GNN
config.use_edge_info = False  # Whether to use edge features in GNN
config.edge_dim = None  # Edge features, set to None if not used

if config.use_edge_info and config.model == "GAT":
    config.edge_dim = 1  # Edge features, set to None if not used

for n_layer in [1, 2, 3]:
    for name in ["MPNet"]:
        config.n_layer = n_layer
        config.model = name

        if config.model == "GAT":
            config.heads = 8 if config.num_hidden % 8 == 0 else 1

        my_dataset = spatial_choice_dataset_interact
        for i in [4]:
            test_idx = np.arange(i, len(hh), 10)
            train_idx = np.setdiff1d(np.arange(len(hh)), test_idx)
            train_dataset = my_dataset(
                comm,
                hh.loc[train_idx, :],
                distance_to_work[train_idx],
                comm_features,
                hh_features,
            )
            test_dataset = my_dataset(
                comm,
                hh.loc[test_idx, :],
                distance_to_work[test_idx],
                comm_features,
                hh_features,
            )

            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=config.bs, shuffle=True
            )
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=len(test_dataset), shuffle=False
            )
            criterion = nn.NLLLoss()

            set_seed(config.seed)
            model = GNNChoiceModel(
                model_name=config.model,
                num_features=train_dataset[0][0].shape[-1],
                num_hidden=config.num_hidden,
                edge_index=edge_index,
                dropout=config.dropout,
                heads=config.heads,
                residual=config.residual,
                n_layer=config.n_layer,
                aggr=config.aggr,
                edge_dim=config.edge_dim,
            ).to(device)

            set_seed(config.seed)
            model = train(
                model,
                criterion,
                train_loader,
                None,
                test_loader,
                config,
                device,
                verbose=True,
                comm=comm,
            )

# %%
