# To test the impact of different number of layers, hidden dimensions, and gnn update functions.
# %% Initial setup
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from functions import *
from models import GCNChoiceModel
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
config.model = "GCNConv"  # or any other model name
config.mode = "online"  # online or disabled
config.aggr = "mean"  # Aggregation method for GCN, 'add', 'mean', 'max', 'lse'
config.norm = "True"  # Normalization for GCN
config.seed = 100
config.n_layer = 2  # Number of layers for GNN

for aggr in ["add", "mean", "max", "lse"]:
    for norm in [True, False]:
        config.norm = norm
        config.aggr = aggr
        my_dataset = spatial_choice_dataset_interact
        for i in range(10):
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
            model = GCNChoiceModel(
                train_dataset[0][0].shape[-1],
                config.num_hidden,
                edge_index,
                dropout=config.dropout,
                aggr=config.aggr,
                n_layer=config.n_layer,
                norm=config.norm,
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
                verbose=False,
                comm=comm
            )

# %%
