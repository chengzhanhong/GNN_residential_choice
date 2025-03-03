"""
Load data, define the datasets to be used in other models
"""
#%% Load data
from functions import *
import pandas as pd
import geopandas as gd

def load_data():
    hh = pd.read_csv(
        "data/household_info_chicago_community-rent-own-full_work_location.csv",
    )
    comm = gd.read_file("data/community areas/community_area_info_all.shp")
    hh["comm_id_home"] = hh["comm_id_home"] - 1  # Adjusting for zero indexing
    comm["comm_id"] = comm["comm_id"] - 1  # Adjusting for one indexing
    # Build graph for the community areas
    # Spatial join to find adjacent communities
    adjacent_pairs = gd.sjoin(comm, comm, how="inner", predicate="touches")
    # Remove self-joins (where a community touches itself)
    adjacent_pairs = adjacent_pairs[
        adjacent_pairs["comm_id_left"] != adjacent_pairs["comm_id_right"]
    ]
    adjacent_pairs = adjacent_pairs.sort_values(["comm_id_left", "comm_id_right"])
    edge_index = (
        torch.tensor(adjacent_pairs[["comm_id_left", "comm_id_right"]].values)
        .t()
        .contiguous()
        .to(torch.long)
    )
    comm.set_index("comm_id", inplace=True)

    # I need pop_density, white_prop, black_prop, single_res, multi_res,
    # office, retail, land_mix, transit_a, med_year, med_value, h_units
    comm["pop_density"] = comm["total_pop"] / comm["ALAND"] * 100
    comm["white_prop"] = comm["white"] / comm["total_pop"]
    comm["black_prop"] = comm["black"] / comm["total_pop"]
    comm["transit_a_scaled"] = comm["transit_a"] / 5
    comm["med_house_age_scaled"] = (2017 - comm["med_year"]) / 50
    comm["med_value_scaled"] = comm["med_value"] / comm["med_value"].max()
    comm["h_units_scaled"] = comm["h_units"] / comm["h_units"].max()
    comm["median_inc_scaled"] = comm["median_inc"] / comm["median_inc"].max()

    # fill nan with community median
    hh["hh_income"] = hh["hh_income"].fillna(hh["comm_id_home"].map(comm["median_inc"]))
    hh["hh_income_scaled"] = hh["hh_income"] / hh["hh_income"].max()
    hh["hh_income_scaled"] = hh["hh_income"] / comm["median_inc"].max()
    hh["race_white"] = (hh.race == 1).astype("float")
    hh["race_black"] = (hh.race == 2).astype("float")
    distance_to_work = haversine_np(
        hh.latitude_work.values.reshape(-1, 1),
        hh.longitude_work.values.reshape(-1, 1),
        comm.latitude.values.reshape(1, -1),
        comm.longitude.values.reshape(1, -1),
    )
    distance_to_work = np.log(distance_to_work + 1)
    return comm, hh, edge_index, distance_to_work


#%% Define the dataset in a way that is commonly used in GNN models/Deep Learning
class spatial_choice_dataset(torch.utils.data.Dataset):
    def __init__(
        self, comm, hh, distance_to_work, comm_features, hh_features
    ):
        self.comm_features = comm_features
        self.hh_features = hh_features
        self.distance_to_work = torch.tensor(
            distance_to_work, dtype=torch.float32
        )  # N_hh x N_comm
        self.target = torch.tensor(hh["comm_id_home"].values, dtype=torch.long)
        self.comm = comm
        self.hh = hh
        self.comm_data = torch.tensor(
            self.comm[self.comm_features].values, dtype=torch.float32
        )  # N_comm x F_comm
        self.hh_data = torch.tensor(
            self.hh[self.hh_features].values, dtype=torch.float32
        )  # N_hh x F_hh
        self.feature_names = (
            self.comm_features + ["distance_to_work"] + self.hh_features
        )

    def __len__(self):
        return len(self.hh)

    def __getitem__(self, idx):
        # Get the household data
        household_data = self.hh_data[idx]  # (F_hh,)
        # Get the distance to work for this household
        distance = self.distance_to_work[idx]  # (N_comm,)
        comm_data = torch.cat(
            (
                self.comm_data,
                distance.unsqueeze(1),
                household_data.unsqueeze(0).expand(self.comm_data.size(0), -1),
            ),
            dim=1,
        )  # (N_comm, F_comm + 1 + F_hh)
        return (
            comm_data,  # Return inputs
            self.target[idx],  # Return target
        )


#%%
class spatial_choice_dataset_interact(torch.utils.data.Dataset):
    def __init__(
        self,
        comm,
        hh,
        distance_to_work,
        comm_features,
        hh_features,
    ):
        self.comm_features = comm_features
        self.hh_features = hh_features
        self.distance_to_work = torch.tensor(
            distance_to_work, dtype=torch.float32
        )  # N_hh x N_comm
        self.target = torch.tensor(hh["comm_id_home"].values, dtype=torch.long)
        self.comm = comm
        self.hh = hh
        self.comm_data_raw = torch.tensor(
            self.comm[self.comm_features].values, dtype=torch.float32
        )  # N_comm x F_comm
        self.hh_data = torch.tensor(
            self.hh[self.hh_features].values, dtype=torch.float32
        )  # N_hh x F_hh
        self.hh_name2idx = {name: i for i, name in enumerate(self.hh_features)}
        self.comm_name2idx = {name: i for i, name in enumerate(self.comm_features)}
        comm_features2 = comm_features.copy()
        comm_features2.remove("median_inc_scaled")
        comm_features2.remove("white_prop")
        comm_features2.remove("black_prop")
        self.comm_data = torch.tensor(
            self.comm[comm_features2].values, dtype=torch.float32
        )
        self.feature_names = (comm_features2 +
                              ['distance_to_work',
                              'home_income_interact',
                              'white_interact',
                              'black_interact'])

    def __len__(self):
        return len(self.hh)

    def __getitem__(self, idx):
        # Get the household data
        household_data = self.hh_data[idx]  # (F_hh,)
        # Get the distance to work for this household
        distance = self.distance_to_work[idx]  # (N_comm,)

        # interact_data,
        # household_data
        hh_income_interact = (
            household_data[self.hh_name2idx["hh_income_scaled"]]
            - self.comm_data_raw[:, self.comm_name2idx["median_inc_scaled"]]
        )
        hh_white_interact = (
            household_data[self.hh_name2idx["race_white"]]
            * self.comm_data_raw[:, self.comm_name2idx["white_prop"]]
        )
        hh_black_interact = (
            household_data[self.hh_name2idx["race_black"]]
            * self.comm_data_raw[:, self.comm_name2idx["black_prop"]]
        )
        interact_data = torch.stack(
            [hh_income_interact, hh_white_interact, hh_black_interact], dim=1
        )

        comm_data = torch.cat(
            (
                self.comm_data,
                distance.unsqueeze(1),
                interact_data,
            ),
            dim=1,
        )  # (N_comm, F_comm + 1 + F_hh)
        return (
            comm_data,  # Return inputs
            self.target[idx],  # Return target
        )
