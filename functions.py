import os
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from math import sin, cos, sqrt, atan2, radians
import numpy as np
import wandb
import torch
import random
import time
from sklearn.metrics import f1_score
from sklearn.metrics import top_k_accuracy_score


class Config(object):
    def __init__(self, **kwargs):
        self.project_name = "GNN_residential_choice"
        self.optimizer = "adam"  # one of [adam, sgd]
        self.lr = 0.001
        self.lr_scheduler = "one_cycle"  # one of [step, one_cycle, exp, none]
        self.n_epoch = 20
        self.patience = 10
        self.max_run_time = 2  # in hours
        self.bs = 32
        self.model = "GCNConv"  # or any other model name

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getitem__(self, key):
        # Use the dictionary of instance attributes to retrieve the item
        if key in self.__dict__:
            return self.__dict__[key]
        else:
            raise KeyError(f"{key} not found in Config")

    def __setitem__(self, key, value):
        # Set the item in the dictionary of instance attributes
        self.__dict__[key] = value


def config_from_wandb(path):
    api = wandb.Api()
    run = api.run(path)
    config = Config(**run.config)
    config.name = run.name
    return config


def set_seed(seed):
    # Set seed for Python's random module
    random.seed(seed)

    # Set seed for NumPy
    np.random.seed(seed)

    # Set seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Ensure that CuDNN uses deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_distance(lat1, lon1, lat2, lon2):
    # Approximate radius of earth in km
    R = 6371.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c

    return distance


def haversine_np(lat1, lon1, lat2, lon2):
    # Radius of the Earth in kilometers
    R = 6371.0

    # Convert latitude and longitude from degrees to radians
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    # Differences in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Distance in kilometers
    distance = R * c

    return distance


def train(model, criterion, train_loader, val_loader, test_loader, config, device, verbose=True, **kwargs):
    """
    The standard procedure for model training
    """
    # Set the optimizer and learning rate scheduler
    if config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    elif config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)
    elif config["optimizer"] == "lbfgs":
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=config["lr"],
            max_iter=20,
            max_eval=None,
            tolerance_grad=1e-07,
            tolerance_change=1e-09,
            history_size=100,
            line_search_fn="strong_wolfe",
        )
        config["lr_scheduler"] = "none" # LBFGS does not need a scheduler


    if config["lr_scheduler"] == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=len(train_loader),
            gamma=(1 / 1000000) ** (1 / config["n_epoch"]),
        )  # decay lr
    elif config["lr_scheduler"] == "one_cycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config["lr"],
            steps_per_epoch=len(train_loader),
            pct_start=1 / 10,
            epochs=config["n_epoch"],
            anneal_strategy="linear",
            div_factor=1000000,
            final_div_factor=1000000,
        )
    elif config["lr_scheduler"] == "exp":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=(1 / 1000000) ** (1 / (config["n_epoch"] * len(train_loader)))
        )  # decay lr
    elif config['lr_scheduler'] == "cyclic":
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=config["lr"] / 1000000,
            max_lr=config["lr"],
            step_size_up=len(train_loader),
            step_size_down=len(train_loader)*19,
            mode="triangular2",
        )
    elif config["lr_scheduler"] == "none":
        scheduler = None

    run = wandb.init(project=config["project_name"], config=vars(config), mode=config["mode"])
    now = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    run.name = f"{config['project_name']}_{config['model']}_{now}"

    epoch_loss = []
    patience = config.patience
    best_val_loss = np.inf
    epochs_no_improve = 0
    start_time = time.time()
    model.to(device)
    for epoch in range(config.n_epoch):
        model.train()
        train_loss = 0.0
        for i, (inputs, target) in enumerate(train_loader):
            inputs = [input.to(device) for input in inputs] if isinstance(inputs, list) else inputs.to(device)
            target = target.to(device)

            if config["optimizer"] == "lbfgs":
                def closure():
                    optimizer.zero_grad()
                    output = model(*inputs) if isinstance(inputs, list) else model(inputs)
                    loss = criterion(output, target)
                    loss.backward()
                    return loss
                loss = optimizer.step(closure)
            else:
                optimizer.zero_grad()
                output = model(*inputs) if isinstance(inputs, list) else model(inputs)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            if scheduler is not None:
                scheduler.step()

        epoch_loss.append(train_loss / len(train_loader))
        wandb.log({"train_loss": epoch_loss[-1]}, step=epoch)
        if verbose:
            print(
                f"Epoch [{epoch}/{config.n_epoch}], Loss: {epoch_loss[-1]:.4f}"
                f"\t total time: {time.time() - start_time:.2f}"
            )

        # Calculate validation loss
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            for i, (inputs, target) in enumerate(val_loader):
                inputs = [input.to(device) for input in inputs] if isinstance(inputs, list) else inputs.to(device)
                target = target.to(device)
                output = model(*inputs) if isinstance(inputs, list) else model(inputs)
                loss = criterion(output, target)
                val_loss += loss.item()
            val_loss /= len(val_loader)
            if verbose:
                print(
                    f"Epoch [{epoch}/{config.n_epoch}], Val Loss: {val_loss:.4f} \t Train Loss: {epoch_loss[-1]:.4f} "
                    f"\t total time: {time.time() - start_time:.2f}"
                )

            best_val_loss = min(best_val_loss, val_loss)
            wandb.log({"val_loss": np.mean(val_loss)}, step=epoch)

            # Save the current best model
            if val_loss == best_val_loss:
                torch.save(model.state_dict(), f"log/{run.name}.pth")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve == patience:
                print("Early stopping!")
                break
            if time.time() - start_time > config.max_run_time * 3600:
                print(f"Time limit {config.max_run_time} hours reached! Stopping training.")
                break

    if val_loader is not None:
        model.load_state_dict(torch.load(f"log/{run.name}.pth"))
    else:
        torch.save(model.state_dict(), f"log/{run.name}.pth")
        pass

    train_results = evaluate_nn(model, train_loader, kwargs.get("comm", None))
    train_results = {f"train_{key}": value for key, value in train_results.items()}
    wandb.log(train_results)
    if verbose:
        print(f"Train results: {train_results}")

    if test_loader is not None:
        test_results = evaluate_nn(model, test_loader, kwargs.get("comm", None))
        test_results = {f"test_{key}": value for key, value in test_results.items()}
        wandb.log(test_results)
        if verbose:
            print(f"Test results: {test_results}")

    wandb.finish()
    return model


#%% Evaluation criteria
def mrr(target, pred):
    """
    Mean Reciprocal Rank (MRR)
    target: (N,)
    pred: (N, K), can be logits or probabilities
    """
    MRR = 0
    for i in range(len(target)):
        MRR += 1 / (np.where(np.argsort(-pred[i]) == target[i])[0][0] + 1)
    MRR /= len(target)
    return MRR


def error_distance(target, pred, comm):
    """
    Calculate the average distance between the target and predicted communities
    """
    target_coords = comm.loc[target, ["latitude", "longitude"]].values
    pred_coords = comm.loc[pred, ["latitude", "longitude"]].values
    distances = haversine_np(
        target_coords[:, 0], target_coords[:, 1], pred_coords[:, 0], pred_coords[:, 1]
    )
    return distances.mean()


def error_distance_avg(target, p_pred, comm):
    """
    Calculate the average distance between the target and predicted communities,
    weighted by the probability of the predicted community.
    target: (N,)
    p_pred: (N, K), probabilities of the K communities
    comm: Geo
    """
    distance_matrix = haversine_np(
        comm["latitude"].values.reshape(-1, 1),
        comm["longitude"].values.reshape(-1, 1),
        comm["latitude"].values.reshape(1, -1),
        comm["longitude"].values.reshape(1, -1),
    )
    error = 0
    for i in range(len(target)):
        error += (p_pred[i] * distance_matrix[target[i]]).sum()
    error /= len(target)
    return error


def evaluate_nn(model, data_loader, comm):
    model.eval()
    with torch.no_grad():
        log_pred = []
        target = []
        for inputs, output in data_loader:
            if isinstance(inputs, tuple) or isinstance(inputs, list):
                log_pred.append(model(*inputs))
            else:
                log_pred.append(model(inputs))
            target.append(output)

    log_pred = torch.cat(log_pred, dim=0)
    target = torch.cat(target, dim=0)
    log_pred = log_pred.cpu().detach().numpy()
    target = target.cpu().detach().numpy()

    return get_criteria(target, log_pred, comm)


def get_criteria(target, log_pred, comm):
    # loss = nn.NLLLoss(reduction="sum")
    # LLL = -loss(log_pred, target).item()
    LLL = log_pred[np.arange(0, target.shape[0]), target].sum()
    pred = log_pred.argmax(axis=1)
    f1 = f1_score(target, pred, average="macro")
    accuracy = (pred == target).mean()
    MRR = mrr(target, log_pred)
    top_3 = top_k_accuracy_score(
        target, log_pred, k=3, labels=np.arange(log_pred.shape[1])
    )
    top_5 = top_k_accuracy_score(
        target, log_pred, k=5, labels=np.arange(log_pred.shape[1])
    )
    top_10 = top_k_accuracy_score(
        target, log_pred, k=10, labels=np.arange(log_pred.shape[1])
    )
    error_d = error_distance(target, pred, comm)
    error_d_avg = error_distance_avg(target, np.exp(log_pred), comm)
    results = {
        "f1": f1,
        "accuracy": accuracy,
        "MRR": MRR,
        "top_3": top_3,
        "top_5": top_5,
        "top_10": top_10,
        "error_d": error_d,
        "error_d_avg": error_d_avg,
        "LLL": LLL,
    }
    return results


def get_elasticity(model, x, i):
    # Calculate the elasticity of the i-th alternative with respect to its features
    model.zero_grad()
    x.requires_grad = True
    y = model(x.unsqueeze(0)).exp().squeeze()[i]  # (num_comm,)
    d = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y))[
        0
    ]  # (num_comm, num_features)
    elasticity = d[i] * x[i] / y
    model.zero_grad()
    x.grad = None
    x.requires_grad = False
    return elasticity


def get_cross_elasticity(model, x, i, j):
    # Calculate the cross elasticity of the i-th alternative with respect to changes in the j-th alternative
    # x in (num_comm, num_features)
    model.zero_grad()
    x.requires_grad = True  # Enable differentiation
    y = model(x.unsqueeze(0)).exp().squeeze()[i]  # (num_comm,)
    d = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y))[
        0
    ]  # (num_comm, num_features)
    elasticity = d[j] * x[j] / y
    model.zero_grad()
    # zero grad x
    x.grad = None
    x.requires_grad = False
    return elasticity


def find_neighbors(edge_index, node_id):
    """Find the neighbors of a given node in the graph.
    edge_index:
        The edge index matrix (n x 2) of the graph.
    node_id: int
    """
    neighbors = []
    for i in range(edge_index.shape[1]):
        if edge_index[0][i] == node_id:
            if type(edge_index[1][i]) == torch.Tensor:
                neighbors.append(edge_index[1][i].item())
            else:
                neighbors.append(edge_index[1][i])
    return neighbors


def find_k_hop_neighbors(edge_index, node_id, k):
    """Find the k-hop neighbors of a given node in the graph.
    edge_index:
        The edge index matrix (n x 2) of the graph.
    node_id: int
    k: int
    """
    neighbors = [node_id]
    for _ in range(k):
        new_neighbors = []
        for n in neighbors:
            new_neighbors.extend(find_neighbors(edge_index, n))
        neighbors = list(set(new_neighbors))
    return neighbors
