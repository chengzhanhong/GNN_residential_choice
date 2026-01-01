# GNN for residential location choice

- Cheng, Z., Hu, L., Bu, Y., Zhou, Y., & Wang, S. (2025). Graph neural networks for residential location choice: connection to classical logit models. arXiv preprint: https://arxiv.org/abs/2509.07123

## Usage
The main scripts and notebooks used in the experiments are listed below:

### Core implementation
- [models.py](models.py): Contains the implementation of various GNNs and MNL, SCL baselines used in the experiments.
- [functions.py](functions.py): Utility functions for data processing, model training, and evaluation.

### Experiments
- [exp_training.ipynb](exp_training.ipynb): Experiment script for training and evaluating GNN models and baselines on residential location choice tasks.
- [exp_ablation_gnn.py](exp_ablation_gnn.py): Experiment script for ablation studies on different GNN architectures and configurations.
- [exp_interpret.ipynb](exp_interpret.ipynb): Experiment script for interpreting GNN models.
- [exp_elasticity.ipynb](exp_elasticity.ipynb): Experiment script for elasticitiy-related analyses.
- [analyze_gat_attentions.ipynb](analyze_gat_attentions.ipynb): Notebook for analyzing attention weights in GAT models.

## Dependencies
Environment under which the code is tested:
- Python 3.13
- PyTorch 2.7.0
- PyTorch Geometric 2.6.1
- Numpy 2.2.5
- Pandas 2.2.3

Additional packages:
- `matplotlib` and `seaborn` for visualization
- `wandb` for experiment tracking and logging

## Notes
- Some notebooks assume access to preprocessed datasets or model loads and may require path adjustments.