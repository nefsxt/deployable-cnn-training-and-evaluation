# Centralized Constraints and Configurations

import torch
import numpy as np

# Reproducibility
SEED = 11
torch.manual_seed(SEED)
np.random.seed(SEED)

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
DATA_DIR = "./data"
NUM_CLASSES = 10

# Training
TUNING_EPOCHS = 25
FINAL_EPOCHS = 100

# Dataset metadata
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Dataloader
NUM_WORKERS = 2 # how many subprocesses are used to load data in parallel
PIN_MEMORY = True # tells PyTorch to allocate data in page-locked (pinned) memory on the host (CPU) side 
#                       -> no swap to disk when loading data, quicker DMA to GPU (does not help w CPU - safe to be True when CPU only)  

# MLflow
EXPERIMENT_NAME = "cifar10_cnn_optuna"
MODEL_NAME = "simple_cnn"

# Metrics
TOP_K = 5

#Optuna 
NUM_STARTUP_TRIALS = 5 # number of initial trials Optuna lets run fully before it starts comparing trials (MedianPruner)
NUM_WARMUP_STEPS = 10  # number of steps/epochs in each trial before pruning is allowed (MedianPruner)
INTERVAL_STEPS = 1 # check for pruning at every 1 epoch (MedianPruner)
NUM_OPTUNA_TRIALS=25  # will execute 25 trials (exploration of search space) - Random Search 