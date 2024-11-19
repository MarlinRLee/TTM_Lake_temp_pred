import math
import os
import tempfile
import pandas as pd
import torch

from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed
from transformers.integrations import INTEGRATION_TO_CALLBACK

from tsfm_public import TrackingCallback, count_parameters, load_dataset
from tsfm_public.toolkit.get_model import get_model
from tsfm_public.toolkit.lr_finder import optimal_lr_finder
from tsfm_public.toolkit.visualization import plot_predictions

from tsfm_public import (
    TimeSeriesPreprocessor,
    TrackingCallback,
    count_parameters,
    get_datasets,
)

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

DATA_ROOT_PATH = "/scratch.global/csci8523_group_7/sst_ml_data"
csv_files = [os.path.join(DATA_ROOT_PATH, f) for f in os.listdir(DATA_ROOT_PATH) if f.endswith('.csv')]

#Set seed for reproducibility
SEED = 42
set_seed(SEED)

target_dataset = "SST"

# Results dir
OUT_DIR = "ttm_finetuned_models/"


# Forecasting parameters
context_length = 512
forecast_length = 96

# Load the data file and see the columns
timestamp_column = "Time"
valid_columns = [
    "SST_lat0_lon0", "SST_lat0_lon1", "SST_lat0_lon2",
    "SST_lat1_lon0", "SST_lat1_lon1", "SST_lat1_lon2",
    "SST_lat2_lon0", "SST_lat2_lon1", "SST_lat2_lon2"
]
id_columns = []

#data = pd.read_csv(
#    DATA_ROOT_PATH,
#    parse_dates=[timestamp_column],
#)


#data[timestamp_column] = pd.to_datetime(data[timestamp_column])
#print(data.head())
column_specifiers = {
    "timestamp_column": timestamp_column,
    "id_columns": id_columns,
    "target_columns": valid_columns,
}

# Initialize TimeSeriesPreprocessor
tsp = TimeSeriesPreprocessor(
    **column_specifiers,
    context_length=context_length,
    prediction_length=forecast_length,
    scaling=True,
    encode_categorical=False,
    scaler_type="standard",
)

# Load model
TTM_MODEL_PATH = "ibm-granite/granite-timeseries-ttm-r1"
zeroshot_model = get_model(
    TTM_MODEL_PATH,
    context_length=context_length,
    prediction_length=forecast_length,
    prediction_channel_indices=tsp.prediction_channel_indices,
    num_input_channels=tsp.num_input_channels,
)

# Move the model to GPU
zeroshot_model.to(device)


temp_dir = tempfile.mkdtemp()
# zeroshot_trainer
zeroshot_trainer = Trainer(
    model=zeroshot_model,
    args=TrainingArguments(
        output_dir=temp_dir,
        per_device_eval_batch_size=64
    ),
)

split_params = {"train": [0, 0.5], "valid": [0.5, 0.75], "test": [0.75, 1.0]}

#train_dataset, valid_dataset, test_dataset = get_datasets(
#    tsp,
#    data,
#    split_params,
#)


#zeroshot_trainer.evaluate(test_dataset)

# Iterate over each CSV for evaluation
results = []
for csv_file in csv_files:
    print(f"Processing file: {csv_file}")
    
    # Load and preprocess data
    data = pd.read_csv(csv_file, parse_dates=["Time"])
    data = data[["Time"] + valid_columns]  # Keep only valid columns
    
    # Split into train/valid/test datasets
    split_params = {"train": [0, 0.5], "valid": [0.5, 0.75], "test": [0.75, 1.0]}
    train_dataset, valid_dataset, test_dataset = get_datasets(
        tsp, data, split_params
    )
    
    # Evaluate on the test dataset
    result = zeroshot_trainer.evaluate(test_dataset)
    results.append((csv_file, result))
    print(f"Finished evaluating {csv_file}: {result}")

# Output results
for csv_file, result in results:
    print(f"File: {csv_file}, Result: {result}")