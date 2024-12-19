import os
import tempfile
import pandas as pd
import torch
from transformers import Trainer, TrainingArguments, set_seed
from transformers.integrations import INTEGRATION_TO_CALLBACK

from tsfm_public.toolkit.visualization import plot_predictions

from util import load_data, lat_lon_pred, load_tsp, prep_env, load_model
from tsfm_public import (
    get_datasets,
)

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

device = prep_env()

# Results dir
OUT_DIR = "ttm_finetuned_models/temp/"
os.makedirs(OUT_DIR, exist_ok=True)
context_length = 512
forecast_length = 96

data = load_data()
tsp = load_tsp(context_length, forecast_length)

split_params = {"train": [0, 0.5], "valid": [0.5, 0.75], "test": [0.75, 1.0]}
train_dataset, valid_dataset, test_dataset = get_datasets(
    tsp,
    data,
    split_params,
)
    
# Load model
zeroshot_model = load_model(tsp, context_length, forecast_length, device, isTraining = False)

temp_dir = tempfile.mkdtemp()
# zeroshot_trainer
zeroshot_trainer = Trainer(
    model=zeroshot_model,
    args=TrainingArguments(
        output_dir=temp_dir,
        per_device_eval_batch_size=32,
        disable_tqdm = True
    ),
)
zeroshot_output = zeroshot_trainer.evaluate(test_dataset)
print(zeroshot_output, flush = True)

plot_predictions(
    model=zeroshot_trainer.model,
    dset=test_dataset,
    plot_dir=os.path.join(OUT_DIR, "Zero_Pred"),
    plot_prefix="test_zeroshot",
    channel=0)


#output_file = os.path.join(OUT_DIR, "lat_lon_errors_zeroshot.csv")
#lat_lon_pred(data, tsp, zeroshot_trainer, output_file, context_length, forecast_length)