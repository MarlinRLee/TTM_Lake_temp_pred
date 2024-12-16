import os
from util import Train_Model, load_data, lat_lon_pred, load_model, load_tsp, prep_env
from tsfm_public import get_datasets
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
split_params = {"train": .1, "test": .6}
train_dataset, valid_dataset, test_dataset = get_datasets(
    tsp,
    data,
    split_params,
)

# Load model
finetune_forecast_model = load_model(tsp, context_length, forecast_length, device, isTraining = True)

num_epochs = 30 
batch_size = 128

finetune_forecast_trainer = Train_Model(finetune_forecast_model, train_dataset, valid_dataset, test_dataset, num_epochs, batch_size, OUT_DIR,
                                        Train_Type = "Few Shot")

output_file = os.path.join(OUT_DIR, "lat_lon_errors_fewshot.csv")
lat_lon_pred(data, tsp, finetune_forecast_trainer, output_file, context_length, forecast_length)