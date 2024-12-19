import os
from util import Train_Model, load_data, lat_lon_pred, load_model, load_tsp, prep_env
from tsfm_public import get_datasets
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

device = prep_env()
context_length = 512
forecast_length = 96
num_epochs = 30 
batch_size = 128

# Load data and TSP
data = load_data()
tsp = load_tsp(context_length, forecast_length)

# Get unique lakes
unique_lakes = data["Lake"].unique()
train_split_params = {"train": .75, "test": .05}
test_split_params = {"train": .1, "test": .8}

# Results files tracking
results_files = []

# Loop through each lake and training mode
for lake in unique_lakes:
    OUT_DIR = f"ttm_finetuned_models/temp_exclude_{lake}/"
    os.makedirs(OUT_DIR, exist_ok=True)
    
    train_data = data[data["Lake"] != lake]
    test_data = data[data["Lake"] == lake]
    
    print(f"\nTraining excluding lake {lake}")
    print(f"Train data size: {len(train_data)}")
    print(f"Test data size: {len(test_data)}")
    
    train_dataset, valid_dataset, _ = get_datasets(
        tsp,
        train_data,
        train_split_params,
    )
    _, _, test_dataset = get_datasets(
        tsp,
        test_data,
        test_split_params,
    )
    
    finetune_forecast_model = load_model(tsp, context_length, forecast_length, device, isTraining=True)
    
    finetune_forecast_trainer = Train_Model(
        finetune_forecast_model, 
        train_dataset, 
        valid_dataset, 
        test_dataset, 
        num_epochs, 
        batch_size, 
        OUT_DIR,
        Train_Type=f"Exclude {lake}"
    )
    
    output_file = os.path.join(OUT_DIR, f"lat_lon_errors_finetune_{lake}_exclude.csv")
    lat_lon_pred(data, tsp, finetune_forecast_trainer, output_file, context_length, forecast_length)
    results_files.append(output_file)

print("\nTraining completed. Results files:")
for file in results_files:
    print(f"- {file}")