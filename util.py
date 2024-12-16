import pandas as pd
import os
import torch
from transformers import set_seed
from tsfm_public import (
    TimeSeriesPreprocessor,
    get_datasets,
)
from tsfm_public.toolkit.get_model import get_model
from tsfm_public import count_parameters, TrackingCallback
from tsfm_public.toolkit.lr_finder import optimal_lr_finder
from tsfm_public.toolkit.visualization import plot_predictions

from torch.optim.lr_scheduler import OneCycleLR
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed
from transformers.integrations import INTEGRATION_TO_CALLBACK
from torch.optim import AdamW
from math import ceil

def prep_env():
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #Set seed for reproducibility
    SEED = 42
    set_seed(SEED)

def load_data(DATA_Path = "/scratch.global/csci8523_group_7/combined_data_w_lakes.csv", 
            timestamp_column = "Time"):
    
    data = pd.read_csv(
        DATA_Path,
        parse_dates=[timestamp_column],
    )


    data[timestamp_column] = pd.to_datetime(data[timestamp_column])
    data = data.sort_values(by=timestamp_column).reset_index(drop=True)
    print(data.head())
    return data

def load_tsp(context_length, 
            forecast_length,
            timestamp_column = "Time", 
            id_columns = ["Lat", "Lon"], 
            target_columns = [
                                "SST_lat0_lon0", "SST_lat0_lon1", "SST_lat0_lon2",
                                "SST_lat1_lon0", "SST_lat1_lon1", "SST_lat1_lon2",
                                "SST_lat2_lon0", "SST_lat2_lon1", "SST_lat2_lon2",
                            ], 
            control_columns = ["lake_depth", "ice_cover", "water_level"]):
    
    column_specifiers = {
        "timestamp_column": timestamp_column,
        "id_columns": id_columns,
        "target_columns": target_columns,
        "control_columns": control_columns,
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
    return tsp

def load_model(tsp, 
               context_length, 
               forecast_length, 
               device,
               TTM_MODEL_PATH = "ibm-granite/granite-timeseries-ttm-r1",
               isTraining = False,
               verbose = False):
    if isTraining:
        # Load model
        loaded_model = get_model(
            TTM_MODEL_PATH,
            context_length=context_length,
            prediction_length=forecast_length,
            prediction_channel_indices = tsp.prediction_channel_indices,
            num_input_channels=tsp.num_input_channels,
            
            
            decoder_mode="mix_channel",  # ch_mix:  set to mix_channel for mixing channels in history
            exogenous_channel_indices=tsp.exogenous_channel_indices,
            fcm_use_mixer=True,  # exog: Try true (1st option) or false
            fcm_mix_layers=2,  # exog: Number of layers for exog mixing
            enable_forecast_channel_mixing=True,  # exog: set true for exog mixing
            #fcm_context_length=1,  # exog: indicates lag length to use in the exog fusion. for Ex. if today sales can get affected by discount on +/- 2 days, mention 2
            #fcm_prepend_past=False,  # exog: set true to include lag from history during exog infusion.
        )
        if verbose:
            print(
                "Number of params before freezing backbone",
                count_parameters(loaded_model),
            )

        # Freeze the backbone of the model
        for param in loaded_model.backbone.parameters():
            param.requires_grad = False
            
        if verbose:
            # Count params
            print(
                "Number of params after freezing the backbone",
                count_parameters(loaded_model),
            )
        
        
    else:
        loaded_model = get_model(
            TTM_MODEL_PATH,
            context_length=context_length,
            prediction_length=forecast_length,
            prediction_channel_indices=tsp.prediction_channel_indices,
            num_input_channels=tsp.num_input_channels,
        )
    # Move the model to GPU
    loaded_model.to(device)
    return loaded_model
    

    


def Train_Model(finetune_forecast_model, train_dataset, valid_dataset, test_dataset, num_epochs, batch_size, OUT_DIR, Train_Type):
    learning_rate, finetune_forecast_model = optimal_lr_finder(
        finetune_forecast_model,
        train_dataset,
        batch_size=batch_size,
        enable_prefix_tuning=False,
    )
    print("OPTIMAL SUGGESTED LEARNING RATE =", learning_rate)


    print(f"Using learning rate = {learning_rate}")
    finetune_forecast_args = TrainingArguments(
        output_dir=os.path.join(OUT_DIR, "output"),
        overwrite_output_dir=True,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        do_eval=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=8,
        report_to=None,
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=1,
        logging_dir=os.path.join(OUT_DIR, "logs"),  # Make sure to specify a logging directory
        load_best_model_at_end=True,  # Load the best model when training ends
        metric_for_best_model="eval_loss",  # Metric to monitor for early stopping
        greater_is_better=False,  # For loss
    )

    # Create the early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience = 5,  # Number of epochs with no improvement after which to stop
        early_stopping_threshold = 0.0,  # Minimum improvement required to consider as improvement
    )
    tracking_callback = TrackingCallback()

    # Optimizer and scheduler
    optimizer = AdamW(finetune_forecast_model.parameters(), lr=learning_rate)
    scheduler = OneCycleLR(
        optimizer,
        learning_rate,
        epochs=num_epochs,
        steps_per_epoch = ceil(len(train_dataset) / (batch_size)),
    )

    finetune_forecast_trainer = Trainer(
        model=finetune_forecast_model,
        args=finetune_forecast_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        callbacks=[early_stopping_callback, tracking_callback],
        optimizers=(optimizer, scheduler),
    )

    # Fine tune
    finetune_forecast_trainer.train()

    finetune_output = finetune_forecast_trainer.evaluate(test_dataset)
    print(finetune_output, flush = True)

    plot_predictions(
        model=finetune_forecast_trainer.model,
        dset=test_dataset,
        plot_dir=os.path.join(OUT_DIR, "SST_Preds"),
        plot_prefix = Train_Type,
        channel=0,
    )
    return finetune_forecast_trainer


def lat_lon_pred(data, tsp, trainer, out_file, context_length, forecast_length):
    # Extract unique lat-lon pairs
    lat_lon_pairs = data[["Lat", "Lon"]].drop_duplicates().reset_index(drop=True)

    # DataFrame to store results
    results = []

    # Iterate over each latitude-longitude pair
    for _, row in lat_lon_pairs.iterrows():
        lat = row["Lat"]
        lon = row["Lon"]
        
        # Subset the test dataset for this lat-lon pair
        subset = data[(data["Lat"] == lat) & (data["Lon"] == lon)]
        
        if subset.empty or len(subset) < 10 * (context_length + forecast_length):
            continue  # Skip if no data for this pair
        
        print(f"{len(subset)}    {lat} - {lon}", flush = True)
        # Prepare datasets for evaluation
        split_params = {"train": .2, "test": .7}
        _, _, test_dataset = get_datasets(
            tsp,
            subset,
            split_params,
        )
        
        # Evaluate model on the subset
        eval_output = trainer.evaluate(test_dataset)
        
        # Store the results
        results.append({
            "Latitude": lat,
            "Longitude": lon,
            "Error": eval_output["eval_loss"],  # Replace with appropriate metric key
        })

    # Save results to a CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(out_file, index=False)
    print(f"Error data saved to {out_file}")
