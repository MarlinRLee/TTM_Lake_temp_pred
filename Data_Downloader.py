import os
#import requests
import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime#, timedelta
import time

# Base URL for downloading files
#BASE_URL = "https://apps.glerl.noaa.gov/erddap/files/GLSEA_GCS"

# Output directories
DOWNLOAD_DIR = "./sst_data"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Start and end dates
start_date = datetime(1995, 1, 1)
end_date = datetime(2023, 11, 1)

# Prepare a list of dates
dates = pd.date_range(start=start_date, end=end_date)

All_paths = []
for date in dates:
    year = date.year
    month = f"{date.month:02d}"
    day_of_year = f"{date.timetuple().tm_yday:03d}"
    file_name = f"{year}_{day_of_year}_glsea_sst.nc"
    file_url = ""#f"{BASE_URL}/{year}/{month}/{file_name}"
    local_file_path = os.path.join(DOWNLOAD_DIR, file_name)
    All_paths.append((local_file_path, file_url))

# Download each file
"""
for local_file_path, file_url in All_paths:
    # Download the file if not already downloaded
    if not os.path.exists(local_file_path):
        print(f"Downloading: {file_url}", flush=True)
        for attempt in range(5):  # Retry up to 5 times
            try:
                response = requests.get(file_url, stream=True, timeout=60)
                response.raise_for_status()  # Raise error for HTTP errors
                with open(local_file_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        f.write(chunk)
                print("Download successful", flush=True)
                break
            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt + 1} failed: {e}", flush=True)
                time.sleep(10)  # Wait before retrying
        else:
            print("Download failed after multiple attempts.", flush=True)
        time.sleep(1)
"""
print(f"Start Loading", flush=True)

ML_DATA_DIR = "./sst_ml_data"
os.makedirs(ML_DATA_DIR, exist_ok=True)

#Load all data into one file
all_data = []
for lat in range(0, 83 - 3, 3):
    for lon in range(0, 118 - 3, 3):
        print(f"Start Lat {lat} Long {lon}", flush=True)
        chunk_data = []
        skip = None #Weird logic but it makes sense.
        for local_file_path, file_url in All_paths:
            with xr.open_dataset(local_file_path) as ds:
                resampled = ds.coarsen(lat=10, lon=10, boundary="trim").mean()
                assert resampled.dims['lat'] == 83
                assert resampled.dims['lon'] == 118
                
                # Extract a 3x3 region of the SST data
                square = resampled.isel(time=0, lat=slice(lat, lat + 3), lon=slice(lon, lon + 3))
                spatial_data = square.to_array()[0].values.flatten()

                if skip == None:
                    if np.all(np.isnan(spatial_data.astype(float))):
                        skip = True
                        break
                    else:
                        skip = False
                
                file_time = resampled["time"].values[0]
                # Generate column names with lat/lon coordinates
                lat_values = resampled.lat[lat:lat + 3].values
                lon_values = resampled.lon[lon:lon + 3].values
                column_names = [f"SST_lat{i}_lon{j}" for i in range(3) for j in range(3)]
                chunk_data.append([file_time] + list(spatial_data))
        if skip:
            print(f"Skipping chunk at Lat {lat} Lon {lon} - all values are zero.")
            continue
        df = pd.DataFrame(chunk_data, columns=["Time"] + column_names)
        data_path = os.path.join(ML_DATA_DIR, f"SST_{lat_values[0]:.2f}_{lon_values[0]:.2f}.csv")
        df.to_csv(data_path, index=False)