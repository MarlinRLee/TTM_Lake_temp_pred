import pandas as pd

sst = pd.read_csv("combined_sst_data.csv")
glerl = pd.read_csv("combined_LC_data.csv")

glerl = glerl.rename(columns = {'longitude' : 'Lon', 'latitude' : 'Lat'})

#NA rows with faulty data in glerl
glerl.loc[glerl["lake_depth"] > 0, ["lake_depth", "ice_cover", "water_level"]] = pd.NA

#Average glerl data over 3x3 chunks used in sst data
glerl["Lat_bin"] = glerl["Lat"].where(glerl["Lat"].isin(sst["Lat"]), other = pd.NA)
glerl["Lat_bin"] = glerl["Lat_bin"].ffill()

glerl["Lon_bin"] = glerl["Lon"].where(glerl["Lon"].isin(sst["Lon"]), other = pd.NA)
glerl["Lon_bin"] = glerl["Lon_bin"].ffill()

glerl = (glerl.drop(columns = ["Lat", "Lon"])
    .groupby(["Lat_bin", "Lon_bin", "Time"])
    .mean()
    .reset_index()
    .rename(columns = {"Lat_bin" : "Lat", "Lon_bin":"Lon"})
)

combined_data = (pd.merge(sst, glerl, how = 'left', on = ['Time', 'Lat', 'Lon'])
    .dropna(axis = 'index', subset = ["SST_lat0_lon0","SST_lat0_lon1","SST_lat0_lon2","SST_lat1_lon0","SST_lat1_lon1","SST_lat1_lon2","SST_lat2_lon0","SST_lat2_lon1","SST_lat2_lon2"], thresh = 7))

def separate_lakes(x,y):
    if x < -84.35 and y > 46.4:
        return 's'
    elif x < -84.73 and y < 46.14:
        return 'm'
    elif x > -83.56 and y < 42.93:
        return 'e'
    elif x > -79.96 and y < 44.35:
        return 'o'
    else:
        return 'h'

combined_data["Lake"] = pd.Series([separate_lakes(long, lat) for long, lat in combined_data[["Lon", "Lat"]].itertuples(index = False)])

combined_data.to_csv("combined_data_w_lakes.csv")
