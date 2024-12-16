import pandas as pd

sst = pd.read_csv("combined_sst_data.csv")
glerl = pd.read_csv("combined_LC_data.csv")

combined_data = pd.merge(sst, glerl, how = 'left', on = ['Time', 'Lat', 'Lon'] )
combined_data.to_csv("sst_glerl_complete_data.csv",index = False)