import os
import xarray as xr
import pandas as pd
import numpy as np
from löschen import rekursives_löschen

def convert(source_path: str, result_path: str, result_name: str) -> None:
    file_pair: list = os.listdir(path = source_path)
    temp: list = [xr.open_dataset(source_path + "/" + file, engine="netcdf4") for file in file_pair]
    ds_merged = xr.merge(temp, compat="override", join="outer")
    df = ds_merged.to_dataframe().reset_index()

    df["tp"].fillna(value = 0.0, inplace = True)
    df["sd"].fillna(value = 0.0, inplace = True)

    df["valid_time"] = pd.to_datetime(df["valid_time"])
    df["jahr"] = df["valid_time"].dt.year
    df["monat"] = df["valid_time"].dt.month
    df["tag"] = df["valid_time"].dt.day
    df["stunde"] = df["valid_time"].dt.hour

    ostwind = df['u10']
    nordwind = df['v10']
    df['windgeschwindigkeit'] = np.sqrt(ostwind ** 2 + nordwind ** 2)
    df['windrichtung'] = (np.degrees(np.arctan2(ostwind, nordwind)) + 180) % 360 # Windrichtung (meteorologisch, 0° = Norden, 90° = Osten)

    df["lufttemperatur"] = df["t2m"] - 273.15 # Kelvin to Celsius

    df.drop(columns = ["t2m", "sst", "number", "expver", "u10", "v10", "valid_time"], errors = "ignore", inplace = True)
    df.rename(columns = {"sp": "luftdruck",
                         "d2m": "luftfeuchtigkeit",
                         "sd": "schneetiefe",
                         "tp": "niederschlag",
                         "latitude": "längengrad",
                         "longitude": "breitengrad"},
              inplace = True)

    df.to_parquet(path = result_path + "/" + result_name, engine = "pyarrow")
    print(f"\033[92m{result_name} erfolgreich abgespeichert\033[0m")

if __name__ == "__main__":
    extracted_folder = os.listdir(path = "extracted_era5_datasets/")
    for folder in extracted_folder:
        convert(source_path = "extracted_era5_datasets/" + folder, 
                result_path = "parquet_era5_datasets", 
                result_name = folder + ".parquet")
    for folder in extracted_folder:
        rekursives_löschen(datei = folder)