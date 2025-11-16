import os
import xarray as xr
import pandas as pd

def convert(source_path: str, result_path: str, result_name: str) -> None:
    file_pair: list = os.listdir(source_path)
    temp: list = [xr.open_dataset(source_path + "/" + file, engine="netcdf4") for file in file_pair]
    ds_merged = xr.merge(temp, compat="override", join="outer")
    df = ds_merged.to_dataframe().reset_index()

    df.drop(columns=["number", "expver"], errors="ignore", inplace=True)
    df.rename(columns={
        "u10": "ostwind",
        "v10": "nordwind",
        "t2m": "lufttemperatur",
        "sst": "wassertemperatur",
        "sp": "luftdruck",
        "d2m": "luftfeuchtigkeit",
        "sd": "schneetiefe",
        "tp": "niederschlag"
    }, inplace=True)

    df["niederschlag"] = df["niederschlag"].fillna(0.0)
    df["wassertemperatur"] = df["wassertemperatur"].ffill().bfill()

    df["valid_time"] = pd.to_datetime(df["valid_time"])
    df["year"] = df["valid_time"].dt.year
    df["month"] = df["valid_time"].dt.month
    df["day"] = df["valid_time"].dt.day
    df["hour"] = df["valid_time"].dt.hour
    df.drop(columns=["valid_time"], inplace=True)

    df.to_parquet(result_path + "/" + result_name, index = False, engine = "pyarrow", compression = "snappy")
    print(f"{result_name} erfolgreich abgespeichert")

if __name__ == "__main__":
    extracted_folder = os.listdir("extracted_era5_datasets")
    for folder in extracted_folder:
        convert(source_path="extracted_era5_datasets/"+folder, result_path="parquet_era5_datasets", result_name=folder+".parquet")