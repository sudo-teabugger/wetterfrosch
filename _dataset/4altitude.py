from os import listdir
import pandas as pd
from requests import get
from json import loads

def list_coordinates():
    unique_coords_df = pd.DataFrame()
    df_paths = listdir("parquet_era5_datasets/")
    for df_path in df_paths:
        df = pd.read_parquet("parquet_era5_datasets/" + df_path)
        df_coords = df[["latitude", "longitude"]].drop_duplicates()
        unique_coords_df = pd.concat(objs=[unique_coords_df, df_coords], axis=1).drop_duplicates()
        print(f"Aus {df_path} wurden alle eindeutigen Elemente kopiert...")
    unique_coords_df.to_parquet("list_unique_coords.parquet")
    print("Ein Dataframe mit allen Koordinaten der Messungsstandorte wurde erstellt...")
    return unique_coords_df

def fetch_altitudes(coords_df):
    for index, row in coords_df.iterrows():
        url = f"https://api.open-meteo.com/v1/elevation?latitude={row["latitude"]}&longitude={row["longitude"]}"
        response = get(url)
        data = loads(response.text)
        elevation = int(round(list(data['elevation'])[0]))
        coords_df.loc[index, "altitude"] = elevation
        print(f"{index}: altitude = {elevation}m")

if __name__ == "__main__":
    unique_coords_df = list_coordinates()
    fetch_altitudes(coords_df=unique_coords_df)
