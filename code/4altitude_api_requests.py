from os import listdir
import pandas as pd
from requests import get
from json import loads
import time as t

def einzigartige_koordinate() -> object:
    unique_coords_list: list = []
    df_paths = listdir("parquet_era5_datasets/")
    for df_path in df_paths:
        df = pd.read_parquet("parquet_era5_datasets/" + df_path, columns = ["längengrad", "breitengrad"])
        df.drop_duplicates(ignore_index = True, inplace = True)
        unique_coords_list.append(df)
        print(f"Aus {df_path} wurden alle eindeutigen Elemente kopiert...")
    unique_coords_df = pd.concat(objs = unique_coords_list)
    unique_coords_df.drop_duplicates(ignore_index = True, inplace = True)

    if "unique_coords.parquet" in listdir():
        df = pd.read_parquet(path = "unique_coords.parquet").sort_values(by = ["längengrad", "breitengrad"])
        if unique_coords_df.sort_values(by = ["längengrad", "breitengrad"]).equals(df) == True:
            return df
        else:
            unique_coords_df.to_parquet("unique_coords.parquet", engine = "pyarrow")
        print("\033[92mEin Dataframe mit allen Koordinaten der Messungsstandorte wurde erstellt...\033[0m")
        return unique_coords_df
    else:
        unique_coords_df.to_parquet("unique_coords.parquet", engine = "pyarrow")
        print("\033[92mEin Dataframe mit allen Koordinaten der Messungsstandorte wurde erstellt...\033[0m")
        return unique_coords_df

if __name__ == "__main__":
    koordinate_df = einzigartige_koordinate()
    gruppierter_koordinaten_df = koordinate_df.groupby(by = ["längengrad"])
    gruppenlängen: list = gruppierter_koordinaten_df.size().tolist()
    api_key_list: list = ["YOUR_API_KEYS_HERE",
                          "YOUR_API_KEYS_HERE",
                          "YOUR_API_KEYS_HERE",
                          "YOUR_API_KEYS_HERE",
                          "YOUR_API_KEYS_HERE",
                          "YOUR_API_KEYS_HERE",
                          "YOUR_API_KEYS_HERE",
                          "YOUR_API_KEYS_HERE",
                          "YOUR_API_KEYS_HERE",
                          "YOUR_API_KEYS_HERE"]
    i: int = 1
    koordinaten_höhe_dict: dict = {"längengrad": [],
                              "breitengrad": [],
                              "höhe": []}

    if len(api_key_list) > 0:
        for _, group in gruppierter_koordinaten_df:
            url: str = "https://api.gpxz.io/v1/elevation/points?latlons="
            for _, row in group.iterrows():
                url = url + f"{row["längengrad"]},{row["breitengrad"]}|"

            antwort = get(url = url[:-1],
                          headers = {"x-api-key": api_key_list[i % (len(api_key_list) - 1)]}).json()

            for daten_paar in antwort["results"]:
                koordinaten_höhe_dict["längengrad"].append(daten_paar["lat"])
                koordinaten_höhe_dict["breitengrad"].append(daten_paar["lon"])
                koordinaten_höhe_dict["höhe"].append(daten_paar["elevation"]) 

            i += i        
            t.sleep(1)

        koordinaten_höhe = pd.DataFrame(data = koordinaten_höhe_dict)
        koordinaten_höhe.to_parquet(path = "unique_coords_with_altitude.parquet", engine = "pyarrow")
    else:
        print("\033[91mDas Script benötigt API-Keys für https://www.gpxz.io/\033[0m")