import os
import pandas as pd
from löschen import rekursives_löschen

koordinaten_df = pd.read_parquet(path = "unique_coords_with_altitude.parquet")
datei_liste: list = os.listdir(path = "parquet_era5_datasets/")

for datei in datei_liste:
    df = pd.read_parquet(path = f"parquet_era5_datasets/{datei}")
    df = df.merge(right = koordinaten_df,
                  on = ["längengrad", "breitengrad"],
                  how = "left")
    df.to_parquet(path = f"parquet_altitude_era5_datasets/{datei}", engine = "pyarrow")
    print(f"\033[92mDer Datei {datei} würden erfolgreich die Höhendaten hinzugefügt\033[0m")
    rekursives_löschen(datei = "parquet_era5_datasets/" + datei)