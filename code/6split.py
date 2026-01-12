from os import listdir
import pandas as pd
from löschen import rekursives_löschen

def split(parquet_file: str) -> None:
    df = pd.read_parquet("parquet_altitude_era5_datasets/" + parquet_file)

    df = df.sample(frac = 1.0).reset_index(drop = True)

    n = len(df)
    n_test = n // 5 # 20%


    test_df = df.iloc[:n_test]
    train_df = df.iloc[n_test:]

    train_df.to_parquet(
        "TRAIN/train-" + parquet_file,
        index=False,
        engine="pyarrow",
        compression="snappy"
    )
    test_df.to_parquet(
        "TEST/test-" + parquet_file,
        index=False,
        engine="pyarrow",
        compression="snappy"
    )

    print(f"\033[92m{parquet_file} erfolgreich geteilt\033[0m")

if __name__ == "__main__":
    parquets = listdir("parquet_altitude_era5_datasets/")
    for file in parquets:
        split(parquet_file = file)
        rekursives_löschen(datei = "parquet_altitude_era5_datasets/" + file)