from os import listdir
import pandas as pd
import random as r

def split(parquet_file: str) -> None:
    train_df = pd.DataFrame(parquet_file)
    
    len_train_df: int = len(train_df)
    portion_test_df: int = len_train_df / 5
    test_indexes: list = []
    pool: list = [i for i in range(0, len_train_df)]
    for _ in range(0,portion_test_df):
        index: int = r.choice(pool)
        test_indexes.append(index)
        pool.remove(index)
    test_df = pd.DataFrame()
    test_df = pd.concat(test_df, train_df[test_indexes])
    train_df.drop(train_df.index[test_indexes])

    len_test_df: int = len(test_df)
    portion_val_df: int = len_test_df / 2
    val_indexes: list = []
    pool: list = [i for i in range(0, len_test_df)]
    for _ in range(0,portion_val_df):
        index: int = r.choice(pool)
        val_indexes.append(index)
        pool.remove(index)
    val_df = pd.DataFrame()
    val_df = pd.concat(val_df, test_df[val_indexes])
    test_df.drop(test_df.index[val_indexes])

    train_df.to_parquet("TRAIN/" + "train/" + parquet_file, index = False, engine = "pyarrow", compression = "snappy")
    test_df.to_parquet("TEST/" + "test/" + parquet_file, index = False, engine = "pyarrow", compression = "snappy")
    val_df.to_parquet("VALIDATION/" + "validation/" + parquet_file, index = False, engine = "pyarrow", compression = "snappy")
    print(f"{parquet_file} erfolgreich in Test-, Train- und Validation-Dataset geteilt")

if __name__ == "__main__":
    parquets: list = listdir("parquet_altitude_era5_datasets")
    for file in parquets:
        split(parquet_file=file)