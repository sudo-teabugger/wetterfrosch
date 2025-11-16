import os

def recursive_delete(item: str) -> None:
    try:
        os.remove(item)
    except PermissionError:
        try:
            os.rmdir(item)
        except:
            for element in os.listdir(item):
                recursive_delete(item = item + element)
            try:
                os.remove(item)
            except PermissionError:
                os.rmdir(item)

if __name__ == "__main__":
    for file in os.listdir("downloaded_era5_datasets"):
        recursive_delete(file)
    for file in os.listdir("extracted_era5_datasets"):
        recursive_delete(file)
    for file in os.listdir("parquet_era5_datasets"):
        recursive_delete(file)
    for file in os.listdir("parquet_altitude_era5_datasets"):
        recursive_delete(file)