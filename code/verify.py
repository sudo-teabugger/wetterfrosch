import os
import zipfile

def verify_zip(zip_path: str) -> bool:
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            return zf.testzip() is None
    except zipfile.BadZipFile:
        return False
    
if __name__ == "__main__":
    fehlerhafte_zipfile: list = []
    zip_files: list = os.listdir("downloaded_era5_datasets/")
    for zip_file in zip_files:
        zip_status = verify_zip(zip_path = "downloaded_era5_datasets/" + zip_file)
        if zip_status == True:
            print(f"\033[92m{zip_file}\033[0m")
        else:
            print(f"\033[91m{zip_file}\033[0m")
            fehlerhafte_zipfile.append(zip_file)
    
    if len(fehlerhafte_zipfile) != 0:
        for name in fehlerhafte_zipfile:
            print(f"\t- \033[91m{name}\033[0m")