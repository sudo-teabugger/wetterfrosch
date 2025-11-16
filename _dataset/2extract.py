import os
import zipfile

def extract(source_path: str, element_name: str, target_path: str) -> None:
    extracted_name: str = element_name.replace(".zip", "/")
    with zipfile.ZipFile(source_path + element_name, "r") as zip_ref:
        zip_ref.extractall(os.path.join(target_path, extracted_name))

if __name__ == "__main__":
    zip_files: list = os.listdir("downloaded_era5_datasets/")
    for zip_file in zip_files:
        extract(source_path="downloaded_era5_datasets/", element_name=zip_file, target_path="extracted_era5_datasets")
        print(f"{zip_file} erfolgreich extrahiert")