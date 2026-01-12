import os
import zipfile
from löschen import rekursives_löschen

def extract(source_path: str, element_name: str, target_path: str) -> None:
    extracted_name: str = element_name.replace(".zip", "/")
    with zipfile.ZipFile(source_path + element_name, "r") as zip_ref:
        zip_ref.extractall(os.path.join(target_path, extracted_name))

if __name__ == "__main__":
    fehlerhafte_zipfile: list = []
    zip_files: list = os.listdir("downloaded_era5_datasets/")
    for zip_file in zip_files:
        try:
            extract(source_path = "downloaded_era5_datasets/",
                element_name = zip_file,
                target_path = "extracted_era5_datasets")
            print(f"\033[92m{zip_file} erfolgreich extrahiert\033[0m")

        except:
            print(f"\033[91mBeim Extrahieren von {zip_file} ist ein Fehler aufgetreten\033[0m")
            fehlerhafte_zipfile.append(zip_file)
        rekursives_löschen(datei = "downloaded_era5_datasets/" + zip_file)
        
    
    if len(fehlerhafte_zipfile) != 0:
        print("\033[91m\n\nBei folgenden Zipfiles ist beim Extrahieren ein Fehler aufgetreten:\033[0m")
        for name in fehlerhafte_zipfile:
            print(f"\t- \033[91m{name}\033[0m")