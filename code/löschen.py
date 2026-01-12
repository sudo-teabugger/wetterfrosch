import os

def rekursives_löschen(datei: str) -> None:
    try:
        os.remove(datei) # löschen, falls Datei
    except PermissionError:
        try:
            os.rmdir(datei) # löschen, falls leerer Ordner
        except:
            for element in os.listdir(datei):
                rekursives_löschen(datei = datei + "/" + element) # löschen aller Dateien im Ordner
            os.rmdir(datei) # löschen des nun leeren Ordners