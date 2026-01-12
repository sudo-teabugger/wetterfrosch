import pandas as pd
import numpy as np
from os import listdir
from löschen import rekursives_löschen

anzahl_entstehende_dateien: int = 650 # da Datensatz ungefähr 650GB groß ist und daher ~ 1GB große Dateien entstehen

# Den Datensätzen wir eine zusäatzliche Spalte mit Zahlen von 0 bis anzahl_entstehende_dateien hinzugefügt, nach denen dieser später aufgeteilt wird
for dateiname in listdir(path = "TRAIN/"):
    df = pd.read_parquet(path = "TRAIN/" + dateiname)
    df["index_for_shuffle"] = np.random.randint(low = 0, high = anzahl_entstehende_dateien, size = len(df))
    df.to_parquet(path = f"TRAIN_shuffled/shuffled_{dateiname}", engine = "pyarrow")
    print(f"\033[92m{dateiname} wurden erfolgreich shuffled Indizes hinzugefügt\033[0m")
    rekursives_löschen(datei = "TRAIN/" + dateiname)
print(f"\n\033[92mAllen Dateien wurden erfolgreich shuffled Indizes hinzugefügt\033[0m\n")

# Der Datensatz wird anhand der hinzugefügten Indizes auf verschiedene Dateien aufgeteilt
parquets_mit_shuffle_index = listdir(path = "TRAIN_shuffled/")
for i in range(0, anzahl_entstehende_dateien):
    df_liste: list = []
    for dateiname in parquets_mit_shuffle_index:
        df = pd.read_parquet(path = "TRAIN_shuffled/" + dateiname)
        ausgewählte_zeilen = df[df["index_for_shuffle"] == i]
        df_liste.append(ausgewählte_zeilen)
        print(f"\033[92mAus {dateiname} wurden alle Zeilen mit dem Shuffle-Index {i} extrahiert\033[0m")
    df = pd.concat(objs = df_liste, ignore_index = True)
    df.to_parquet(path = "TRAIN_shuffled/" + f"shuffled_{i}.parquet")
    df_liste: list = []
    print(f"\n\033[92mAlle Zeilen mit dem Shuffle-Index {i} wurden abgespeichert\033[0m\n")

for parquet in parquets_mit_shuffle_index:
    rekursives_löschen(datei = "TRAIN_shuffled/" + parquet)    

# Die Reihenfolge der Zeilen jeder entstandenen Datei wird zufällig verändert 
for dateiname in listdir(path = "TRAIN_shuffled/"):
    df = pd.read_parquet(path = "TRAIN_shuffled/" + dateiname)
    df.drop(labels = ["index_for_shuffle"], inplace = True)
    df = df.sample(n = len(df), ignore_index = True)
    df.to_parquet(path = "TRAIN_shuffled/" + dateiname, engine = "pyarrow", index = False)