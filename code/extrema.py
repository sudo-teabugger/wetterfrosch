import pandas as pd
from os import listdir
import json

def collect_maximum(path: str) -> dict:
    extrema: dict = {
        "jahr": [20000, 0],
        "monat": [20000, 0],
        "tag": [20000, 0],
        "stunde": [20000, 0],
        "längengrad": [20000, 0],
        "breitengrad": [20000, 0],
        "höhe": [20000, 0],
        "lufttemperatur": [20000, 0],
        "luftdruck": [20000, 0],
        "luftfeuchtigkeit": [20000, 0],
        "niederschlag": [20000, 0],
        "windrichtung": [20000, 0],
        "windgeschwindigkeit": [20000, 0],
        "schneetiefe": [20000, 0]
    }
    for file in listdir(path = path):
        df = pd.read_parquet(path + file)
        for key in extrema:
            pot_min = df[key].min()
            if pot_min < extrema[key][0]:
                extrema[key][0] = float(pot_min)
        for key in extrema:
            pot_max = df[key].max()
            if pot_max > extrema[key][1]:
                extrema[key][1] = float(pot_max)
        print(f"Aus der Datei {file} werden alle Maxima extrahiert...")
    return extrema

if __name__ == "__main__":
    extrema = collect_maximum(path = "parquet_altitude_era5_datasets/")

    with open("extrema.json", "w") as file:
        json.dump([extrema], file)
