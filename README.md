an GE (die Person wird wissen, wenn sie gemeint ist): Die aktuell noch Englischen Bezeichnungen für Variablen, Funtionen, ... werde ich, falls sie es für sinnvoll halten, noch ins Deutsche übersetzen :)

# Küntsliche Intelligenz in der Meteorologie
## Installation
### Python virtuelle Umgebung erstellen (optional)
~~~
python -m venv "wetterfrosch_venv"
cd wetterfrosch_venv/
.\Scripts\activate
~~~
### Installation des Reposiories und der Dependencies
~~~
git clone https://github.com/sudo-teabugger/wetterfrosch.git
cd code/
pip install -r requirements.txt
~~~
Löschen sie jede `DELETE ME` Datei in den Unterordnern von `code`, diese dienen lediglich dazu, dass github diese Ordner nicht ignoriert.
## Dokumentation
Alle Dateien mit einer Zahl als erstes Zeichen des Dateinamen dienen zur Datenvorbereitung. Sind die Daten vorbereitet, so kann mit `NN_training.py` ein neuronales Netzwerk trainiert werden. Die exportierten Parameter können daraufhin erst in `NN_test.py` importiert werden, um die angepassten Parameter mit einem Testdatensatz zu bewerten oder auch direkt durch die Ausführung von `NN_benutzer.py` ausprobiert werden.
### `1download.py`
lädt unter Verwendung der API des Climate Data Storage (https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview) die Daten monatsweise in Zip-Dateien herunter, die jeweils eine ungefähre Größe von 300MB haben und jeweils zwei .nc Dateien beinhalten und in `downloaded_era5_datasets/` zu finden sind.
### `2extract.py`
extrahiert die heruntergeladenen Zip-Dateien in den Ordner `extracted_era5_datasets/`.
### `3convert.py`
wandelt die extrahierten NetCDF-Dateien in Parquet-Dataframes um, rechnet Einheiten um, löscht nicht benötigte Spalten und speichert die Dataframes unter `parquet_era5_datasets/`.
### `4altitude_api_requests.py`
kopiert alle einzigartigen Koordinatenpaare aus dem gesamten Datensatz heraus und speichert diese als `unique_coords.parquet` ab. Daraufhin wird unter Verwendung der GPXZ-API (https://www.gpxz.io/) für jedes Koordinatenpaar die Höhe des Bodens über dem Meeresspiegel ermittelt. Wurde für jedes Koordinatenpaar eine Höhe ermittelt, so wird diese in `unique_coords_with_altitude.parquet` abgespeichert.
### `5add_altitude_to_files.py`
wiederrum liest `unique_coords_with_altitude.parquet` aus und fügt jeder Zeile des gesamten Datensatzes die Höhendaten hinzu und speichert diese in `parquet_altitude_era5_datasets/`.
### `6split.py`
teilt alle Parquet-Dateien nach einem festgelegten Verhältnis, jedoch zufällig, in einen Trainingsdatensatz, abgespeichert unter `TRAIN/`, beziehungsweise einen Testdatensatz, abgespeichert unter `TEST/`, auf.
### `7shuffle_training_data.py`
fügt zuerst jeder Zeile einen zufälligen Index hinzu und sortiert daraufhin nach diesem den Datensatz. Danach wird die Reihenfolge der Zeilen in jeder erzeugten Dateien in `TRAIN_shuffled/` nochmals zufällig verendert, sodass ein komplett ungeordneter Trainingsdatensatz entsteht.
### `NN_training.py`
trainiert anhand des vorbereiteten Datensatzes ein neurales Netzwerk und exportiert anschließend die Weights un Biases in `parameter.json`. Während des Trainings werden konstant Parquet-Dateien, welche den Loss jedes Durchgangs beinhalten, unter `TRAIN_loss/` abgespeichert.
### `NN_test.py`
importiert die zuvor entwickelten Parameter und testet diese. Der Entstehende Loss wird auch hier diesmal unter `TEST_loss/` abgespeichert.
### `NN_benutzer.py`
dient primär dazu, das zuvor trainierte und getestete neurale Netzwerk für Vorhersagen zu nutzen.
### `extrema.py`
extrahiert aus dem gesamten Datensatz die Maxima und Minima jeder Spalte und speichert diese als `extrema.json` ab, um später die Eingabefeatures des neuralen Netzwerks zu skaliern und die Ausgabefeatures zurückzuskalieren.
### `verify.py`
dient dazu, die heruntergeladenen Zip-Dateien auf integrität zu überprüfen
### `löschen.py`
enthält die Methode `rekursives_löschen(datei: str)`, welche für den zuvor beschriebenen Datenvorbereitungsprozess in mehreren Python-Dateien genutzt wird, um Hinterlassenschaften des Prozesses zu beseitigen.