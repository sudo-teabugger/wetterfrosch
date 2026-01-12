# Küntsliche Intelligenz in der Meteorologie
## Installation
### Python virtuelle Umgebung (optional)
~~~
python -m venv "wetterfrosch_venv"
cd wetterfrosch_venv/
.\Scripts\activate
~~~
### Installation des Reposiories und der Dependencies
~~~
git clone https://github.com/sudo-teabugger/wetterfrosch.git
cd Code/
pip install -r requirements.txt
~~~
## Dokumentation
Alle Dateien mit einer Zahl als erstes Zeichen des Dateinamen dienen zur Datenvorbereitung. Sind die Daten vorbereitet, so kann mit `NN_training.py` ein neuronales Netzwerk trainiert werden. Die exportierten Parameter können daraufhin erst in `NN_test.py` importiert werden, um die angepassten Parameter mit einem Testdatensatz zu bewerten oder auch direkt durch die Ausführung von `NN_benutzer.py` ausprobiert werden.
### `1download.py`
lädt unter Verwendung der API des Climate Data Storage (https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview) die Daten monatsweise in Zip-Dateien herunter, die jeweils eine ungefähre Größe von 300MB haben und jeweils zwei .nc Dateien beinhalten.
### `1download.py`
...
