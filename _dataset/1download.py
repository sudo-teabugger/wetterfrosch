import cdsapi
import multiprocessing as mp
import time as t

def download_anfrage(api_url: str, api_key: str, jahr: str, monat: str, target_path: str) -> None:
    try:
        dataset: str = "reanalysis-era5-single-levels"
        anfrage: dict = {
            "product_type": ["reanalysis"],
            "variable": [
                "10m_u_component_of_wind",  #\ Windgeschwindigkeit und Richtung m/s
                "10m_v_component_of_wind",  #/
                "2m_temperature",           # Temperatur Kelvin
                "sea_surface_temperature",  # Wassertemperatur Kelvin
                "surface_pressure",         # Luftdruck Pascals
                "2m_dewpoint_temperature",  # Luftfeuchtigkeit
                "total_precipitation",      # Niederschlag Meter
                "snow_depth"                # Schneetiefe Meter
                ],
            "year": [jahr],
            "month": [monat],
            "day": [
                "01", "02", "03", "04", "05", "06",
                "07", "08", "09", "10", "11", "12",
                "13", "14", "15", "16", "17", "18",
                "19", "20", "21", "22", "23", "24",
                "25", "26", "27", "28", "29", "30", "31"
                ],
            "time": [
                "00:00", "01:00", "02:00", "03:00", "04:00", "05:00",
                "06:00", "07:00", "08:00", "09:00", "10:00", "11:00",
                "12:00", "13:00", "14:00", "15:00", "16:00", "17:00",
                "18:00", "19:00", "20:00", "21:00", "22:00", "23:00"
                ],
            "data_format": "netcdf",
            "download_format": "zip",
            "area": [72, -25, 34, 45]
            }
        name: str = f"era5_{jahr}_{monat}.zip"
        print(f"\033[91mDownload von {name} angefangen\033[0m")
        client: object = cdsapi.Client(url = api_url, key = api_key, quiet = True)
        client.retrieve(dataset, anfrage).download(target_path + name)
        print(f"\033[92m{name} erfolgreich heruntergeladen\033[0m")
    except:
        t.sleep(1800)
        download_anfrage(api_url = api_url, api_key = api_key, jahr = jahr, monat = monat, target_path = target_path)



if __name__ == "__main__":
    fertig: dict = {
        "1940": [],
        "1941": [],
        "1942": [],
        "1943": [],
        "1944": [],
        "1945": [],
        "1946": [],
        "1947": [],
        "1948": [],
        "1949": [],
        "1950": [],
        "1951": [],
        "1952": [],
        "1953": [],
        "1954": [],
        "1955": [],
        "1956": [],
        "1957": [],
        "1958": [],
        "1959": [],
        "1960": [],
        "1961": [],
        "1962": [],
        "1963": [],
        "1964": [],
        "1965": [],
        "1966": [],
        "1967": [],
        "1968": [],
        "1969": [],
        "1970": [],
        "1971": [],
        "1972": [],
        "1973": [],
        "1974": [],
        "1975": [],
        "1976": [],
        "1977": [],
        "1978": [],
        "1979": [],
        "1980": [],
        "1981": [],
        "1982": [],
        "1983": [],
        "1984": [],
        "1985": [],
        "1986": [],
        "1987": [],
        "1988": [],
        "1989": [],
        "1990": [],
        "1991": [],
        "1992": [],
        "1993": [],
        "1994": [],
        "1995": [],
        "1996": [],
        "1997": [],
        "1998": [],
        "1999": [],
        "2000": [],
        "2001": [],
        "2002": [],
        "2003": [],
        "2004": [],
        "2005": [],
        "2006": [],
        "2007": [],
        "2008": [],
        "2009": [],
        "2010": [],
        "2011": [],
        "2012": [],
        "2013": [],
        "2014": [],
        "2015": [],
        "2016": [],
        "2017": [],
        "2018": [],
        "2019": [],
        "2020": [],
        "2021": [],
        "2022": [],
        "2023": [],
        "2024": [],
        "2025": [],
        }
    jahre: list = [str(year) for year in range(1940, 2024)]
    monate: list = ["01","02","03","04","05","06","07","08","09","10","11","12"]
    url: str = "https://cds.climate.copernicus.eu/api"
    api_keys: list = ["######################", "######################", "######################", "######################", "######################", "######################"]
    amount_api_keys: int = len(api_keys)
    queue_list: list = []
    for _ in range(0, amount_api_keys):
        queue_list.append([])
    index: int = 0
    for jahr in jahre:
        for monat in monate:
            if monat not in fertig[jahr]:
                process = mp.Process(target=download_anfrage, args=(url, api_keys[index], jahr, monat, "downloaded_era5_datasets/"))
                queue_list[index].append(process)
                index += 1
                if index == amount_api_keys:
                    index = 0
    for queue in queue_list:
        queue[0].start()
    while len(queue_list[0]) != 0 and len(queue_list[1]) != 0 and len(queue_list[2]) != 0 and len(queue_list[3]) != 0:
        for i in range(0, amount_api_keys):
            try:
                if queue_list[i][0].is_alive() == False:
                    queue_list[i].pop(0)
                    queue_list[i][0].start()
            except:
                pass