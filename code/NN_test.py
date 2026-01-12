import numpy as np
from os import listdir
import pandas as pd
import json

input_features: int = 7
neuron_layout: dict = {
    "hidden_1": 10,
    "hidden_2": 10,
    "hidden_3": 10,
    "output": 7
}
batch_size: int = 100
durchgang: int = 0

class Layer:
    def __init__(self, 
                 anzahl_neuronen: int, 
                 activation_function: bool = False):
        self.anzahl_neuronen = anzahl_neuronen
        self.weights = 0
        self.bias = 0
        self.activation_function = activation_function
    def relu(self):
        '''
        Leaky ReLU:
        if x > 0 -> x
        if x <= 0 -> x * 0.01
        '''
        self.output_vor_activation = self.output # Voraktivierungsoutput für Backpropagation
        self.output = np.where(self.output > 0, self.output, 0.01 * self.output)
    def mse_loss(self, richtiges_y):
        '''
        Durchschnitt von Fehler^2
        '''
        self.loss_per_output = np.mean(a = (np.array(object = richtiges_y) - np.array(object = self.output)) ** 2, axis = 0)
        self.loss = np.sum(a = (np.array(object = richtiges_y) - np.array(object = self.output)) ** 2, axis = 1) / self.anzahl_neuronen
    def forward(self, inputs):
        '''
        output = x * weight + bias
        '''
        self.inputs = inputs
        self.output = (self.inputs @ self.weights) + self.bias
        if self.activation_function == True:
            self.relu()

hidden_1 = Layer(anzahl_neuronen = 10,
                 activation_function = True)
hidden_2 = Layer(anzahl_neuronen = 10,
                 activation_function = True)
hidden_3 = Layer(anzahl_neuronen = 10,
                 activation_function = True)
output_1 = Layer(anzahl_neuronen = 7)

# Auslesen von NN_parameter.json
with open("NN_parameter.json", "r") as file:
    parameter_dict = json.load(file)

# Initialisierung aller weights
output_1.weights = parameter_dict["output_1_weights"]
hidden_3.weights = parameter_dict["hidden_3_weights"]
hidden_2.weights = parameter_dict["hidden_2_weights"]
hidden_1.weights = parameter_dict["hidden_1_weights"]

# Initialisierung aller biases
output_1.bias = parameter_dict["output_1_bias"]
hidden_3.bias = parameter_dict["hidden_3_bias"]
hidden_2.bias = parameter_dict["hidden_2_bias"]
hidden_1.bias = parameter_dict["hidden_1_bias"]

# Extrema
extrema_df = pd.read_json(path_or_buf = "extrema.json")

input_maxima = extrema_df[["jahr", "monat", "tag", "stunde", "längengrad", "breitengrad", "höhe"]]
input_maxima = input_maxima.map(lambda x: float(x[1])).to_numpy()
input_minima = extrema_df[["jahr", "monat", "tag", "stunde", "längengrad", "breitengrad", "höhe"]]
input_minima = input_minima.map(lambda x: float(x[0])).to_numpy()
output_maxima = extrema_df[["lufttemperatur", "niederschlag", "windgeschwindigkeit", "windrichtung", "luftdruck", "luftfeuchtigkeit", "schneetiefe"]]
output_maxima = output_maxima.map(lambda x: float(x[1])).to_numpy()
output_minima = extrema_df[["lufttemperatur", "niederschlag", "windgeschwindigkeit", "windrichtung", "luftdruck", "luftfeuchtigkeit", "schneetiefe"]]
output_minima = output_minima.map(lambda x: float(x[0])).to_numpy()

iterationen_dict: dict = {"durchgang": [],
                          "jahr": [],
                          "monat": [],
                          "loss": [],
                          "lufttemperatur_loss": [],
                          "niederschlag_loss": [],
                          "windgeschwindigkeit_loss": [],
                          "windrichtung_loss": [],
                          "luftdruck_loss": [],
                          "luftfeuchtigkeit_loss": [],
                          "schneetiefe_loss": [],
                          "lufttemperatur_loss_absolut": [],
                          "niederschlag_loss_absolut": [],
                          "windgeschwindigkeit_loss_absolut": [],
                          "windrichtung_loss_absolut": [],
                          "luftdruck_loss_absolut": [],
                          "luftfeuchtigkeit_loss_absolut": [],
                          "schneetiefe_loss_absolut": []}

for file in listdir("TEST/"):
    df = pd.read_parquet(path = "TEST/" + file)
    
    for i in range(0, len(df), batch_size):
        durchgang += 1
        batch = df.iloc[i:(i + batch_size)]   
        test_data_unskaliert = batch[["jahr", "monat", "tag", "stunde", "längengrad", "breitengrad", "höhe"]].to_numpy()
        richtige_data_unskaliert = batch[["lufttemperatur", "niederschlag", "windgeschwindigkeit", "windrichtung", "luftdruck", "luftfeuchtigkeit", "schneetiefe"]].to_numpy()

    # Skalierung der Daten, anhand der Maxima jedes Features aus dem gesamten Datensatz
        test_data = (test_data_unskaliert - input_minima) / (input_maxima - input_minima)
        zufällige_reihenfolge = np.random.permutation(test_data.shape[0])
        test_data = test_data[zufällige_reihenfolge]
        richtige_data = (richtige_data_unskaliert - output_minima) / (output_maxima - output_minima)
        richtige_data = richtige_data[zufällige_reihenfolge]

    # Forward Propagation
        hidden_1.forward(inputs = test_data)
        hidden_2.forward(inputs = hidden_1.output)
        hidden_3.forward(inputs = hidden_2.output)
        output_1.forward(inputs = hidden_3.output)

    # Berechnung des Loss
        output_1.mse_loss(richtiges_y = richtige_data)

        loss_lufttemperatur = output_1.loss_per_output[0]
        loss_niederschlag = output_1.loss_per_output[1]
        loss_windgeschwindigkeit = output_1.loss_per_output[2]
        loss_windrichtung = output_1.loss_per_output[3]
        loss_luftdruck = output_1.loss_per_output[4] 
        loss_luftfeuchtigkeit = output_1.loss_per_output[5]
        loss_schneetiefe = output_1.loss_per_output[6]
        
        iterationen_dict["durchgang"].append(float(durchgang))
        iterationen_dict["loss"].append(float(np.mean(a = output_1.loss)))
        iterationen_dict["jahr"].append(float(file.split("_")[1]))
        iterationen_dict["monat"].append(float((file.split("_")[2][:-8])))
        iterationen_dict["lufttemperatur_loss"].append(float(loss_lufttemperatur))
        iterationen_dict["niederschlag_loss"].append(float(loss_niederschlag))
        iterationen_dict["windgeschwindigkeit_loss"].append(float(loss_windgeschwindigkeit))
        iterationen_dict["windrichtung_loss"].append(float(loss_windrichtung))
        iterationen_dict["luftdruck_loss"].append(float(loss_luftdruck))
        iterationen_dict["luftfeuchtigkeit_loss"].append(float(loss_luftfeuchtigkeit))
        iterationen_dict["schneetiefe_loss"].append(float(loss_schneetiefe))

    # Berechnung des absoluten Loss (ohne Skalierung)
    # ** 0.5 um den MSE-Loss rückgangig zu machen
    # * (max - min) um die Min-Max-Skalierung der Werte rückgängig zu machen
        iterationen_dict["lufttemperatur_loss_absolut"].append((loss_lufttemperatur ** 0.5) * (extrema_df.loc[0, "lufttemperatur"][1] - extrema_df.loc[0, "lufttemperatur"][0]))
        iterationen_dict["niederschlag_loss_absolut"].append((loss_niederschlag ** 0.5) * (extrema_df.loc[0, "niederschlag"][1] - extrema_df.loc[0, "niederschlag"][0]))
        iterationen_dict["windgeschwindigkeit_loss_absolut"].append((loss_windgeschwindigkeit ** 0.5) * (extrema_df.loc[0, "windgeschwindigkeit"][1] - extrema_df.loc[0, "windgeschwindigkeit"][0]))
        iterationen_dict["windrichtung_loss_absolut"].append((loss_windrichtung ** 0.5) * (extrema_df.loc[0, "windrichtung"][1] - extrema_df.loc[0, "windrichtung"][0]))
        iterationen_dict["luftdruck_loss_absolut"].append((loss_luftdruck ** 0.5) * (extrema_df.loc[0, "luftdruck"][1] - extrema_df.loc[0, "luftdruck"][0]))
        iterationen_dict["luftfeuchtigkeit_loss_absolut"].append((loss_luftfeuchtigkeit ** 0.5) * (extrema_df.loc[0, "luftfeuchtigkeit"][1] - extrema_df.loc[0, "luftfeuchtigkeit"][0]))
        iterationen_dict["schneetiefe_loss_absolut"].append((loss_schneetiefe ** 0.5) * (extrema_df.loc[0, "schneetiefe"][1] - extrema_df.loc[0, "schneetiefe"][0]))

        if (durchgang % 2000) == 0:
            print(f"Datei: {file}; Iteration: {durchgang}; Loss: {np.mean(a = output_1.loss)}\n" + 
                  f"  - Lufttemperatur: {loss_lufttemperatur} ~~ {(loss_lufttemperatur ** 0.5) * (extrema_df.loc[0, "lufttemperatur"][1] - extrema_df.loc[0, "lufttemperatur"][0])}\n" + 
                  f"  - Niederschlag: {loss_niederschlag} ~~ {(loss_niederschlag ** 0.5) * (extrema_df.loc[0, "niederschlag"][1] - extrema_df.loc[0, "niederschlag"][0])}\n" +
                  f"  - Windgeschwindigkeit: {loss_windgeschwindigkeit} ~~ {(loss_windgeschwindigkeit ** 0.5) * (extrema_df.loc[0, "windgeschwindigkeit"][1] - extrema_df.loc[0, "windgeschwindigkeit"][0])}\n" +
                  f"  - Windrichtung: {loss_windrichtung} ~~ {(loss_windrichtung ** 0.5) * (extrema_df.loc[0, "windrichtung"][1] - extrema_df.loc[0, "windrichtung"][0])}\n" +
                  f"  - Luftdruck: {loss_luftdruck} ~~ {(loss_luftdruck ** 0.5) * (extrema_df.loc[0, "luftdruck"][1] - extrema_df.loc[0, "luftdruck"][0])}\n" +
                  f"  - Luftfeuchtigkeit: {loss_luftfeuchtigkeit} ~~ {(loss_luftfeuchtigkeit ** 0.5) * (extrema_df.loc[0, "luftfeuchtigkeit"][1] - extrema_df.loc[0, "luftfeuchtigkeit"][0])}\n" +
                  f"  - Schneetiefe: {loss_schneetiefe} ~~ {(loss_schneetiefe ** 0.5) * (extrema_df.loc[0, "schneetiefe"][1] - extrema_df.loc[0, "schneetiefe"][0])}\n")

        if (durchgang % 1000000) == 999999:
            iterationen_df = pd.DataFrame(data = iterationen_dict)
            iterationen_df.to_parquet(path = f"TRAIN_loss/{durchgang}.parquet")

            iterationen_dict: dict = {"durchgang": [],
                                      "jahr": [],
                                      "monat": [],
                                      "loss": [],
                                      "lufttemperatur_loss": [],
                                      "niederschlag_loss": [],
                                      "windgeschwindigkeit_loss": [],
                                      "windrichtung_loss": [],
                                      "luftdruck_loss": [],
                                      "luftfeuchtigkeit_loss": [],
                                      "schneetiefe_loss": [],
                                      "lufttemperatur_loss_absolut": [],
                                      "niederschlag_loss_absolut": [],
                                      "windgeschwindigkeit_loss_absolut": [],
                                      "windrichtung_loss_absolut": [],
                                      "luftdruck_loss_absolut": [],
                                      "luftfeuchtigkeit_loss_absolut": [],
                                      "schneetiefe_loss_absolut": []}

iterationen_df = pd.DataFrame(data = iterationen_dict)
iterationen_df.to_parquet(path = f"TEST_loss/{durchgang}.parquet", engine = "pyarrow")
