import numpy as np
from os import listdir
import pandas as pd
import json

batch_size: int = 64
lern_durchgang: int = 0

lernrate_anfang: float = 0.007
abfall: float = 0.00000019
lernrate = lernrate_anfang / (1 + abfall * lern_durchgang)

class Layer:
    def __init__(self, 
                 anzahl_inputs: int, 
                 anzahl_neuronen: int, 
                 activation_function: bool = False):
        self.anzahl_neuronen = anzahl_neuronen
        self.weights = np.random.randn(anzahl_inputs, anzahl_neuronen) * np.sqrt(2 / anzahl_inputs) # He-Initialisierung
        self.bias = np.zeros((1, anzahl_neuronen))
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

hidden_1 = Layer(anzahl_inputs = 7, 
                 anzahl_neuronen = 64,
                 activation_function = True)
hidden_2 = Layer(anzahl_inputs = 64, 
                 anzahl_neuronen = 64,
                 activation_function = True)
hidden_3 = Layer(anzahl_inputs = 64, 
                 anzahl_neuronen = 32,
                 activation_function = True)
output_1 = Layer(anzahl_inputs = 32,
                 anzahl_neuronen = 7)

# Extrema
extrema_df = pd.read_json(path_or_buf = "extrema.json")

training_maxima = extrema_df[["jahr", "monat", "tag", "stunde", "längengrad", "breitengrad", "höhe"]]
training_maxima = training_maxima.map(lambda x: float(x[1])).to_numpy()
training_minima = extrema_df[["jahr", "monat", "tag", "stunde", "längengrad", "breitengrad", "höhe"]]
training_minima = training_minima.map(lambda x: float(x[0])).to_numpy()
richtige_maxima = extrema_df[["lufttemperatur", "niederschlag", "windgeschwindigkeit", "windrichtung", "luftdruck", "luftfeuchtigkeit", "schneetiefe"]]
richtige_maxima = richtige_maxima.map(lambda x: float(x[1])).to_numpy()
richtige_minima = extrema_df[["lufttemperatur", "niederschlag", "windgeschwindigkeit", "windrichtung", "luftdruck", "luftfeuchtigkeit", "schneetiefe"]]
richtige_minima = richtige_minima.map(lambda x: float(x[0])).to_numpy()

iterationen_dict: dict = {"durchgang": [],
                          "jahr": [],
                          "monat": [],
                          "loss": []}

for file in listdir("TRAIN_shuffled/"):
    df = pd.read_parquet(path = "TRAIN_shuffled/" + file)
    
    for i in range(0, len(df), batch_size):
        lern_durchgang += 1
        batch = df.iloc[i:(i + batch_size)]   
        training_data_unskaliert = batch[["jahr", "monat", "tag", "stunde", "längengrad", "breitengrad", "höhe"]].to_numpy()
        richtige_data_unskaliert = batch[["lufttemperatur", "niederschlag", "windgeschwindigkeit", "windrichtung", "luftdruck", "luftfeuchtigkeit", "schneetiefe"]].to_numpy()

    # Skalierung der Daten, anhand der Maxima jedes Features aus dem gesamten Datensatz
        training_data = (training_data_unskaliert - training_minima) / (training_maxima - training_minima)
        zufällige_reihenfolge = np.random.permutation(training_data.shape[0])
        training_data = training_data[zufällige_reihenfolge]
        richtige_data = (richtige_data_unskaliert - richtige_minima) / (richtige_maxima - richtige_minima)
        richtige_data = richtige_data[zufällige_reihenfolge]

    # Forward Propagation
        hidden_1.forward(inputs = training_data)
        hidden_2.forward(inputs = hidden_1.output)
        hidden_3.forward(inputs = hidden_2.output)
        output_1.forward(inputs = hidden_3.output)
        
    # Berechnung des Loss
        output_1.mse_loss(richtiges_y = richtige_data)
        
        if (lern_durchgang % 2000) == 0:
            print(f"Datei: {file}; Iteration: {lern_durchgang}; Loss: {np.mean(a = output_1.loss)}\n" + 
                  f"  - Lufttemperatur: {output_1.loss_per_output[0]}\n" + 
                  f"  - Niederschlag: {output_1.loss_per_output[1]}\n" +
                  f"  - Windgeschwindigkeit: {output_1.loss_per_output[2]}\n" +
                  f"  - Windrichtung: {output_1.loss_per_output[3]}\n" +
                  f"  - Luftdruck: {output_1.loss_per_output[4]}\n" +
                  f"  - Luftfeuchtigkeit: {output_1.loss_per_output[5]}\n" +
                  f"  - Schneetiefe: {output_1.loss_per_output[6]}\n")
        
        iterationen_dict["durchgang"].append(float(lern_durchgang))
        iterationen_dict["loss"].append(float(np.mean(a = output_1.loss)))
        iterationen_dict["jahr"].append(float(file.split("_")[1]))
        iterationen_dict["monat"].append(float((file.split("_")[2][:-8])))

        if (lern_durchgang % 1000000) == 999999:
            iterationen_df = pd.DataFrame(data = iterationen_dict)
            iterationen_df.to_parquet(path = f"TRAIN_loss/{lern_durchgang}.parquet")

            iterationen_dict: dict = {"durchgang": [],
                                      "jahr": [],
                                      "monat": [],
                                      "loss": []}

    # Backward Propagation
        def ableitung_ReLU_nach_input(x):
            return np.where(x > 0, 1.0, 0.01)
        fehlerterm_output = 2 * (output_1.output - richtige_data) / (training_data.shape[0] * output_1.anzahl_neuronen)
        fehlerterm_hidden_3 = (fehlerterm_output @ output_1.weights.T) * ableitung_ReLU_nach_input(x = hidden_3.output_vor_activation)
        fehlerterm_hidden_2 = (fehlerterm_hidden_3 @ hidden_3.weights.T) * ableitung_ReLU_nach_input(x = hidden_2.output_vor_activation)
        fehlerterm_hidden_1 = (fehlerterm_hidden_2 @ hidden_2.weights.T) * ableitung_ReLU_nach_input(x = hidden_1.output_vor_activation)

        # Aktualisierung aller weights
        output_1.weights -= (hidden_3.output.T @ fehlerterm_output) * lernrate
        hidden_3.weights -= (hidden_2.output.T @ fehlerterm_hidden_3) * lernrate
        hidden_2.weights -= (hidden_1.output.T @ fehlerterm_hidden_2) * lernrate
        hidden_1.weights -= (training_data.T @ fehlerterm_hidden_1) * lernrate

        # Aktualisierung aller biases
        output_1.bias -= np.sum(a = fehlerterm_output, axis = 0, keepdims = True) * lernrate
        hidden_3.bias -= np.sum(a = fehlerterm_hidden_3, axis = 0, keepdims = True) * lernrate
        hidden_2.bias -= np.sum(a = fehlerterm_hidden_2, axis = 0, keepdims = True) * lernrate
        hidden_1.bias -= np.sum(a = fehlerterm_hidden_1, axis = 0, keepdims = True) * lernrate

        # Aktualisierung der Lernrate
        lernrate = lernrate_anfang / (1 + abfall * lern_durchgang)

angepasste_parameter: dict = {
    "hidden_1_bias": hidden_1.bias.tolist(),
    "hidden_1_weights": hidden_1.weights.tolist(),
    "hidden_2_bias": hidden_2.bias.tolist(),
    "hidden_2_weights": hidden_2.weights.tolist(),
    "hidden_3_bias": hidden_3.bias.tolist(),
    "hidden_3_weights": hidden_3.weights.tolist(),
    "output_1_bias": output_1.bias.tolist(),
    "output_1_weights": output_1.weights.tolist()
}

iterationen_df = pd.DataFrame(data = iterationen_dict)
iterationen_df.to_parquet(path = f"TRAIN_loss/{lern_durchgang}.parquet", engine = "pyarrow")

with open("NN_parameter.json", "w") as file:
    json.dump(angepasste_parameter, file)