import numpy as np
import pandas as pd
import json

input_features: int = 7
neuron_layout: dict = {
    "hidden_1": 10,
    "hidden_2": 10,
    "hidden_3": 10,
    "output": 7
}

class Layer:
    def __init__(self, 
                 activation_function: bool = False):
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
    def forward(self, inputs):
        '''
        output = x * weight + bias
        '''
        self.inputs = inputs
        self.output = (self.inputs @ self.weights) + self.bias
        if self.activation_function == True:
            self.relu()

hidden_1 = Layer(activation_function = True)
hidden_2 = Layer(activation_function = True)
hidden_3 = Layer(activation_function = True)
output_1 = Layer()

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

# Auslesen von NN_parameter.json
extrema_df = pd.read_json(path_or_buf = "extrema.json")

training_minima = extrema_df[["jahr", "monat", "tag", "stunde", "längengrad", "breitengrad", "höhe"]]
training_minima = training_minima.map(lambda x: float(x[0])).to_numpy()

training_maxima = extrema_df[["jahr", "monat", "tag", "stunde", "längengrad", "breitengrad", "höhe"]]
training_maxima = training_maxima.map(lambda x: float(x[1])).to_numpy()

# Anfrage und Skalierung aller Inputfeatures
anfrage: list = [
    (float(input("Längengrad: "))  - training_minima)  / (training_maxima - training_minima),
    (float(input("Breitengrad: "))  - training_minima)  / (training_maxima - training_minima),
    (float(input("Höhe: "))  - training_minima)  / (training_maxima - training_minima),
    (float(input("Jahr: "))  - training_minima)  / (training_maxima - training_minima),
    (float(input("Monat: "))  - training_minima)  / (training_maxima - training_minima),
    (float(input("Tag: "))  - training_minima)  / (training_maxima - training_minima),
    (float(input("Stunde: "))  - training_minima)  / (training_maxima - training_minima),
].to_numpy()

# Forward Propagation
hidden_1.forward(inputs = anfrage)
hidden_2.forward(inputs = hidden_1.output)
hidden_3.forward(inputs = hidden_2.output)
output_1.forward(inputs = hidden_3.output)

# Zurückskalierung der Outputs
output = (output_1.output * (training_maxima - training_minima)) + training_minima
print(f"Lufttemperatur: {output[0]}°C\n" + 
      f"Niederschlag: {output[1]}\n" + 
      f"Windgeschwindigkeit: {output[2]}\n" + 
      f"Windrichtung: {output[3]}\n" + 
      f"Luftdruck: {output[4]}\n" + 
      f"Luftfeuchtigkeit: {output[5]}\n" + 
      f"Schneetiefe: {output[6]}")