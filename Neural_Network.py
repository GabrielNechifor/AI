import numpy as np


class NeuralNetwork:
    def __init__(self, number_of_layers, number_of_neurons_in_layers):
        self.input = np.zeros(shape=(number_of_layers,1))
        self.layers = [ np.zeros(shape=(number_of_neurons_in_layers[index],1)) for index in range(0, number_of_layers) ]


        self.number_of_layers = number_of_layers
        self.neurons_in_layers = number_of_neurons_in_layers

        self.weights = list()
        self.biases = list()

        self.initialize_weights_and_biasses()


    def initialize_weights_and_biasses(self):
        for layer_index in range(1, self.number_of_layers):
            neurons_in_previous_layer = self.neurons_in_layers[layer_index - 1]
            neurons_in_current_layer = self.neurons_in_layers[layer_index]
            self.weights.append( np.random.uniform(-1,1,(neurons_in_current_layer, neurons_in_previous_layer)) )
            if layer_index != 0:
                self.biases.append( np.random.uniform(-1,1,(neurons_in_previous_layer,1)) )
            else:
                self.biases.append(0)  #first layer does't have biases

        neurons_in_output_layer = self.neurons_in_layers[self.number_of_layers -1]
        self.biases.append(np.random.uniform(-1,1,(neurons_in_output_layer,1))) #biases for output neurons

    
    def compute_neurons_value(self, input_neurons):
        number_of_neurons_in_first_layer = self.neurons_in_layers[0]
        input_neurons = np.reshape(input_neurons, newshape=(number_of_neurons_in_first_layer ,1))
        self.layers[0] = input_neurons

        last_layer_index = self.number_of_layers - 1

        for layer_index in range(1, self.number_of_layers):
            previous_layer_index = layer_index -1
            weights = self.weights[previous_layer_index]
            neurons = self.layers[previous_layer_index]
            biases = self.biases[layer_index]

            print("computing in layer",layer_index)
            print("weights: ")
            print(weights)
            print("neurons:")
            print(neurons)
            print("biases: ")
            print(biases)
            
            result = np.matmul(weights, neurons)
            result = np.add(result, biases)
            result = self.sigmoid(result)
            print("result:")
            print(result)

            self.layers[layer_index] = result
            
            print()


    def sigmoid(self, value):
        return 1/(1 + np.exp(-value))


    



    def print(self):
        for layer_index in range(0, self.number_of_layers):
            print("layer: ", layer_index)
            print("neurons: ")
            print(self.layers[layer_index])
            if(layer_index != self.number_of_layers -1):
                print("weights: ")
                print(self.weights[layer_index])
            if(layer_index != 0):
                print("biases: ")
                print( self.biases[layer_index])
            print()
            print()


NN = NeuralNetwork(4, [4,2,2,1])
NN.compute_neurons_value([0.1, 0.5, 0.3, 0.9])
#NN.print()           

        
