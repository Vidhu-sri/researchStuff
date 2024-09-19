import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
from mealpy.swarm_based import GWO
from mealpy.swarm_based import ACOR
from mealpy.swarm_based import PSO
from mealpy.evolutionary_based import FPA
from matplotlib import pyplot as plt
import seaborn as sns




class HybridMlp:

    def __init__(self, dataTrain, dataTest, yTrain, yTest, n_hidden_nodes, epoch, pop_size, algo):
        self.X_train, self.y_train, self.X_test, self.y_test =dataTrain, yTrain, dataTest, yTest
        self.n_hidden_nodes = n_hidden_nodes
        self.epoch = epoch
        self.pop_size = pop_size
        self.algo = algo


        self.n_inputs = self.X_train.shape[1]
        self.model, self.problem_size, self.n_dims, self.problem = None, None, None, None
        self.optimizer, self.solution, self.best_fit = None, None, None

    def create_network(self):
        # create model
        model = keras.Sequential()
        model.add(keras.layers.InputLayer(input_shape=(self.X_train.shape[1],)))

        for i in range(1,len(self.n_hidden_nodes)):
          model.add(keras.layers.Dense(self.n_hidden_nodes[i], activation = 'relu'))

        model.add(keras.layers.Dense(6, activation = 'softmax'))
        # Compile model
        # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = model
        self.problem_size = self.n_dims = np.sum([np.size(w) for w in self.model.get_weights()])

    def create_problem(self):
        self.problem = {
            "fit_func": self.fitness_function,
            "lb": [-1, ] * self.n_dims,
            "ub": [1, ] * self.n_dims,
            "minmax": "max",
            "log_to": None,
            "save_population": False
        }

    def decode_solution(self, solution):
        # solution: is a vector.
        # solution = [w11, w21, w31, w12, w22, w32, b1, b2, wh11, wh21, wh12, wh22, wh13, wh23, bo1, bo2, bo3 ]
        # number of weights = n_inputs * n_hidden_nodes + n_hidden_nodes + n_hidden_nodes * n_outputs + n_outputs
        # we decode the solution into the neural network weights
        # we return the model with the new weight (weight from solution)
        weight_sizes = [(w.shape, np.size(w)) for w in self.model.get_weights()]
        # ( (3, 5),  15 )
        weights = []
        cut_point = 0
        for ws in weight_sizes:
            temp = np.reshape(solution[cut_point: cut_point + ws[1]], ws[0])
            # [0: 15], (3, 5),
            weights.append(temp)
            cut_point += ws[1]
        self.model.set_weights(weights)

    def prediction(self, solution, x_data):
        self.decode_solution(solution)
        return self.model.predict(x_data)

    def training(self):
        self.create_network()
        self.create_problem()
        if(self.algo == 'GWO'):
          self.optimizer = GWO.OriginalGWO(self.epoch, self.pop_size)
        elif(self.algo =='ACOR'):
          self.optimizer = ACOR.OriginalACOR(self.epoch, self.pop_size)
        elif(self.algo == 'PSO'):
          self.optimizer = PSO.OriginalPSO()
        else:
         self.optimizer = FPA.OriginalFPA(self.problem, self.epoch, self.pop_size)

        self.solution, self.best_fit = self.optimizer.solve(self.problem)

    def fitness_function(self, solution):  # Used in training process
        # Assumption that we have 3 layer , 1 input layer, 1 hidden layer and 1 output layer
        # number of nodes are 3, 2, 3
        # solution = [w11, w21, w31, w12, w22, w32, b1, b2, wh11, wh21, wh12, wh22, wh13, wh23, bo1, bo2, bo3 ]
        self.decode_solution(solution)
        yhat = self.model.predict(self.X_train)
        yhat = np.argmax(yhat, axis=-1).astype('int')
        acc = accuracy_score(self.y_train, yhat)
        return acc
