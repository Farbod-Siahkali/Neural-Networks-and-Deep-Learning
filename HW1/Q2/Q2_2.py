import numpy as np
import matplotlib.pyplot as plt

class Neuron:
  def __init__(self, weights, bias, epoch, activation_function, learning_rate):
    self.weights = weights
    self.bias = bias
    self.epoch = epoch
    self.activation_function = activation_function
    self.learning_rate = learning_rate
    self.counter = 0
  
  def calculate_net_value(self, inputs):
    return np.dot(inputs, self.weights) + self.bias

  def calculate_h_value(self, inputs):
    return(self.activation_function(self.calculate_net_value(inputs)))

  def update(self, target, inputs):
    self.bias = self.bias + self.learning_rate * (target - self.calculate_net_value(inputs))
    self.weights = self.weights +  np.dot(self.learning_rate * int(target - self.calculate_net_value(inputs)) , inputs)
    self.counter += 1

  def get_weights(self):
    return self.weights

  def get_bias(self):
    return self.bias

def activation_function(x):
  if x >= 0:
    return 1
  else:
    return 0

def and_operator(a, b):
  if a == 1 and b == 1:
    return 1
  elif a == 1 and b == 0:
    return 0
  elif a == 0 and b == 1:
    return 0
  elif a == 0 and b == 0:
    return 1

def line_plot(weight, bias, x1, x2):
  a_out1 = (x1 * weight[0] + bias) / weight[1]
  a_out2 = (x2 * weight[0] + bias) / weight[1]
  return [x1, x2] , [a_out1, a_out2]

class Network:
  def __init__(self, inputs, neuron_number):
    self.neuron_number = neuron_number
    self.inputs = inputs
    self.neurons = []

  def plot_hyperplanes(self, x1, x2):
    fig = plt.figure()
    plt.scatter(x1, x2)
    for neuron in self.neurons:
      x_hyperplane, y_hyperplane = line_plot(neuron.weights, neuron.bias, x1, x2)
      plt.plot(x_hyperplane, y_hyperplane, '-')
    plt.show()

  def make_neurons(self):
    for i in range(self.neuron_number):
      w = np.random.normal(1, 3, 2)
      b = 1
      self.neurons.append(Neuron(w, b, 1000, activation_function, 0.15))

  def train(self, limit):
    self.make_neurons()
    h = [j for j in range(len(self.inputs))]
    for i in h:
      hidden_out = []
      for j in range(len(self.neurons)):
        hidden_out.append(self.neurons[j].calculate_h_value(inputs[i]))
      and_result = 1
      for l in range(len(hidden_out)):
        and_result = and_operator(and_result, hidden_out[l])
      
      if i >= limit:
        target = 1
      elif i < limit:
        target = 0

      if and_result == 1 and target == 0:
        net_values = []
        for j in range(len(self.neurons)):
          net_values.append(self.neurons[j].calculate_net_value(inputs[i]))
        min_net_value = 99999
        index = 0
        for k in range(len(net_values)):
          if net_values[k] < min_net_value:
            min_net_value = net_values[k]
            index = k
        self.neurons[index].update(target, self.inputs[i])
        
      elif and_result == 0 and target == 1:
        for j in range(len(self.neurons)):
          if hidden_out[j] == 0:
            self.neurons[j].update(target, self.inputs[i])
    weights = []
    bias = []
    for i in range(len(self.neurons)):
      weights.append(self.neurons[i].get_weights())  
      bias.append(self.neurons[i].get_bias())
    return weights, bias

  def test(self, limit):
    total = 0
    error = 0
    for i in range(len(self.inputs)):
      hidden_out = []
      for j in range(len(self.neurons)):
        hidden_out.append(self.neurons[j].calculate_h_value(inputs[i]))
      and_result = 1
      for k in range(len(hidden_out)):
        and_result = and_operator(and_result, hidden_out[k])
      total += 1
      if i < limit:
        if and_result == 0:
          error += 1
      else:
        if and_result == 1:
          error += 1
    print(error / total)

def plot_hyperplane(X, weights, bias):
  slope = - weights[0]/weights[1]
  intercept = bias/weights[1]
  x_hyperplane = np.linspace(-2,2,10)
  y_hyperplane = slope * x_hyperplane + intercept
  plt.plot(x_hyperplane, y_hyperplane, '-')

csvdata = np.genfromtxt('MadaLine.csv', delimiter=',')

x1 = csvdata[csvdata[:, 2] == 1][:, 0]
x2 = csvdata[csvdata[:, 2] == 1][:, 1]
x1_2 = csvdata[csvdata[:, 2] == 0][:, 0]
x2_2 = csvdata[csvdata[:, 2] == 0][:, 1]

inputs = []

for i in range(x1.shape[0]):
  inputs.append([x1[i], x2[i]])

for i in range(x1_2.shape[0]):
  inputs.append([x1_2[i], x2_2[i]])

#net3 = Network(inputs, 3)
#net4 = Network(inputs, 4)
net8 = Network(inputs, 8)

#first_weights, first_bias = net3.train(x1.shape[0])
#second_weights, second_bias = net4.train(x1.shape[0])
third_weights, third_bias = net8.train(x1.shape[0])

#net3.test(x1.shape[0])
#net4.test(x1.shape[0])
net8.test(x1.shape[0])
'''
fig = plt.figure(figsize=(8,6))
plt.scatter(x1, x2) 
plt.scatter(x1_2, x2_2) 
for i in range(len(first_bias)):
  plot_hyperplane(inputs, first_weights[i], first_bias[i])
plt.title("Dataset and decision hyperplane")

fig = plt.figure(figsize=(8,6))
plt.scatter(x1, x2) 
plt.scatter(x1_2, x2_2) 
for i in range(len(second_bias)):
  plot_hyperplane(inputs, second_weights[i], second_bias[i])
plt.title("Dataset and decision hyperplane")'''

fig = plt.figure(figsize=(8,6))
plt.scatter(x1, x2) 
plt.scatter(x1_2, x2_2) 
for i in range(len(third_bias)):
      plot_hyperplane(inputs, third_weights[i], third_bias[i])
plt.title("Dataset and decision hyperplane")

plt.show()