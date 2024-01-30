import torch
import torchvision
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import math
from typing import List
from types import SimpleNamespace
from tqdm import tqdm


# these are pytorch's dataset class. it downloads and sets up the dataset
mnist_train = torchvision.datasets.MNIST(root="data", 
                                         train=True, 
                                         download=True, 
                                         transform=T.Compose([T.ToTensor(),
                                                              T.Lambda(lambda x: torch.flatten(x)),
                                                              ]),
                                         )
mnist_test = torchvision.datasets.MNIST(root="data", 
                                        train=False, 
                                        download=True, 
                                        transform=T.Compose([T.ToTensor(),
                                                             T.Lambda(lambda x: torch.flatten(x)),
                                                             ]),
                                        )



# these are pytorch's dataloader class. they shuffle and yield sample points
# for us. you don't have to code data manipulations yourself in this exercise
train_loader = DataLoader(mnist_train, batch_size=1, shuffle=True)
val_loader = DataLoader(mnist_test, batch_size=1, shuffle=True)


# hyperparameters. sets up the network dimensions
# how many nodes in each
layers_list = [784, 30, 10]
L = len(layers_list)

def create_single_layer(input_size: int, output_size: int, scaled_init=False):
  """
  Create a single layer
  output: a layer object. weights and biases are in layer.weight and layer.bias 

  Usage:
    layer = create_single_layer(784, 10)
    layer_weight_matrix = layer.weight
    layer_bias_vector = layer.bias
  """
  weight_matrix = np.random.randn(output_size, input_size)
  bias_vector = np.random.randn(output_size)
  if scaled_init:
    weight_matrix = initialization_scaling(weight_matrix)

  
  # The output being a SimpleNamespace is just a coding convenience. It 
  # emulates how we access the weights and biases in pytorch.
  # without this we would have to access weights using layers[0][0],
  # with this we can do layers[0].weight so that it is a bit more readable
  return SimpleNamespace(weight=weight_matrix, bias=bias_vector)



def create_layers(layers_list: List):
  """
  Creates the weights and biases if layers according to the given list
  output: a list of layers

  Usage:
    layers = create_layers(layers_list)
    some_layer = layers[np.random.randint(L)]
  """
  
  # To match the Lecture notation we start with idx=1, i.e. there is no W^(0) or b^(0)
  layers = [None]
  for idx in range(L-1):
      layer = create_single_layer(layers_list[idx], layers_list[idx+1], True)
      layers.append(layer)
  return layers


def sigmoid(z):
  """
  The sigmoid function
  Usage:
    activation_output = sigmoid(some_input)
  """
  return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
  """
  $\sigma\prime$ in the lecture notes
  """
  return sigmoid(z)*(1-sigmoid(z))



def loss_fn(y_hat, y):
  """
  This is the MSE loss for a single sample.
  """
  assert y_hat.size == y.size
  size = y_hat.size

  return (1/size) * ((y_hat - y)**2)
  

def dloss_dyhat(y_hat, y):
  """
  Gradient of the MSE loss for a single sample.
  """
  assert y_hat.size == y.size
  size = y_hat.size

  return (1/size) * (2*(y_hat - y))


def calculate_preactivation(layer_weight, layer_bias, previous_activation):
  z = (layer_weight @ previous_activation).squeeze() + layer_bias.squeeze()
  return z


def calculate_activation(z):
  a = sigmoid(z)
  return a


def calculate_g_L(y, a, z):
  dj_da = np.atleast_1d(dloss_dyhat(a, y).squeeze())
  da_dz = np.diag(np.atleast_1d(sigmoid_prime(z).squeeze()))

  return dj_da @ da_dz


def calculate_g_l(g_l_previous, layer_weight_previous, z):
  g_l = (g_l_previous @ layer_weight_previous) \
        @ np.diag(np.atleast_1d(sigmoid_prime(z).squeeze()))

  return g_l


def calculate_nabla_weight(g_l, a_next):
  a_next = a_next.reshape((1, a_next.size)) # turn into row vector
  g_l = g_l.reshape((g_l.size,1))           # turn into column vector
  g_w = (g_l @ a_next)
  return g_w

  
def calculate_nabla_bias(g_l):
  return g_l.T

  
# initialization = create the layers
layers = create_layers(layers_list) ### YOUR CODE HERE ###
lr = 3.0

# this is one epoch
# ====== loop through 1 epoch using the data loader =====
for (x, y) in tqdm(train_loader, desc=f'Epoch progress'):
  # we used pytorch loaders, convert back to numpy and proper shape
  x = x.numpy().T
  # our network uses one_hot encoding to classify the digit so
  # we convert the target to a one_hot encoding
  y = torch.nn.functional.one_hot(y,num_classes=10).numpy().T
  # actual updates occur here
  activation = x
  # set up the variables for saving the a & z values
  # a^(0) is set to x
  activations = [activation]
  # z^(0) does not exist so we store a None for it
  pre_activations = [None]

  # ====== do forward propagation =====
  # looping through the layers, computing and storing activations & pre-activations
  for layer in layers[1:]: #since layers[0] does not exist/is None
    pre_activation = calculate_preactivation(layer.weight, layer.bias, activations[-1]) ### YOUR CODE HERE ###
    activation = calculate_activation(pre_activation) ### YOUR CODE HERE ###
    # store these for the backprop
    pre_activations.append(pre_activation)
    activations.append(activation)

  # ===== do backpropagation =====
  # setting up record keeping of the gradients, nabla_weights and nabla_biases
  # note that we have no gradients for W^(0) since we index starting at 1
  g_ls = [None]
  nabla_weights = [None]
  nabla_biases = [None]
  g_ls.extend([np.zeros((1, m)) for m in layers_list] )
  nabla_weights.extend([np.zeros_like(layer.weight) for layer in layers[1:]])
  nabla_biases.extend([np.zeros_like(layer.bias) for layer in layers[1:]])

  
  # looping through the layers, compute gradients and nabla's
  for l in range(L-1, 0, -1):
    # g^L is calculated differently, so we check and handle that first
    if l == L-1:
      g_ls[l] = calculate_g_L(y.squeeze(), activations[l], pre_activations[l]) ### YOUR CODE HERE ###
    else:
      g_ls[l] = calculate_g_l(g_ls[l+1], layers[l+1].weight, pre_activations[l]) ### YOUR CODE HERE ###

    nabla_weights[l] = calculate_nabla_weight(g_ls[l], activations[l-1]) #### YOUR CODE HERE ###
    nabla_biases[l] = calculate_nabla_bias(g_ls[l]) ### YOUR CODE HERE ###
    
    # update the weights with the lr according to SGD algorithm
    layers[l].weight = layers[l].weight - lr*nabla_weights[l] ### YOUR CODE HERE ###
    layers[l].bias = layers[l].bias - lr*nabla_biases[l]### YOUR CODE HERE ###
    
    
# create a forward function for evaluation, without backprop
def forward(layers, x):
  """
  Forward propagation. 
  """
  activation = x
  for layer in layers[1:]: #since layers[0] does not exist
    activation = sigmoid((layer.weight @ activation).squeeze() + layer.bias) 
  return activation

# Evaluate
test_results = []
for (x,y) in val_loader:
  x = x.numpy().T
  y = y.numpy().T
  test_results.append((np.argmax(forward(layers, x)), y))

accuracy = sum(int(x == y) for (x, y) in test_results)/len(test_results)*100
print(accuracy)


# and show an example
x, y = next(iter(val_loader))
x = x.numpy().T
y = y.numpy().T
plt.title(f'Prediction is {np.argmax(forward(layers, x))}')
plt.imshow(x.reshape(28,28), cmap='gray');