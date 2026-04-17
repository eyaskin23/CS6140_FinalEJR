import torch
import torch.nn as nn
import numpy as np
#class hidden layers implements the two-layer structure, containing a pair of hidden layers, which includes a set
#of weights initialized randomly (gaussian distribution) and biases set to 0, with subsequent iterations/epochs
#utilizing backpropagation to adjust weights and bias values
class hidden_layers(nn.Module):
    def __init__(self, Input_Size, num_nodes1, num_nodes2):
        super().__init__()
        # functions for applying linear transformations to input data, generates tensors for all linear functions
        #xW^T + b
        # to call specific elements of the tensor, use h1.weight or h1.bias
        self.h1 = nn.Linear(Input_Size, num_nodes1)
        self.h2 = nn.Linear(num_nodes1, num_nodes2)

        #self.learn_rate = learn_rate
        # randomly initializes weight values
        nn.init.kaiming_normal_(self.h1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.h2.weight, mode='fan_in', nonlinearity='relu')



    def forward(self, inputs):
       hl_output1 = nn.functional.relu(self.h1(inputs))
       hl_output2 = nn.functional.relu(self.h2(hl_output1))
       return hl_output2








