import torch
import torch.nn as nn
from data_preprocessing_1 import get_input_shape
import torch.nn.functional as F

from hidden_layer import hidden_layers
from output_layer import OutputLayer

#this combines the hidden layer and output_layer_example files into a single neural network module whose inherent
# bias and weight parameters which may be adjusted by the loss.backward() and optimizer.step() functions
class CombinedWorkflow(nn.Module):
    def __init__(self, input_nodes, num_nodes1, num_nodes2, hidden_size, num_classes):
        super().__init__()
        #this new component implements a convolutional neural network structure, including a CNN layer which converts the
        #images into a feature map prior to feeding the outputs into the hidden layers
        zeros_tensor = torch.zeros(1, *get_input_shape())
        self.conv1 = nn.Conv2d(1,32,3)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32, 128, 3)

        #this produces an initial, pool tensor of zeros based on the size of the input tensors to determine the necessary
        #number of inputs into the first hidden layer
        pool_set1 = self.pool(F.relu(self.conv1(zeros_tensor)))
        pool_set2 = self.pool(F.relu(self.conv2(pool_set1)))
        pool_flat1 = pool_set1.view(1, -1)
        pool_flat2 = pool_set2.view(1,-1)
        flat2_size = pool_flat2.shape[1]
        self.hidden = hidden_layers(flat2_size, num_nodes1, num_nodes2)
        self.out = OutputLayer(hidden_size, num_classes)
    def forward(self, inputs):
        conv_layer1 = self.pool(F.relu(self.conv1(inputs)))
        conv_layer2 = self.pool(F.relu(self.conv2(conv_layer1)))
        x = conv_layer2.view(conv_layer2.size(0), -1)
        hl_output = self.hidden(x)
        output_layer = self.out(hl_output)
        return output_layer

