import torch.nn as nn

from hidden_layer import hidden_layers
from output_layer import OutputLayer

#this combines the hidden layer and output_layer files into a single neural network module whose inherent
# bias and weight parameters  may be adjusted by the loss.backward() and optimizer.step() functions
#note: hidden size refers to the number of outputs from the second hidden layer to the output layer
class CombinedWorkflow(nn.Module):
    def __init__(self, Input_Size, num_nodes1, num_nodes2, hidden_size, num_classes):
        super().__init__()
        self.hidden = hidden_layers(Input_Size, num_nodes1, num_nodes2)
        self.out = OutputLayer(hidden_size, num_classes)
    def forward(self, inputs):
        hl_output = self.hidden(inputs)
        output_layer = self.out(hl_output)
        return output_layer

