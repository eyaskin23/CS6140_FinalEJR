import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import data_preprocessing_1
from data_preprocessing_1 import NoteDataset, get_input_shape
from Combined_Workflow import CombinedWorkflow
from output_layer import predict_note

total_loss = 0
num_epochs = 10
# back slash for windows, forward slash for mac (r"../data"), (r"..\data")
input_array = NoteDataset(r"../data")
tensor, label = input_array[0]
tensor = tensor.unsqueeze(0)


#num_nodes_1 = int(tensor.size(0) / 2)
#num_nodes_2 = int(num_nodes_1 / 2)
#hard coded the number of nodes for the hidden layers since the previous format for finding tensor dimensions was
#no longer valid
num_nodes_1 = 1024
num_nodes_2 = 512
print(num_nodes_1)
print(num_nodes_2)

#this portion calls the file which combines the hidden layer and output layer into a single workflow nn.Module that can
#be called and have its weights and biases modified by the training loop below
model = CombinedWorkflow(tensor.shape[0], num_nodes_1, num_nodes_2, num_nodes_2, num_classes=12)
logits = model(tensor)
# print(logits)
#this sets the learning rate hyper parameter
learn_rate = 0.0001
#this sets the loss as being calculated using cross-entropy, as well as selects the stochastic gradient descent method
#as the "optimizer" (i.e., the function and method used to calculate the gradients that back propagate and adjust
#the weights of the hidden layers and output layer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learn_rate)
#this is the actual training loop.  The current version uses a batchsize of 1, but once we have worked out the last
#step I will modify it to use larger batchsizes
cuda_check = torch.cuda.is_available()
print("Cuda is "+ str(cuda_check))

train_loader = DataLoader(input_array, batch_size=64, shuffle=True)
for epochs in range(num_epochs):
    count = 0
    for tensor, label in train_loader:
        optimizer.zero_grad()
        logits = model(tensor)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss
        count = count + 1
    avg_loss = total_loss/count
    print(f'Epoch: {epochs} + Loss:{avg_loss}')
    total_loss = 0
    count = 0
#saves the model so it can be loaded later
torch.save(model.state_dict(), 'model.pth')


#4-7-26, training loop now uses a Dataloader to use minibatches greater than 1.
# this portion I am still working on, I need the input-layer step to produce a test dataset for the model
#to read after it finishes training

input_array_test = NoteDataset(r"data", split='train')
tensor_test, label_test = input_array_test[0]
logits_test = model(tensor_test.unsqueeze(0))


print(logits_test)