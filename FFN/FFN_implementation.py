#this file combines the data_preprocessing and Combined_Workflow together to perform the training and validation
#steps of the model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from torchmetrics.classification import MulticlassRecall
from torchmetrics.functional.classification import multiclass_f1_score
import pandas as pd
from pathlib import Path


from data_preprocessing_1 import NoteDataset
from Combined_Workflow import CombinedWorkflow

if __name__ == '__main__':
    #establishes the use of GPU in the coda
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #these multipliers are used to alter the size of the hidden layers for each k-fold cross validation, altering the
    #size of the first hidden layer by a multiple of the number of input nodes (the second hidden layer is always
    #half the size of the first hidden layer)
    multipliers = [0.25, 0.5, 1.0, 2.0]

    #establishes the variables we use for training, i.e., the calculated loss and the number of training epochs for
    #each fold
    total_loss = 0
    count = 0
    num_epochs = 50

    #uses the NoteDataset class from the data_preprocessing file to generate the inputs for the 1st layer of the
    #hidden array (i.e., the 1D vector of pixel values)
    # back slash for windows, forward slash for mac (r"../data"), (r"..\data")
    input_array = NoteDataset(r"../data")
    #each mult_indx value is one of the indices in the multipliers array
    for mult_indx in range(0, 4):

        #the "tensor" variable is used to calculate the number of inputs in the nodes below
        tensor, label = input_array[0]
        input_size = tensor.shape[0]

        #num_nodes_ variables are used to establish the number of nodes in each of the hidden layers
        num_nodes_1 = int((multipliers[mult_indx]*tensor.size(0)))
        num_nodes_2 = int(num_nodes_1 / 2)
        print(num_nodes_1)
        print(num_nodes_2)

        #these empty arrays are used later in the program to store the performance metrics for each of the folds
        train_final_recalls, train_final_accuracies, train_final_f1s = [], [], []
        test_final_recalls, test_final_accuracies, test_final_f1s = [], [], []

        #this sets the learning rate hyperparameter
        learn_rate = 0.01
        #this sets the loss as being calculated using cross-entropy
        loss_funct = nn.CrossEntropyLoss()

        #this creates the k-fold object that is used to split the datasets into train and test subsets, re-shuffling the
        #sets into different combinations for each fold.  A k=5 split divides the data into 80% training and 20% test
        kf = KFold(n_splits = 5, shuffle=True, random_state=42)
        all_cm_rows = []

        #this for-loop iterates through the data combinations in each fold while also establising the indices for the
        #total dataset associated with the training data and those with the test data
        for fold, (train_ids, test_ids) in enumerate(kf.split(input_array)):
            print(f"Fold {fold} started")
            total_loss = 0
            count = 0

            #this takes the indices established above associated with each subset of the data and generates a iterator
            #for each dataset
            train_subsample = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsample = torch.utils.data.SubsetRandomSampler(test_ids)

            #This produces the dataloader objects used in the pytorch neural network module, which uses the subsample
            #iterators above to load the divided data from the input_array.  num_workers established the number of subprocesses
            #assigned to this task, and pin_memory allows for asynchronous data transfer to utilized GPU
            train_loader = DataLoader(input_array, batch_size=128, sampler = train_subsample, num_workers=2, pin_memory=use_gpu, persistent_workers=True)
            test_loader = DataLoader(input_array, batch_size=128, sampler = test_subsample, num_workers=2, pin_memory=use_gpu, persistent_workers=True)

            # this portion calls the file which combines the hidden layer and output layer into a single workflow nn.Module that can
            # be called and have its weights and biases modified by the training loop below
            model = CombinedWorkflow(input_size, num_nodes_1, num_nodes_2, num_nodes_2, num_classes=12)
            #this tells the model to use either the CPU or GPU, depending on availability
            model.to(device, non_blocking=True)

            #this calls the Adam optimizer which is used to perform the back-propagation component of the NN training
            #process to adjust the weights and biases of the preceding layers
            optimizer = optim.Adam(model.parameters(), lr=learn_rate)
            #arrays which store the predictions and correct values for each read note
            test_predictions = []
            test_all_targets = []

            #this for-loop carries out the actual training of the model for each epoch
            for epochs in range(num_epochs):
                train_predictions = []
                train_all_targets = []
                correct, total = 0, 0
                for tensor, label in train_loader:
                    #this step indicates that calculations performed using either of these objects should be
                    #performed using the GPU (if available)
                    tensor, label = tensor.to(device, non_blocking=True), label.to(device, non_blocking = True)
                    t_targets = label
                    #this clears any current gradients calculated by the optimizer
                    optimizer.zero_grad()
                    #the logits variable calls a forward-pass through model and produces a 12-object tensor,
                    # containing the values indicating the likelihood
                    #of a class being the correct classification
                    logits = model(tensor)
                    #this is the step where the predited class is actually chosen by taking the class associated with
                    #the maximum value of the logits tensor
                    _, t_prediction = torch.max(logits, 1)
                    #adds predicted and correct note labels to each array for later comparison
                    train_predictions.append(t_prediction)
                    train_all_targets.append(t_targets)
                    #calculates the error using the cross-entropy method indicated earlier for the upcoming backpropagation
                    #step.  Note, the pytorch nn.CrossEntropy automatically appies softmax function to the raw logit scores
                    #in its calculations
                    loss = loss_funct(logits, label)
                    #initiates the backpropagation and gradient calculation steps
                    loss.backward()
                    optimizer.step()
                    #adds the calculated loss to the total loss
                    total_loss += loss.item()
                    count = count + 1
                    total += t_targets.size(0)
                    #adds the prediction to "correct" count if it matches the actual label
                    correct += (t_prediction == t_targets).sum().item()
                #adds predictions and targets to the arrays following the completion of the training epoch
                train_predictions = torch.cat(train_predictions)
                train_all_targets = torch.cat(train_all_targets)
                #determines the average loss per iteration of the epoch
                avg_loss = total_loss/count
                print(f'Epoch: {epochs} + Loss:{avg_loss}')
                total_loss = 0
                count = 0
            #saves the model so it can be loaded later
            torch.save(model.state_dict(), 'model.pth')
            #once the fold is completed, this block of code calculates the accuracy, recall, and f1-score metrics
            #for that fold's training data, then outputs each of them to a specific .csv file
            recall_metric = MulticlassRecall(num_classes=12, average='macro')
            recall = recall_metric(train_predictions.cpu(), train_all_targets.cpu())
            f1_metric = multiclass_f1_score(train_predictions.cpu(), train_all_targets.cpu(), num_classes=12, average = 'macro')
            print(f'Training Accuracy for fold {fold}: {100.0 * correct / total:.2f}%')
            print(f'Training Recall for fold {fold}: {100.0 * recall:.2f}%')
            print(f'Training F1-Score for fold {fold}: {100.0 * f1_metric:.2f}%')
            train_final_accuracies.append((correct / total))
            train_final_recalls.append(recall.item())
            train_final_f1s.append(f1_metric.item())

            correct, total = 0, 0
            #after indicating that no further gradient calcualtions should take place, this block of code uses the
            # trained model on the fold's test data, evaluating the model's ability to make predictions on novel data
            with torch.no_grad():
                for tensor, label in test_loader:
                    tensor, label = tensor.to(device, non_blocking = True), label.to(device, non_blocking = True)
                    inputs, targets = tensor, label
                    pred_logits = model(inputs)
                    #note, raw logit scores are used for predictions, as they give the same prediction results as those
                    #calculated using the softmax function without additional computational expense
                    _ , prediction = torch.max(pred_logits, 1)
                    test_predictions.append(prediction)
                    test_all_targets.append(targets)
                    total += targets.size(0)
                    correct += (prediction == targets).sum().item()
            #determines the performance metrics for the test data
            test_predictions = torch.cat(test_predictions)
            test_all_targets = torch.cat(test_all_targets)
            recall_metric = MulticlassRecall(num_classes=12, average='macro')
            recall = recall_metric(test_predictions.cpu(), test_all_targets.cpu())
            f1_metric = multiclass_f1_score(test_predictions.cpu(), test_all_targets.cpu(), num_classes=12, average = 'macro')
            test_final_accuracies.append((correct / total))
            test_final_recalls.append(recall.item())
            test_final_f1s.append(f1_metric.item())
            print(f'Test Accuracy for fold {fold}: {100.0 * correct / total:.2f}%')
            print(f'Test Recall for fold {fold}: {100.0 * recall:.2f}%')
            print(f'Test F1-Score for fold {fold}: {100.0 * f1_metric:.2f}%')

            for pred, true in zip(test_predictions.tolist(), test_all_targets.tolist()):
                all_cm_rows.append({'fold': fold, 'true_label': true, 'predicted_label': pred})

        folds = list(range(len(train_final_accuracies)))
        #these final code blocks output the training and test data matrics to 3 .csv files, as well as
        #the confusion matrix data for the last completed fold
        accuracy_path = 'accuracy_results.csv'
        accuracy_path = Path(accuracy_path)
        if accuracy_path.is_file() == False:
            accuracy_df = pd.DataFrame({
                'fold': folds,
                'train_accuracy': train_final_accuracies,
                'test_accuracy': test_final_accuracies
            }   )
            accuracy_df.to_csv(accuracy_path, index=False)
        else:
            accuracy_df = pd.DataFrame({
                'fold': folds,
                'train_accuracy': train_final_accuracies,
                'test_accuracy': test_final_accuracies
            }   )
            accuracy_df.to_csv(accuracy_path, mode ='a', index=False)

        recall_path = 'recall_results.csv'
        recall_path = Path(recall_path)
        if recall_path.is_file() == False:
            recall_df = pd.DataFrame({
                'fold': folds,
                'train_recall': train_final_recalls,
                'test_recall': test_final_recalls
            })
            recall_df.to_csv(recall_path, index=False)
        else:
            recall_df = pd.DataFrame({
                'fold': folds,
                'train_recall': train_final_recalls,
                'test_recall': test_final_recalls
            })
            recall_df.to_csv(recall_path, mode = 'a', index=False)


        f1_path = 'f1_results.csv'
        f1_path = Path(f1_path)
        if f1_path.is_file() == False:
            f1_df = pd.DataFrame({
                'fold': folds,
                'train_f1': train_final_f1s,
                'test_f1': test_final_f1s
            })
            f1_df.to_csv(f1_path, index=False)
        else:
            f1_df = pd.DataFrame({
                'fold': folds,
                'train_f1': train_final_f1s,
                'test_f1': test_final_f1s
            })
            f1_df.to_csv(f1_path, mode = 'a', index=False)


        cm_df = pd.DataFrame(all_cm_rows)
        cm_df.to_csv('confusion_matrix_data.csv', index=False)

        print("\nCSV files saved: accuracy_results.csv, recall_results.csv, f1_results.csv, confusion_matrix_data.csv")
