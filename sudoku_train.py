#!/usr/bin/env python
# coding: utf-8
import torch
import os
import numpy as np
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("Loading the data...")
# ## Load data
def load_sudoku(FILE_PATH):
    quizzes = np.zeros((1000000, 81), np.int32)
    solutions = np.zeros((1000000, 81), np.int32)
    for i, line in enumerate(open(FILE_PATH, 'r').read().splitlines()[1:]):
        quiz, solution = line.split(",")
        for j, q_s in enumerate(zip(quiz, solution)):
            q, s = q_s
            quizzes[i, j] = q
            solutions[i, j] = s
    quizzes = quizzes.reshape((-1, 9, 9))
    solutions = solutions.reshape((-1, 9, 9))
    return quizzes, solutions

quizzes, solutions = load_sudoku(os.path.join("data", "sudoku.csv"))

quizzes = quizzes[:100]
solutions = solutions[:100]

# ## Split data for training, validation, and test set

# Ratio for train, validation, and test set = 9:1:1
# separate train and test set
np.random.seed(10)
test_size = 0.1  # test + val size

index = np.arange(0, quizzes.shape[0])
index_train = np.random.choice(index, size=int((1 - test_size) * len(index)), replace=False)
index_test = np.array(list(set(index).difference(set(index_train))))

X_train = quizzes[index_train]
y_train = solutions[index_train]
X_test = quizzes[index_test]
y_test = solutions[index_test]

# separate val and test set
np.random.seed(11)
val_size = 0.5  # test : val size

index = np.arange(0, X_test.shape[0])
index_val = np.random.choice(index, size=int(val_size * len(index)), replace=False)
index_test = np.array(list(set(index).difference(set(index_val))))

X_val = X_test[index_val]
y_val = y_test[index_val]
X_test = X_test[index_test]
y_test = y_test[index_test]


print("Preprocessing Data...")
# ## Data preparation
# #### Normalization

# Neural network converge way faster with normalize input
def normalize_sudoku(sudoku):
    
    # divide by max - min
    norm_sudoku = sudoku / 9
    
    # centered around 0
    norm_sudoku -= np.mean(norm_sudoku)
    
    return norm_sudoku

def normalize_batch(sudokus):
    
    sudokus = sudokus.astype(np.float)
    
    for i in range(sudokus.shape[0]):
        sudokus[i] = normalize_sudoku(sudokus[i])
        
    return sudokus

X_train = normalize_batch(X_train)
X_val = normalize_batch(X_val)
X_test = normalize_batch(X_test)


# #### Add channel dimension

# The cnn model takes four dimensional inputs: batch, channel, height, and width. Our inputs only have three dimensions, so we have to reshape the inputs.
X_train = X_train.reshape((-1, 1, 9, 9))
X_val = X_val.reshape((-1, 1, 9, 9))
X_test = X_test.reshape((-1, 1, 9, 9))

# #### Reshape target variables

# Also since each cell is a number between 1 to 9, we have to subtract the target by 1 and make it from 0 to 8
y_train = y_train - 1
y_val = y_val - 1
y_test = y_test - 1

# #### Convert to DataLoader
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

def create_dataloader(X, y, batch_size=128, randomize=False):
    '''
    Function to create dataloader for batch training
    '''
    data = TensorDataset(torch.tensor(X, dtype=torch.float), torch.tensor(y, dtype=torch.long))
    
    if randomize:
        sampler = RandomSampler(data)
    else:
        sampler = SequentialSampler(data)
    loader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return loader

train_loader = create_dataloader(X_train, y_train)
val_loader = create_dataloader(X_val, y_val)
test_loader = create_dataloader(X_test, y_test)


# ## Define the model
print("Defining Model")
import torch.nn as nn
import torch.nn.functional as F


class SudokuCNN(nn.Module):
    '''
    Convolution + Batch Normalization + Relu
    '''
    
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        
        # Get the padding size
        if type(kernel_size) == tuple:
            pad_height = kernel_size[0] // 2
            pad_width = kernel_size[1] // 2
        else:
            pad_height = kernel_size // 2
            pad_width = kernel_size // 2
        
        self.padding = nn.ReflectionPad2d((pad_width, pad_width, pad_height, pad_height))
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        
        # Apply reflection padding
        x = self.padding(x)
        
        # Convolution
        x = self.conv(x)
        
        # Batch Normalization
        x = self.batch_norm(x)
        
        # Relu
        x = F.relu(x)
        
        return x

class SudokuModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.first_layer = SudokuCNN(in_channels=1, out_channels=1024, kernel_size=3)
        self.middle_layer = nn.Sequential(
            SudokuCNN(in_channels=1024, out_channels=1024, kernel_size=(3,3)),
            SudokuCNN(in_channels=1024, out_channels=512, kernel_size=(3,3)),
            SudokuCNN(in_channels=512, out_channels=512, kernel_size=(3,3)),
            SudokuCNN(in_channels=512, out_channels=512, kernel_size=(3,3)),
            SudokuCNN(in_channels=512, out_channels=512, kernel_size=(3,3)),
            SudokuCNN(in_channels=512, out_channels=512, kernel_size=(3,3)),
            SudokuCNN(in_channels=512, out_channels=512, kernel_size=(3,3)),
            SudokuCNN(in_channels=512, out_channels=256, kernel_size=(3,3)),
            SudokuCNN(in_channels=256, out_channels=256, kernel_size=(3,3)),
            SudokuCNN(in_channels=256, out_channels=256, kernel_size=(3,3)),
            SudokuCNN(in_channels=256, out_channels=128, kernel_size=(3,3)),
            SudokuCNN(in_channels=128, out_channels=128, kernel_size=(3,3))
        )
        self.last_layer = nn.Conv2d(in_channels=128, out_channels=9, kernel_size=(1,1))
        
    def forward(self, x):
        
        # First layer
        x = self.first_layer(x.float())
        
        # Middle layer
        x = self.middle_layer(x)
        
        # Last layer
        x = self.last_layer(x)
        
        return x

model = SudokuModel()
print(model)
# ## Set up optimizer
import torch.optim as optim
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


# ## Set up loss function and sudoku accuracy
criterion = nn.CrossEntropyLoss()


# We can define the sudoku accuracy as the number of correct answers
def sudoku_accuracy(output, target):
    return torch.mean((torch.sum(torch.argmax(output, dim=1) == (target + 1), dim=(1,2)) == 81).float()).item()


# ## Train the model
def save_model(model, file_name):
    MODEL_PATH = os.path.join(".", "model")
    
    # if directory doesn't exists, create new one
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    file_name = os.path.join(MODEL_PATH, file_name)
    torch.save(model.state_dict(), file_name)
    
def load_model(model, file_name):
    FILE_PATH = os.path.join(os.path.join(".", "model"), file_name)
    
    # if file exists
    if os.path.exists(FILE_PATH):
        model.load_state_dict(torch.load(FILE_PATH))
        
    return model
    

def train(model, num_epochs, gpu=True, save=True, load=True, file_name="model.pth"):
    
    num_batch = (X_train.shape[0] // 128) + 1
    
    if gpu:
        model.to(device)
        
    if load:
        model = load_model(model, file_name)  
    
    train_loss_histories = []
    val_loss_histories = []
    accuracy_histories = []
    
    for epoch in range(num_epochs):

        print()
        print(f"======= Epoch {epoch + 1} / {num_epochs} ======")
        print('Training...')

        train_loss = 0
        n_step = 0

        for step, batch in enumerate(train_loader, 0):
            
            print(f"Train batch {n_step+1}/{num_batch}", end='\r')
            
            batch_data, batch_labels = batch
            
            if gpu:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

            # clear previously calculated gradients
            model.zero_grad()

            # perform a forward pass
            output = model(batch_data)

            # compute loss
            loss = criterion(output, batch_labels)
            train_loss += loss.item()

            # perform backward pass
            loss.backward()

            # step
            optimizer.step()

            # increment step
            n_step += 1

        print(f"Training loss: {train_loss / n_step}")
        train_loss_histories.append(train_loss/n_step)

        print("Validation...")

        # variables for tracking
        val_loss = 0
        accuracy = 0
        n_step = 0

        for batch in val_loader:

            batch_data, batch_labels = batch
            
            if gpu:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

            # tell pytorch to not compute the gradient
            with torch.no_grad():

                # perform a forward pass
                output = model(batch_data)

            # compute loss
            val_loss += criterion(output, batch_labels).item()

            # compute accuracy
            accuracy += sudoku_accuracy(output, batch_labels)

            # increment step
            n_step += 1

        print(f"Validation loss: {val_loss / n_step}")
        val_loss_histories.append(val_loss / n_step)
        
        print(f"Sudoku accuracy: {accuracy / n_step}")
        accuracy_histories.append(accuracy / n_step)
        
        
        
        #########################
        #      Saving model     #
        #########################

        if save:
            save_model(model, file_name)

print("\n\n")
print("Begin training")
num_epochs = 5
train(model, num_epochs, gpu=True, save=True, load=True, file_name="model.pth")