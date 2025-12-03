import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.multiprocessing as mp

import numpy as np

from nn_dmc import *

# Set the number of CPU threads and multiprocessing method
torch.set_num_threads(6)

try:
    mp.set_start_method("spawn", force=True)  # Use fork instead of spawn (better for CUDA)
except RuntimeError:
    pass  # Fork may not work in some environments; ignore if it's already set.

assert torch.cuda.is_available(), "GPU is not available, check the directions above (or disable this assertion to use CPU)"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

system_name = 'h5o3'
decay = 0
descriptor = 'molec_atom_sorted'

if system_name == 'h3o2':
    atom_list = [1,8,1,8,1]
    groups = [[0],[1,2],[3,4]]
elif system_name == 'h5o2':
    atom_list = [1,1,1,8,1,1,8]
    groups = [[0],[1,2,3],[4,5,6]]
elif system_name == 'h5o3':
    atom_list = [8,1,8,1,1,8,1,1]
    groups = [[0,1],[2,3,4],[5,6,7]]
elif system_name == 'h7o4':
    atom_list = [8,1,8,1,1,8,1,1,8,1,1]
    groups = [[0,1],[2,3,4],[5,6,7],[8,9,10]]
elif system_name == 'h9o5':
    atom_list = [8,1,8,1,1,8,1,1,8,1,1,8,1,1]
    groups = [[0,1],[2,3,4],[5,6,7],[8,9,10],[11,12,13]]

#load in training data
train_coords = np.load(f'{system_name}_full_training_cds.npy')
train_energies = np.load(f'{system_name}_full_training_energies.npy')

#calculate training features
X_train_array = molec_atom_sorted_CM(train_coords,atom_list,groups,False,False)
y_train_array = np.log10(train_energies+100)

#load in test data
test_coords = np.load(f'{system_name}_10_test_cds.npy')
test_energies = np.load(f'{system_name}_10_test_energies.npy')

#calculate test features
X_test_array = molec_atom_sorted_CM(test_coords,atom_list,groups,False,False)
y_test_array = np.log10(test_energies+100)

#convert training features to torch tensors
X_train = torch.tensor(X_train_array, dtype=torch.float32)
y_train = torch.tensor(y_train_array, dtype=torch.float32)

full_data = TensorDataset(X_train,y_train)

# Define the ratio for splitting (e.g., 80% training, 20% validation)
train_ratio = 0.9
val_ratio = 1 - train_ratio

# Calculate the number of samples for each split
num_train_samples = int(train_ratio * len(full_data))
num_val_samples = len(full_data) - num_train_samples

# Use random_split to create training and validation datasets
train_data, val_data = random_split(full_data, [num_train_samples, num_val_samples])

train_loader = DataLoader(train_data, shuffle=True, batch_size=64)

val_loader = DataLoader(val_data, shuffle=True, batch_size=num_val_samples)

#convert test features to torch tensors
X_test = torch.tensor(X_test_array, dtype=torch.float32)
y_test = torch.tensor(y_test_array, dtype=torch.float32)

test_data = TensorDataset(X_test,y_test)
test_loader = DataLoader(test_data, shuffle=True, batch_size=64)

X_test = X_test.to(DEVICE)
y_test = y_test.to(DEVICE)

y_test_regular = torch.tensor([(10**i)-100 for i in y_test]) #convert back to cm-1
y_test_regular = y_test_regular.to(DEVICE)

num_atoms = len(atom_list)
hidden = (3*num_atoms-6)*20

# Hyperparameters
input_size = (num_atoms*(num_atoms-1))/2
hidden_size = hidden
output_size = 1
learning_rate = 0.0001
momentum = 0
num_epochs = 1000

# Create the neural network
model = nn.Sequential(nn.Linear(input_size, hidden_size, bias=True),
                    nn.BatchNorm1d(hidden_size),
                    nn.SiLU(),
                    nn.Dropout(p=0),
                    nn.Linear(hidden_size, hidden_size, bias=True),
                    nn.BatchNorm1d(hidden_size),
                    nn.SiLU(),
                    nn.Dropout(p=0),
                    nn.Linear(hidden_size, hidden_size, bias=True),
                    nn.BatchNorm1d(hidden_size),
                    nn.SiLU(),
                    nn.Dropout(p=0),
                    nn.Linear(hidden_size, output_size,bias=True),
                    nn.ReLU()
    )

model = model.to(DEVICE)

#If first training iteration
for layer in model:
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)

# Loss and optimizer
criterion = nn.L1Loss()
MAE = nn.L1Loss()

optimizer = optim.NAdam(model.parameters(), lr=learning_rate, weight_decay = decay)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, threshold = 0.5, threshold_mode = 'abs', verbose=True)

# Training loop

loss_list = []
val_loss_list = []
test_loss_list = []
train_MAE_list = []

epoch = 0

num_batches = num_train_samples // 64

for epoch in range(num_epochs):

    epoch += 1

    losses = 0
    train_MAEs = 0

    for (X_batch, y_batch) in train_loader:

        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        
        y_batch_regular = torch.tensor([(10**i)-100 for i in y_batch])
        y_batch_regular = y_batch_regular.to(DEVICE)

        # Forward pass
        preds = model(X_batch)
        loss = criterion(preds.squeeze(), y_batch)
        
        preds_regular = torch.tensor([(10**i)-100 for i in preds]).to(DEVICE)

        MAE_error = MAE(preds_regular.squeeze(), y_batch_regular)
        train_MAEs += MAE_error.item()

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses += loss.item()

    loss_list.append(losses/num_batches)
    train_MAE_list.append(train_MAEs/num_batches)

    model.eval()
    with torch.no_grad():
        
        for (X_batch, y_batch) in val_loader:

            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            y_batch_regular = torch.tensor([(10**i)-100 for i in y_batch])
            y_batch_regular = y_batch_regular.to(DEVICE)

            val_preds = model(X_batch)
            val_loss = criterion(val_preds.squeeze(), y_batch)

            val_preds_regular = torch.tensor([(10**i)-100 for i in val_preds]).to(DEVICE)

            val_MAE = MAE(val_preds_regular.squeeze(), y_batch_regular)

        val_loss_list.append(val_MAE.cpu().detach().numpy())

        output = model(X_test)

        output_regular = torch.tensor([(10**i)-100 for i in output])
        output_regular = output_regular.to(DEVICE)

        test_loss = criterion(output_regular.squeeze(), y_test_regular)

        test_MAE = MAE(output_regular.squeeze(), y_test_regular)

        test_loss_list.append(test_MAE.cpu().detach().numpy())

        scheduler.step(train_MAEs/num_batches)

             
        print(f'Epoch [{epoch}], Train error: {losses/num_batches:.9f}, Train MAE: {train_MAEs/num_batches:.4f}, Val MAE: {val_MAE:.4f}, Test MAE: {test_MAE:.4f}')

        if optimizer.param_groups[0]['lr'] < 5e-7:
            print('Learning rate less than threshold, stopping training')
            break

torch.save(model.state_dict(),f'{system_name}_nn_model_{descriptor}_{hidden}hidden_{decay}_decay_bn.pth')

losses = np.array([loss_list,train_MAE_list,val_loss_list,test_loss_list])
np.save(f'{system_name}_nn_model_{descriptor}_{hidden}hidden_{decay}_decay_bn_learning_curve.npy', losses)