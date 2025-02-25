import torch
import torch.nn as nn
import numpy as np

from nn_dmc import *

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

num_atoms = len(atom_list)
hidden = (3*num_atoms-6)*20

input_size = (num_atoms*(num_atoms-1))/2
hidden_size = hidden
output_size = 1

model = nn.Sequential(nn.Linear(input_size, hidden_size, bias=True),
                    nn.BatchNorm1d(hidden_size),
                    nn.SiLU(),
                    nn.Dropout(),
                    nn.Linear(hidden_size, hidden_size, bias=True),
                    nn.BatchNorm1d(hidden_size),
                    nn.SiLU(),
                    nn.Dropout(),
                    nn.Linear(hidden_size, hidden_size, bias=True),
                    nn.BatchNorm1d(hidden_size),
                    nn.SiLU(),
                    nn.Dropout(),
                    nn.Linear(hidden_size, output_size,bias=True),
                    nn.ReLU()

)

#load the model's state dictionary from the saved file
model.load_state_dict(torch.load(f'{system_name}_nn_model_{descriptor}_{hidden}hidden_{decay}_decay_bn.pth',map_location=torch.device('cpu')))

# Put the model in evaluation mode
model.eval()

def cart_to_pot(cds):
    """
    Calculates the potential energies of input geometries for use in DMC simulations

    Parameters:
    cds (np.ndarray): An array of shape (n_molecules, n_atoms, 3) containing the Cartesian coordinates in Bohr.

    Returns:
    np.ndarray: An array shape (n_molecules, ) of potenial energies in a.u..
    """

    features = molec_atom_sorted_CM(cds,atom_list,groups,False,False)
      
    energy = model(torch.tensor(np.array(features),dtype=torch.float32))
    #convert output back to unshifted energy in a.u.
    energy_unshifted = torch.tensor([(10**(i)-100)/219474.63136320 for i in energy])   
    return energy_unshifted.detach().numpy().reshape(len(cds))