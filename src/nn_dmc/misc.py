"""
Miscellaneous functions for DMC
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pyvibdmc as pv
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def make_training_set(ml_path, system, isomers, sim_types, sim_nums):
    """
    Function that takes in specified MOB-ML simulations for the isomers of OH-W5
    to form a training data set for NN training

    example input:
    ml_path = 'C:/Users/gjaco/OneDrive/Documents/UW/McCoy Group/Machine Learning'
    system = 'h11o6'
    isomers = ['c5']
    sim_types = ['min', 'wide', 'reg']
    sim_nums = [[1], [1], [1,2,3,4,5]]

    Returns coordinates array and energies array 
    """
    path = f'{ml_path}/{system}/training_data'

    total_cds = []
    total_energies = []

    for isom in isomers:
        for i, sim in enumerate(sim_types):
            for num in sim_nums[i]:
                cds = np.load(f'{path}/{system}_{isom}_mobml_{sim}_sim_{num}_training_cds.npy')
                energies = np.load(f'{path}/{system}_{isom}_mobml_{sim}_sim_{num}_training_energies.npy')

                total_cds.append(cds)
                total_energies.append(energies)

    total_cds = np.concatenate(total_cds)
    total_energies = np.concatenate(total_energies)

    return total_cds, total_energies

def read_xyz_file(filename):
    """
    Reads in a .xyz file (in Bohr) and returns an nx3 numpy array of the coordinates in Angstroms
    """
    au_to_ang = 0.529177249

    # Read the XYZ file
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    # Number of atoms from the first line
    num_atoms = int(lines[0].strip())
    
    # Extracting the Cartesian coordinates
    coordinates = []
    for line in lines[2:2 + num_atoms]:
        _, x, y, z = line.split()
        coordinates.append([float(x), float(y), float(z)])
    
    # Convert to an Nx3 numpy array
    coordinates_array = np.array(coordinates)/au_to_ang
    return coordinates_array

def save_xyz_file(filename, atoms, coordinates):
    """
    Save an XYZ file with given atoms and coordinates.

    Parameters:
    - filename: The name of the file to save.
    - atoms: A list of atom symbols.
    - coordinates: A list of lists, each containing x, y, and z coordinates.

    Example:
    save_xyz_file('molecule.xyz', ['O', 'H', 'H'], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    """
    with open(filename, 'w') as file:
        file.write(f"{len(atoms)}\n")
        file.write("\n")

        for atom, coord in zip(atoms, coordinates):
            file.write(f"{atom} {' '.join(map(str, coord))}\n")
