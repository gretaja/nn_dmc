"""
Functions for NN training and DMC analysis
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

import numpy as np
import pyvibdmc as pv
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def unsorted_CM(molecule_coords, atomic_numbers, full=False, revised=False):
    """
    Calculate the lower triangle of the unsorted Coulomb matrix for a set of structures.
    
    Parameters:
    molecule_coords (np.ndarray): An array of shape (n_molecules, n_atoms, 3) containing the Cartesian coordinates.
    atomic_numbers (np.ndarray): An array of shape (n_atoms,) containing the atomic numbers.
    full (Boolean): True if you want the full Coulomb matrix returned instead of just the lower triangle.
    revised (Boolean): True if you want the diagonal elements to be just the nuclear charges, and the
                        off-diagonal terms to be the inverse distances between pairs of atoms.

    Returns:
    np.ndarray: An array of lower triangles of the unsorted coulomb matrices of shape (n_molecules, (n_atoms x n_atoms-1)/2).
    """

    n_molecules, n_atoms, _ = molecule_coords.shape
    
    # Initialize the Coulomb matrices
    coulomb_matrices = np.zeros((n_molecules, n_atoms, n_atoms))
    
    # Compute pairwise distance matrices
    distance_matrices = np.linalg.norm(molecule_coords[:, :, np.newaxis] - molecule_coords[:, np.newaxis, :], axis=-1)
    
    # Calculate off-diagonal elements
    Z_product = atomic_numbers[:, np.newaxis] * atomic_numbers[np.newaxis, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        if revised == True:
            numerator = 1
        else:
            numerator = Z_product

        coulomb_matrices = numerator / distance_matrices
        coulomb_matrices[distance_matrices == 0] = 0  # Handle division by zero
    
    # Calculate diagonal elements
    if revised == True:
        diagonal_elements = atomic_numbers
    else:
        diagonal_elements = 0.5 * atomic_numbers**2.4

    np.einsum('ijj->ij', coulomb_matrices)[:] = diagonal_elements
    
    lower_triangles = np.zeros((n_molecules, n_atoms * (n_atoms - 1) // 2))
    
    for a in range(n_molecules):

        lower_triangles[a] = coulomb_matrices[a][np.tril_indices(n_atoms, -1)]
    
    if full == True:
        return coulomb_matrices
    
    else:
        return lower_triangles


def molec_sorted_CM(molecule_coords, atomic_numbers, groups, full=False, revised=False):
    """
    Calculate the lower triangle of the molecule sorted Coulomb matrix for a set of structures.
    
    Parameters:
    molecule_coords (np.ndarray): An array of shape (n_molecules, n_atoms, 3) containing the Cartesian coordinates.
    atomic_numbers (np.ndarray): An array of shape (n_atoms,) containing the atomic numbers.
    groups (list): A list of lists containing the groupings of atoms in molecular fragments.
    full (Boolean): True if you want the full Coulomb matrix returned instead of just the lower triangle.
    revised (Boolean): True if you want the diagonal elements to be just the nuclear charges, and the
                        off-diagonal terms to be the inverse distances between pairs of atoms.

    Returns:
    np.ndarray: An array of input features for the nn model of shape (n_molecules, (n_atoms x n_atoms-1)/2).
    """
    n_molecules, n_atoms, _ = molecule_coords.shape

    if revised == True:
        coulomb_matrices = unsorted_CM(molecule_coords, atomic_numbers, full=True, revised=True)
    else:
        coulomb_matrices = unsorted_CM(molecule_coords, atomic_numbers, full=True, revised=False)
    
    lower_triangles = np.zeros((n_molecules, n_atoms * (n_atoms - 1) // 2))
    sorted_CMs = np.zeros((n_molecules, n_atoms, n_atoms))
    
    for a in range(n_molecules):
        
        norms = []
        for g in groups:
            norms.append(np.sum([np.linalg.norm(coulomb_matrices[a][i]) for i in g]))
            
        sorted_indices = np.argsort(norms)[::-1]

        new_index = []
        for i in sorted_indices:
            for j in groups[i]:
                new_index.append(j)

        sorted_CMs[a] = coulomb_matrices[a][:, new_index][new_index, :]
        
        lower_tri = np.tril_indices(n_atoms, -1)
        
        lower_triangles[a] = sorted_CMs[a][lower_tri]
    
    if full == True:
        return sorted_CMs
    
    else:
        return lower_triangles


def molec_atom_sorted_CM(molecule_coords, atomic_numbers, groups, full=False, revised=False):
    """
    Calculate the lower triangle of the molec. + atom sorted Coulomb matrix for a set of structures.
    
    Parameters:
    molecule_coords (np.ndarray): An array of shape (n_molecules, n_atoms, 3) containing the Cartesian coordinates.
    atomic_numbers (np.ndarray): An array of shape (n_atoms,) containing the atomic numbers.
    groups (list): A list of lists containing the groupings of atoms in molecular fragments.
    full (Boolean): True if you want the full Coulomb matrix returned instead of just the lower triangle.
    revised (Boolean): True if you want the diagonal elements to be just the nuclear charges, and the
                        off-diagonal terms to be the inverse distances between pairs of atoms.

    Returns:
    np.ndarray: An array of input features for the nn model of shape (n_molecules, (n_atoms x n_atoms-1)/2).
    """
    n_molecules, n_atoms, _ = molecule_coords.shape

    if revised == True:
        coulomb_matrices = unsorted_CM(molecule_coords, atomic_numbers, full=True, revised=True)
    else:
        coulomb_matrices = unsorted_CM(molecule_coords, atomic_numbers, full=True, revised=False)
    
    group_sizes = [len(i) for i in groups]
    group_starts = [i[0] for i in groups]
    
    lower_triangles = np.zeros((n_molecules, n_atoms * (n_atoms - 1) // 2))
    sorted_CMs = np.zeros((n_molecules, n_atoms, n_atoms))
    
    for a in range(n_molecules):
        reorder = []
        group_norms = []
        for i in range(len(group_sizes)):
            start = group_starts[i]
            end = start+group_sizes[i]
            reorder.append(np.argsort([np.linalg.norm(coulomb_matrices[a][j]) for j in range(start,end)])+start)
            
            group_norm = np.sum([np.linalg.norm(coulomb_matrices[a][j]) for j in range(start,end)])
            group_norms.append(group_norm)

        group_order = np.argsort(group_norms)

        sorted_indices = []
        for i in range(len(group_sizes)):
            sorted_indices.append(reorder[group_order[i]])

        sorted_CMs[a] = coulomb_matrices[a][:, np.concatenate(sorted_indices)][np.concatenate(sorted_indices), :]

        lower_tri = np.tril_indices(n_atoms, -1)
        
        lower_triangles[a] = sorted_CMs[a][lower_tri]
    
    if full == True:
        return sorted_CMs
    
    else:
        return lower_triangles
    

def select_molec_atom_sorted_CM(molecule_coords, atomic_numbers, groups, select_groups, full=False, revised=False):
    """
    Calculate the lower triangle of the molec. + atom sorted Coulomb matrix for a set of structures,
    only considering permutationally invarient molecules.
    
    Parameters:
    molecule_coords (np.ndarray): An array of shape (n_molecules, n_atoms, 3) containing the Cartesian coordinates.
    atomic_numbers (np.ndarray): An array of shape (n_atoms,) containing the atomic numbers.
    groups (list): A list of lists containing the groupings of atoms in molecular fragments.
    select_groups (list): A list of indices for the groups that are to be considered in sorting, ex. [0,2].
    full (Boolean): True if you want the full Coulomb matrix returned instead of just the lower triangle.
    revised (Boolean): True if you want the diagonal elements to be just the nuclear charges, and the
                        off-diagonal terms to be the inverse distances between pairs of atoms.

    Returns:
    np.ndarray: An array of input features for the nn model of shape (n_molecules, (n_atoms x n_atoms-1)/2).
    """
    n_molecules, n_atoms, _ = molecule_coords.shape

    if revised == True:
        coulomb_matrices = unsorted_CM(molecule_coords, atomic_numbers, full=True, revised=True)
    else:
        coulomb_matrices = unsorted_CM(molecule_coords, atomic_numbers, full=True, revised=False)
    
    group_sizes = [len(i) for i in groups]
    group_starts = [i[0] for i in groups]
    
    lower_triangles = np.zeros((n_molecules, n_atoms * (n_atoms - 1) // 2))
    sorted_CMs = np.zeros((n_molecules, n_atoms, n_atoms))
    
    for a in range(n_molecules):
        reorder = []
        group_norms = []
        for i in range(len(group_sizes)):
            start = group_starts[i]
            end = start+group_sizes[i]
            reorder.append(np.argsort([np.linalg.norm(coulomb_matrices[a][j]) for j in range(start,end)])+start)
            
            group_norm = np.sum([np.linalg.norm(coulomb_matrices[a][j]) for j in range(start,end)])
            group_norms.append(group_norm)

        if group_norms[select_groups[0]] < group_norms[select_groups[1]]:
            reorder[select_groups[0]], reorder[select_groups[1]] = reorder[select_groups[1]], reorder[select_groups[0]]

        sorted_CMs[a] = coulomb_matrices[a][:, np.concatenate(reorder)][np.concatenate(reorder), :]

        lower_tri = np.tril_indices(n_atoms, -1)
        
        lower_triangles[a] = sorted_CMs[a][lower_tri]
    
    if full == True:
        return sorted_CMs
    
    else:
        return lower_triangles

def atom_sorted_CM(molecule_coords,atomic_numbers, full=False, revised=False):
    """
    Calculate the lower triangle of the atom sorted Coulomb matrix for a set of structures.
    
    Parameters:
    molecule_coords (np.ndarray): An array of shape (n_molecules, n_atoms, 3) containing the Cartesian coordinates.
    atomic_numbers (np.ndarray): An array of shape (n_atoms,) containing the atomic numbers.
    full (Boolean): True if you want the full Coulomb matrix returned instead of just the lower triangle.
    revised (Boolean): True if you want the diagonal elements to be just the nuclear charges, and the
                        off-diagonal terms to be the inverse distances between pairs of atoms.

    Returns:
    np.ndarray: An array of input features for the nn model of shape (n_molecules, (n_atoms x n_atoms-1)/2).
    """
    n_molecules, n_atoms, _ = molecule_coords.shape

    if revised == True:
        coulomb_matrices = unsorted_CM(molecule_coords, atomic_numbers, full=True, revised=True)
    else:
        coulomb_matrices = unsorted_CM(molecule_coords, atomic_numbers, full=True, revised=False)

    lower_triangles = np.zeros((n_molecules, n_atoms * (n_atoms - 1) // 2))
    sorted_CMs = np.zeros((n_molecules, n_atoms, n_atoms))

    for a in range(n_molecules):

        l1_norms = np.linalg.norm(coulomb_matrices[a], ord=1, axis=0)  # Calculate L1 norms of each column
        sorted_indices = np.argsort(l1_norms)[::-1]  # Sort in descending order of L1 norms
        sorted_CMs[a] = coulomb_matrices[a][:, sorted_indices][sorted_indices, :]
        
        lower_tri = np.tril_indices(n_atoms, -1)
            
        lower_triangles[a] = sorted_CMs[a][lower_tri]
    
    if full == True:
        return sorted_CMs
    
    else:
        return lower_triangles
    
def standardize_inputs(X_path):
    """
    Saves .npy files for the standardized data, means, and standard deviations, 
    given the path to the input data of interest (.npy file)
    """
    X_data = np.load(f'{X_path}')

    means = np.mean(X_data,axis=0)
    stds = np.std(X_data,axis=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        standard_X_data = (X_data - means)/stds

    np.save(f'{X_path}_means.npy',means)
    np.save(f'{X_path}_stds.npy',stds)
    np.save(f'{X_path}_standardized.npy',standard_X_data)
