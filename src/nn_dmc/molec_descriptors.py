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

import networkx as nx
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.distance import cdist

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
    
def determine_molecule_groups(coords, atom_types, tolerance=0.7):
    """
    Determine molecular groupings using a fixed covalent radius heuristic for O/H atoms only.
    
    Parameters:
        coords (np.ndarray): Shape (n_atoms, 3)
        tolerance (float): Distance buffer added to covalent radius sum

    Returns:
        List[List[int]]: List of atom index groupings per molecule
    """   
    n_atoms = len(atom_types)
    r_cov = {'O': 1.2472, 'H': 0.5858}
    
    G = nx.Graph()
    G.add_nodes_from(range(n_atoms))
    
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dist = np.linalg.norm(coords[i] - coords[j])
            r_sum = r_cov[atom_types[i]] + r_cov[atom_types[j]] + tolerance
            if dist < r_sum:
                G.add_edge(i, j)
    
    return [sorted(list(comp)) for comp in nx.connected_components(G)]

def rule_based_fragmenter(coords, atom_types, tolerance=0.7559):
    """
    Rule-based fragmenter for systems with only O and H atoms.
    Groups atoms into H2O (O with 2 H) or OH- (O with 1 H).
    
    Parameters:
        coords: np.ndarray of shape (n_atoms, 3)
        atom_types: list of atom types (e.g., ['O', 'H', 'H', ..., 'O'])
        tolerance: extra buffer added to covalent radius sums
    
    Returns:
        List of fragments (each a list of atom indices)
    """
    coords = np.asarray(coords)

    # Covalent radii
    r_cov = {'O': 1.2472, 'H': 0.5858}
    bond_thresh = r_cov['O'] + r_cov['H'] + tolerance

    # Build distance matrix
    dists = squareform(pdist(coords))

    assigned = set()
    fragments = []

    for i, atom in enumerate(atom_types):
        if atom == 'O' and i not in assigned:
            # Find unassigned H atoms within bond distance
            candidates = [
                j for j, at in enumerate(atom_types)
                if at == 'H' and j not in assigned and dists[i, j] < bond_thresh
            ]

            # Sort by distance to prefer closest Hs
            candidates = sorted(candidates, key=lambda j: dists[i, j])

            if len(candidates) >= 2:
                frag = [i, candidates[0], candidates[1]]  # H2O
            elif len(candidates) == 1:
                frag = [i, candidates[0]]  # OH-
            else:
                frag = [i]  # Lone O (rare edge case)

            fragments.append(sorted(frag))
            assigned.update(frag)

    # Any leftover Hs not bonded to O (edge case)
    for i, atom in enumerate(atom_types):
        if atom == 'H' and i not in assigned:
            fragments.append([i])
            assigned.add(i)

    return fragments

def strict_fragmenter_OH_H2O(coords, atom_types):
    """
    Groups atoms strictly into OH- or H2O without using any distance thresholds or radii.
    
    Parameters:
        coords: np.ndarray, shape (n_atoms, 3)
        atom_types: list or array of 'O' and 'H'
    
    Returns:
        List of fragments (each a list of atom indices)
    """
    coords = np.asarray(coords)
    atom_types = np.array(atom_types)

    # Get indices of O and H atoms
    oxygen_indices = np.where(atom_types == 'O')[0]
    hydrogen_indices = np.where(atom_types == 'H')[0]

    oxygen_coords = coords[oxygen_indices]
    fragments = []
    
    # Map from each oxygen to the Hs assigned to it
    oxygen_to_hydrogens = {idx: [] for idx in oxygen_indices}

    # Assign each H to its closest O
    for h_idx in hydrogen_indices:
        h_coord = coords[h_idx]
        dists = np.linalg.norm(oxygen_coords - h_coord, axis=1)
        closest_o_idx = oxygen_indices[np.argmin(dists)]
        oxygen_to_hydrogens[closest_o_idx].append(h_idx)

    # Build fragments: O with its 1–2 closest Hs
    for o_idx, h_list in oxygen_to_hydrogens.items():
        if len(h_list) == 0:
            raise ValueError(f"Oxygen atom {o_idx} has no hydrogen assigned!")
        elif len(h_list) > 2:
            # To strictly enforce OH- or H2O, keep closest two
            o_coord = coords[o_idx]
            h_coords = coords[h_list]
            dists = np.linalg.norm(h_coords - o_coord, axis=1)
            h_sorted = [h for _, h in sorted(zip(dists, h_list))]
            h_list = h_sorted[:2]
        fragments.append(sorted([o_idx] + h_list))

    return fragments

def constrained_fragmenter_OH_H2O(coords, atom_types):
    coords = np.asarray(coords)
    atom_types = np.array(atom_types)

    oxygen_indices = np.where(atom_types == 'O')[0].tolist()
    hydrogen_indices = np.where(atom_types == 'H')[0].tolist()

    if len(oxygen_indices) != 6 or len(hydrogen_indices) != 11:
        raise ValueError("Expected 6 O and 11 H atoms for OH-(H2O)5 system.")

    # Distance matrix: (6 O) x (11 H)
    dist_matrix = cdist(coords[oxygen_indices], coords[hydrogen_indices])

    # Find the OH⁻ oxygen (closest H overall)
    oh_o_idx = None
    oh_h_idx = None
    min_dist = np.inf
    for i, o in enumerate(oxygen_indices):
        h_idx = np.argmin(dist_matrix[i])
        if dist_matrix[i, h_idx] < min_dist:
            min_dist = dist_matrix[i, h_idx]
            oh_o_idx = oxygen_indices[i]
            oh_h_idx = hydrogen_indices[h_idx]

    # Assign OH⁻ group
    fragments = [[oh_o_idx, oh_h_idx]]
    used_h = {oh_h_idx}
    used_o = {oh_o_idx}

    # Prepare remaining O and H indices
    remaining_oxygens = [o for o in oxygen_indices if o != oh_o_idx]
    remaining_hydrogens = [h for h in hydrogen_indices if h != oh_h_idx]

    # Track assignments
    for o in remaining_oxygens:
        # Find 2 nearest *available* hydrogens
        o_idx = oxygen_indices.index(o)
        h_distances = [(dist_matrix[o_idx, hydrogen_indices.index(h)], h) 
                       for h in remaining_hydrogens if h not in used_h]
        h_distances.sort()
        if len(h_distances) >= 2:
            h1, h2 = h_distances[0][1], h_distances[1][1]
            fragments.append(sorted([o, h1, h2]))
            used_h.update([h1, h2])
        else:
            raise RuntimeError("Not enough hydrogens left to form valid H₂O group.")

    return sorted(fragments)

def molec_atom_sorted_CM_new(molecule_coords, atomic_numbers, atom_types, full=False, revised=False):
    """
    Calculate sorted Coulomb matrix (lower triangle or full) with automatic OH-based grouping.
    
    Parameters:
        molecule_coords (np.ndarray): Shape (n_molecules, n_atoms, 3)
        atomic_numbers (np.ndarray): Shape (n_atoms,)
        atom_types (List[str]): Length n_atoms, entries 'O' or 'H'
        full (bool): Return full sorted Coulomb matrix
        revised (bool): Use revised CM form

    Returns:
        np.ndarray: Shape (n_molecules, ...) depending on full=True/False
    """
    n_molecules, n_atoms, _ = molecule_coords.shape

    if revised:
        coulomb_matrices = unsorted_CM(molecule_coords, atomic_numbers, full=True, revised=True)
    else:
        coulomb_matrices = unsorted_CM(molecule_coords, atomic_numbers, full=True, revised=False)

    lower_triangles = np.zeros((n_molecules, n_atoms * (n_atoms - 1) // 2))
    sorted_CMs = np.zeros((n_molecules, n_atoms, n_atoms))

    for a in range(n_molecules):
        coords = molecule_coords[a]
        groups = constrained_fragmenter_OH_H2O(coords, atom_types)

        reorder = []
        group_norms = []
        for group in groups:
            norms = [np.linalg.norm(coulomb_matrices[a][j]) for j in group]
            reorder.append(np.array(group)[np.argsort(norms)])
            group_norms.append(np.sum(norms))

        group_order = np.argsort(group_norms)
        sorted_indices = [reorder[i] for i in group_order]
        sorted_indices = np.concatenate(sorted_indices)

        sorted_CMs[a] = coulomb_matrices[a][:, sorted_indices][sorted_indices, :]

        lower_tri = np.tril_indices(n_atoms, -1)
        lower_triangles[a] = sorted_CMs[a][lower_tri]

    return sorted_CMs if full else lower_triangles
    
def standardize_inputs(X_path):
    """
    Saves .npy files for the standardized data, means, and standard deviations, 
    given the path to the input data of interest (.npy file)
    """
    X_data = np.load(f'{X_path}.npy')

    means = np.mean(X_data,axis=0)
    stds = np.std(X_data,axis=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        standard_X_data = (X_data - means)/stds

    np.save(f'{X_path}_means.npy',means)
    np.save(f'{X_path}_stds.npy',stds)
    np.save(f'{X_path}_standardized.npy',standard_X_data)
