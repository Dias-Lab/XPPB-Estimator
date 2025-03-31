import numpy as np
from scipy.spatial.distance import pdist, squareform
from collections import Counter
from Bio.PDB import PDBParser, Select, PDBIO
import freesasa
import os
from pathlib import Path

def load_param(filename):

    param = {}
    
    with open(filename, 'r') as file:
        for line in file:
            # Strip whitespace and split the line into atom name and radius
            parts = line.strip().split(':')
            if len(parts) == 2:
                atom_name, radius = parts
                # Convert radius to float and add to dictionary
                param[atom_name] = float(radius)
            else:
                print("ERROR parsing param file", filename)

    return param


def extract_coordinates_and_radii_from_pdb(structure):
    """
    Extract atom sp types, radii, and coordinates from a Bio.PDB structure object.

    Parameters:
    structure (str): Bio.PDB structure object.

    Returns:
    tuple: A tuple containing:
        - atom_types (np.array): Array of atom types.
        - radii (np.array): Array of atom radii.
        - coordinates (np.array): Nx3 array of x, y, z coordinates for each atom.
    """
    
    atom_radii = load_param("pdb_radii.param")
    atom_names = []
    coordinates = []
    radii = []

    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    atom_name = atom.get_name()
                    
                    # Check if the atom name is in the radii dictionary
                    if atom_name in atom_radii:
                        coord = atom.get_coord()
                        coordinates.append(coord)
                        atom_names.append(atom_name)
                        
                        # Get the radius for this atom
                        radius = atom_radii.get(atom_name)
                        radii.append(radius)

    return np.array(atom_names), np.array(radii), np.array(coordinates)



def extract_coordinates_and_radii_from_mol2(filename):
    """
    Extract atom sp types, radii, and coordinates from a MOL2 file.

    Parameters:
    mol2_file (str): Path to the MOL2 file.

    Returns:
    tuple: A tuple containing:
        - atom_types (np.array): Array of atom sp types.
        - radii (np.array): Array of atom radii.
        - coordinates (np.array): Nx3 array of x, y, z coordinates for each atom.
    """
    
    atom_radii = load_param("mol2_radii.param")

    atom_types = []
    coordinates = []
    radii = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        
    atom_section = False
    for line in lines:
        if line.startswith('@<TRIPOS>ATOM'):
            atom_section = True
            continue
        elif line.startswith('@<TRIPOS>BOND'):
            break
        
        if atom_section:
            parts = line.split()
            if len(parts) >= 9 and parts[5] in atom_radii:
                # Get radius
                atom_type = parts[5]
                radius = atom_radii[atom_type]
                radii.append(radius)
                atom_types.append(atom_type)
                coordinates.append([float(parts[2]), float(parts[3]), float(parts[4])])

    return np.array(atom_types), np.array(radii), np.array(coordinates)

def calculate_pairwise_distances_intermolecular(coordinates1, coordinates2, return_dict=False):
    """
    Calculate Euclidean distances between all points in coordinates1 and all points in coordinates2.
    
    Parameters:
    coordinates1 (np.array): Array of shape (n, 3) where n is the number of points
                              and each row represents the x, y, z coordinates of a point.
    coordinates2 (np.array): Array of shape (m, 3) where m is the number of points
                              and each row represents the x, y, z coordinates of a point.
    
    Returns:
    dict: A dictionary where keys are tuples (i, j) representing the index of an atom in coordinates1
          and the index of an atom in coordinates2, and values are the calculated distances.
    """
    # Ensure inputs are numpy arrays
    coordinates1 = np.asarray(coordinates1)
    coordinates2 = np.asarray(coordinates2)
    
    # Check if inputs have the correct shapes
    if coordinates1.ndim != 2 or coordinates1.shape[1] != 3:
        raise ValueError("coordinates1 must be a 2D array with shape (n, 3)")
    if coordinates2.ndim != 2 or coordinates2.shape[1] != 3:
        raise ValueError("coordinates2 must be a 2D array with shape (m, 3)")
    
    # Iterate through all pairs of points in coordinates1 and coordinates2
    if return_dict == True:
        # Initialize dictionary to store distances
        distance_dict = {}
        for i in range(len(coordinates1)):
            for j in range(len(coordinates2)):
                # Calculate Euclidean distance
                diff = coordinates1[i] - coordinates2[j]
                distance = np.sqrt(np.sum(diff**2))

                # Store distance in dictionary with key (i, j)
                distance_dict[(i, j)] = distance

        return distance_dict
    else:
        # Calculate pairwise distances using broadcasting
        diff = coordinates1[:, np.newaxis, :] - coordinates2[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=-1))    
    
        return distances


def calculate_pairwise_distances_intramolecular(coordinates, return_dict=False):
    """
    Calculate Euclidean distances between all pairs of points in the given coordinates.
    
    Parameters:
    coordinates (np.array): Array of shape (n, 3) where n is the number of points
                            and each row represents the x, y, z coordinates of a point.
    
    Returns:
    np.array: Array of pairwise distances, excluding self-distances and repeated pairs.
    """
    # Ensure input is a numpy array
    coordinates = np.asarray(coordinates)
    
    # Check if the input has the correct shape
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("Input must be a 2D array with shape (n, 3)")
    
    # Calculate pairwise distances
    distances = pdist(coordinates)
    
    # Convert condensed distance matrix to square form
    square_distances = squareform(distances)
    n = len(coordinates)
        
    if return_dict == True:
        # Create dictionary
        distance_dict = {}
       
        for i in range(n):
            for j in range(i+1, n):  # Start from i+1 to avoid duplicates and self-distances
                distance_dict[(i, j)] = square_distances[i, j]

        return distance_dict
    else:
        i_idx=[]
        j_idx=[]
        for i in range(n):
            for j in range(i+1, n):  # Start from i+1 to avoid duplicates and self-distances
                i_idx.append(i)
                j_idx.append(j)
        return square_distances[i_idx,j_idx]
