import numpy as np
import h5py
import pandas as pd

def TOPk(similarity_matrix, k):
    n = similarity_matrix.shape[0]
    data_kernel = np.zeros_like(similarity_matrix)

    for i in range(n):
        indices = np.arange(n)
        indices_without_diagonal = indices[indices != i]
        top_k_indices = np.argsort(similarity_matrix[i, indices_without_diagonal])[-k:]
        top_k_indices = indices_without_diagonal[top_k_indices]
        
        # data_kernel[i, top_k_indices] = similarity_matrix[i, top_k_indices]
        data_kernel[i, top_k_indices] = 1 # set the top k value as 1
    return data_kernel

def load_matrix_from_h5(filename, key):
    with h5py.File(filename, 'r') as f:
        matrix = f[key][:]
    return matrix
