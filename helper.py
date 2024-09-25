import numpy as np
import h5py
import pandas as pd
import logging
import yaml

def setup_logger(log_file_path):
    logger = logging.getLogger('KernelProcessingLogger')
    logger.setLevel(logging.INFO)
    f_handler = logging.FileHandler(log_file_path)
    f_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    f_handler.setFormatter(formatter)
    logger.addHandler(f_handler)
    
    return logger

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def file_contains_key(filepath, key):
    """ Check if the h5 file contains the specified key """
    if os.path.exists(filepath):
        with h5py.File(filepath, 'r') as hf:
            if key in hf.keys():
                return True
    return False

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
