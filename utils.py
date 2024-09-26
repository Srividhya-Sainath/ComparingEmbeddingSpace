" Taken from https://github.com/lilian-zh/data_kernel/blob/torch_version/src/utils.py"

#from scipy.spatial.distance import euclidean
#from scipy.sparse import hstack, vstack, csr_matrix
import numpy as np
import os
import h5py
import torch
import time
import psutil

from latent_pos.decomposition import DecomProcessor
from latent_pos.latent_strategy import SVDEmbedding, ASEmbedding, TruncatedSVDEmbedding


# def get_omnibus_matrix_csr(mat_list):

#     """
#     Adapted from https://github.com/graspologic-org/graspologic/blob/main/graspologic/embed/omni.py

#     Obtain the Omnibus matrix of an iterable of n matrices:
#     [           a, .5 * (a + b), ..., .5 * (a + n)],
#          [.5 * (b + a),            b, ..., .5 * (b + n)],
#          [         ...,          ..., ...,          ...],
#          [.5 * (n + a),  .5 * (n + b), ...,            n]
    
#     Inputs:
#     mat_list: list of csr_matrix with each adjacency matrix
#     return:
#     Omnibus matrix in csr_matrix type
#     """
#     rows = []
#     n = len(mat_list)
#     for column_index in range(n):
#         current_row = []
#         for row_index in range(n):
#             if row_index == column_index:
#                 # Diagonal, simply use the matrix as is
#                 current_row.append(mat_list[column_index])
#             else:
#                 # Off-diagonal, average the two matrices
#                 matrices_averaged = 0.5 * (mat_list[column_index] + mat_list[row_index])
#                 current_row.append(matrices_averaged)
#         # Stack the current row horizontally
#         rows.append(hstack(current_row))
#     # Stack all the rows vertically to form the Omnibus matrix
#     omnibus_matrix = vstack(rows)
#     return omnibus_matrix


# def get_omnibus_matrix_csr_torch(mat_list):
#     """
#     Adapted from https://github.com/graspologic-org/graspologic/blob/main/graspologic/embed/omni.py

#     Obtain the Omnibus matrix of an iterable of n matrices:
#     [           a, .5 * (a + b), ..., .5 * (a + n)],
#          [.5 * (b + a),            b, ..., .5 * (b + n)],
#          [         ...,          ..., ...,          ...],
#          [.5 * (n + a),  .5 * (n + b), ...,            n]
    
#     Inputs:
#     mat_list: list of sparse matrices in torch format
#     return:
#     Omnibus matrix in sparse tensor type
#     """
#     n = len(mat_list)
#     # Creating a list to store each row of tensors
#     rows = []
    
#     for column_index in range(n):
#         row_tensors = []
#         for row_index in range(n):
#             if row_index == column_index:
#                 # Diagonal, use the matrix directly
#                 row_tensors.append(mat_list[column_index])
#             else:
#                 # Off-diagonal, average the two matrices
#                 # Since addition directly on sparse tensors isn't supported in all cases, we might need to coalesce first
#                 matrices_averaged = 0.5 * (mat_list[column_index].coalesce() + mat_list[row_index].coalesce())
#                 row_tensors.append(matrices_averaged.to_sparse())
        
#         # Horizontal stack for the current row
#         row_stacked = torch.sparse.hstack(row_tensors)
#         rows.append(row_stacked)
    
#     # Vertical stack for all rows
#     omnibus_matrix = torch.sparse.vstack(rows)
    
#     return omnibus_matrix

def get_omnibus_matrix_dense(mat_list):
    """
    Adapted from https://github.com/graspologic-org/graspologic/blob/main/graspologic/embed/omni.py

    Obtain the Omnibus matrix of an iterable of n matrices:
    [           a, .5 * (a + b), ..., .5 * (a + n)],
         [.5 * (b + a),            b, ..., .5 * (b + n)],
         [         ...,          ..., ...,          ...],
         [.5 * (n + a),  .5 * (n + b), ...,            n]
    
    Inputs:
    mat_list: list of dense matrices in torch.Tensor format
    return:
    Omnibus matrix in dense tensor type
    """
    n = len(mat_list)
    # Creating a list to store each row of tensors
    rows = []
    
    for column_index in range(n):
        row_tensors = []
        for row_index in range(n):
            if row_index == column_index:
                # Diagonal, use the matrix directly
                row_tensors.append(mat_list[column_index])
            else:
                # Off-diagonal, average the two matrices
                matrices_averaged = 0.5 * (mat_list[column_index] + mat_list[row_index])
                row_tensors.append(matrices_averaged)
        
        # Horizontal stack for the current row
        row_stacked = torch.cat(row_tensors, dim=1)  # Concatenate along the column
        rows.append(row_stacked)
    
    # Vertical stack for all rows
    omnibus_matrix = torch.cat(rows, dim=0)  # Concatenate along the row
    
    return omnibus_matrix


# def load_kernels(directory_paths):
#     kernels = {}
#     for directory_path in directory_paths:
#         kernel_name = os.path.basename(directory_path)
#         for filename in os.listdir(directory_path):
#             if filename.endswith("_dataKernel.h5"):
#                 file_path = os.path.join(directory_path, filename)
#                 with h5py.File(file_path, 'r') as file:
#                     data_kernel = csr_matrix(file['kernel'][:])
#                     kernels[kernel_name] = data_kernel
#     return kernels

def load_data_to_csr(kernel_paths):
    kernels = {}
    for kernel_path in kernel_paths:
        kernel_name = os.path.basename(kernel_path)
        for filename in os.listdir(kernel_path):
            if filename.endswith("_dataKernel.h5"):
                file_path = os.path.join(kernel_path, filename)
                with h5py.File(file_path, 'r') as file:
                    # load kernel as dense_tensor
                    data = torch.tensor(file['kernel'][:], dtype=torch.float32)
                    # transfer dense_tensor to sparse CSR
                    sparse_csr_data = data.to_sparse_csr()
                    # save sparse CSR to dict
                    kernels[kernel_name] = sparse_csr_data
    return kernels

def load_data_kernel(kernel_paths, end_string='_dataKernel.h5'):
    kernels = {}
    for kernel_path in kernel_paths:
        #kernel_name = os.path.basename(kernel_path) # Mere folder structure peh yeho chalega
        kernel_name = os.path.basename(os.path.dirname(kernel_path))
        for filename in os.listdir(kernel_path):
            if filename.endswith(end_string):
                file_path = os.path.join(kernel_path, filename)
                with h5py.File(file_path, 'r') as file:
                    # load kernel as dense_tensor
                    data = torch.tensor(file['kernel'][:], dtype=torch.float32)
                    # save sparse CSR to dict
                    kernels[kernel_name] = data
    return kernels


def load_Tnull(directory_paths):
    Tnulls = {}
    for directory_path in directory_paths:
        kernel_name = os.path.basename(directory_path)
        for filename in os.listdir(directory_path):
            if filename.endswith(".h5"):
                file_path = os.path.join(directory_path, filename)
                with h5py.File(file_path, 'r') as file:
                    Tnull = np.array(file['Tnull'])
                    #print(Ti.shape, Ti.dtype)
                    Tnulls[kernel_name] = Tnull
    return Tnulls

def load_matrix_from_h5(filename, key):
    with h5py.File(filename, 'r') as f:
        matrix = f[key][:]
    return matrix


# def compute_distance(kernel1, kernel2, ase_strategy, apply_norm_across_d=False):
#     """
#     Compare two data kernels using spectral decomposition and Euclidean distance.
    
#     Args:
#         kernel1 (np.array): First data kernel.
#         kernel2 (np.array): Second data kernel.
#         ase_strategy (LatentPositionStrategy): Strategy for computing latent positions.
#         apply_norm_across_d (bool): Whether to apply the norm across the d axis.

#     Returns:
#         float or np.array: The computed distance, either as a scalar or a vector.
#     """
#     latent_pos = DecomProcessor(ase_strategy)

#     # Construct the Omnibus matrix
#     #mat_list = [csr_matrix(kernel1), csr_matrix(kernel2)]
#     mat_list = [kernel1, kernel2]
#     omnibus_matrix = get_omnibus_matrix(mat_list)

#     # Get latent positions
#     latent_positions = latent_pos.process_graph(omnibus_matrix)

#     # Split the latent positions
#     split_index = kernel1.shape[0]
#     upper_half = latent_positions[:split_index]
#     lower_half = latent_positions[split_index:]

#     # Compute the distance based on different levels
#     if apply_norm_across_d:
#         # Compute the Euclidean distance across the embedding dimension 'd'
#         #distance = np.linalg.norm(upper_half - lower_half, axis=1)
#         distance = torch.linalg.norm(upper_half - lower_half, dim=1)
#         return distance
#     else:
#         # Compute the Euclidean distance between the averaged embeddings
#         # distance = euclidean(upper_half.mean(axis=0), lower_half.mean(axis=0))
        
#         # Compute the spectral norm distance
#         #distance = np.linalg.norm(upper_half - lower_half, ord=2)
#         distance = torch.linalg.norm(upper_half - lower_half, ord=2)
#         return distance

def compute_distance(kernel1, kernel2, ase_strategy, apply_norm_across_d=False, device=None):
    """
    Compare two data kernels using spectral decomposition and Euclidean distance.

    Args:
        kernel1 (torch.Tensor): First data kernel.
        kernel2 (torch.Tensor): Second data kernel.
        ase_strategy (LatentPositionStrategy): Strategy for computing latent positions.
        apply_norm_across_d (bool): Whether to apply the norm across the d axis.
        device (str or torch.device): Device on which to perform computations ('cpu', 'cuda', etc.).
                                     Defaults to GPU if available, otherwise CPU.

    Returns:
        float or torch.Tensor: The computed distance, either as a scalar or a vector.
    """
    start_time = time.time()

    # Determine the device automatically if not provided
    device = device if device is not None else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #print('device:', device)

    # Ensure the tensors are on the correct device
    kernel1 = kernel1.to(device)
    kernel2 = kernel2.to(device)

    latent_pos = DecomProcessor(ase_strategy)  # Assume DecomProcessor handles device

    # Construct the Omnibus matrix
    mat_list = [kernel1, kernel2]
    omnibus_matrix = get_omnibus_matrix_dense(mat_list)  # Ensure this function is also updated for device handling

    after_o = time.time()
    tick_time = after_o - start_time
    # Get latent positions
    latent_positions = latent_pos.process_graph(omnibus_matrix)

    after_latent = time.time()
    tock_time = after_latent - after_o

    # Split the latent positions
    split_index = kernel1.shape[0]
    upper_half = latent_positions[:split_index]
    lower_half = latent_positions[split_index:]

    # Compute the distance based on different levels
    if apply_norm_across_d:
        distance = torch.linalg.norm(upper_half - lower_half, dim=1)
        distance = distance.cpu().numpy().tolist()
        #return distance.cpu().numpy()
    else:
        distance = torch.linalg.norm(upper_half - lower_half, ord=2)
        distance = distance.item()
        #return distance.cpu().numpy()

    elapsed_time = time.time() - start_time
    memory_use = psutil.Process().memory_info().rss / (1024 * 1024)  # Convert bytes to MB
    print(f"Operation took {tick_time:.2f} seconds after omnibus, {tock_time:.2f} seconds after latent, and total {elapsed_time:.2f} seconds, and used {memory_use:.2f} MB RAM")

    return distance



# Example usage
if __name__ == '__main__':
    # Define three small example adjacency matrices
    matrix1 = torch.tensor([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ], dtype=torch.float32)

    matrix2 = torch.tensor([
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 0]
    ], dtype=torch.float32)

    matrix3 = torch.tensor([
        [0, 0, 1],
        [0, 0, 1],
        [1, 1, 0]
    ], dtype=torch.float32)

    # List of matrices
    mat_list = [matrix1, matrix2]

    # Get Omnibus matrix
    omnibus_matrix = get_omnibus_matrix_dense(mat_list)

    # Print the result
    print("Omnibus Matrix:\n", omnibus_matrix)

    ase_strategy = TruncatedSVDEmbedding(n_components=1000)
    distance = compute_distance(matrix1, matrix2, ase_strategy, apply_norm_across_d=False, device='cuda')
    print(distance)
