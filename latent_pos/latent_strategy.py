"Taken from https://github.com/lilian-zh/data_kernel/blob/torch_version/src/latent_pos/latent_strategy.py"

import numpy as np
import scipy.sparse as sp
from abc import ABC, abstractmethod
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from scipy.sparse import issparse
import torch
import scipy

from .embed.svd import select_svd
#from embed.svd import select_svd
#from .embed.svd_gpu import select_svd


class LatentPositionStrategy(ABC):
    @abstractmethod
    def compute_latent_positions(self, X):
        pass


class SVDEmbedding(LatentPositionStrategy):
    def __init__(self, n_components=2, algorithm='randomized', n_iter=5, svd_seed=None):
        self.n_components = n_components
        self.algorithm = algorithm
        self.n_iter = n_iter
        self.svd_seed = svd_seed

    def compute_latent_positions(self, X):
        U, D, _ = select_svd(X, n_components=self.n_components, algorithm=self.algorithm,
                             n_iter=self.n_iter, svd_seed=self.svd_seed)
        latent_positions = U * np.sqrt(D)
        return latent_positions



class TruncatedSVDEmbedding(LatentPositionStrategy):
    def __init__(self, n_components=2):
        self.n_components = n_components
        #self.device = device if device is not None else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def compute_latent_positions(self, X):
        # Check if X is a tensor and move it to the appropriate device
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        
        # Perform SVD
        U, S, V = torch.svd(X)

        # Truncate matrices
        U = U[:, :self.n_components]
        S = S[:self.n_components]
        V = V[:, :self.n_components]

        # Sort singular values and corresponding vectors in descending order
        sorted_indices = torch.argsort(S, descending=True)
        S = S[sorted_indices]
        U = U[:, sorted_indices]

        # Compute latent positions
        #latent_positions = U * torch.sqrt(S.unsqueeze(1))
        latent_positions = U * torch.sqrt(S)
        return latent_positions
 

class ASEmbedding(LatentPositionStrategy):
    def __init__(self, n_components=2, device=None):
        self.n_components = n_components
        self.device = device if device is not None else torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    def compute_latent_positions(self, X):
        # Check if X is a sparse matrix and convert it to torch.sparse.Tensor
        if isinstance(X, scipy.sparse.csr_matrix):
            # Convert csr_matrix to COO format which is easier to convert to PyTorch sparse tensor
            X = X.tocoo()
            indices = torch.tensor([X.row, X.col], dtype=torch.long, device=self.device)
            values = torch.tensor(X.data, dtype=torch.float32, device=self.device)
            shape = torch.Size(X.shape)
            A = torch.sparse_coo_tensor(indices, values, shape, device=self.device)
        else:
            A = torch.tensor(X, dtype=torch.float32, device=self.device)

        # Compute eigenvalues and eigenvectors
        # NOTE: PyTorch does not support eigen decomposition for sparse tensors directly. This must be dense.
        if A.is_sparse:
            A = A.to_dense()
        eigenvalues, eigenvectors = torch.linalg.eigh(A)

        # Sort eigenvalues and eigenvectors in descending order
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Select top d dimensions
        top_eigenvalues = eigenvalues[:self.n_components]
        top_eigenvectors = eigenvectors[:, :self.n_components]

        # Compute latent positions using ASE method
        Z = top_eigenvectors @ torch.diag(torch.sqrt(top_eigenvalues))
        return Z.cpu().numpy()  # Return as NumPy array if needed outside torch
