'''From https://github.com/graspologic-org/graspologic/blob/main/graspologic/embed/svd.py'''

from typing import Optional, Union
import torch
import numpy as np
import scipy
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.utils.extmath import randomized_svd
from scipy.sparse import csr_array
from scipy.stats import norm
from typing_extensions import Literal
from graspologic.types import List, Tuple

SvdAlgorithmType = Literal["full", "truncated", "randomized", "eigsh"]


def _compute_likelihood(arr: np.ndarray) -> np.ndarray:
    """
    Computes the log likelihoods based on normal distribution given
    a 1d-array of sorted values. If the input has no variance,
    the likelihood will be nan.
    """
    n_elements = len(arr)
    likelihoods = np.zeros(n_elements)

    for idx in range(1, n_elements + 1):
        # split into two samples
        s1 = arr[:idx]
        s2 = arr[idx:]

        # deal with when input only has 2 elements
        if (s1.size == 1) & (s2.size == 1):
            likelihoods[idx - 1] = -np.inf
            continue

        # compute means
        mu1 = np.mean(s1)
        if s2.size != 0:
            mu2 = np.mean(s2)
        else:
            # Prevent numpy warning for taking mean of empty array
            mu2 = -np.inf

        # compute pooled variance
        variance = (np.sum((s1 - mu1) ** 2) + np.sum((s2 - mu2) ** 2)) / (
            n_elements - 1 - (idx < n_elements)
        )
        std = np.sqrt(variance)

        # compute log likelihoods
        likelihoods[idx - 1] = np.sum(norm.logpdf(s1, loc=mu1, scale=std)) + np.sum(
            norm.logpdf(s2, loc=mu2, scale=std)
        )

    return likelihoods


def select_dimension(
    X: Union[np.ndarray, sp.csr_array],
    n_components: Optional[int] = None,
    n_elbows: int = 2,
    threshold: Optional[float] = None,
    return_likelihoods: bool = False,
) -> Union[
    Tuple[List[int], List[float]], Tuple[List[int], List[float], List[np.ndarray]]
]:
    """
    Generates profile likelihood from array based on Zhu and Godsie method.
    Elbows correspond to the optimal embedding dimension.

    Parameters
    ----------
    X : 1d or 2d array-like
        Input array generate profile likelihoods for. If 1d-array, it should be
        sorted in decreasing order. If 2d-array, shape should be
        (n_samples, n_features).
    n_components : int, optional, default: None.
        Number of components to embed. If None, ``n_components =
        floor(log2(min(n_samples, n_features)))``. Ignored if ``X`` is 1d-array.
    n_elbows : int, optional, default: 2.
        Number of likelihood elbows to return. Must be ``> 1``.
    threshold : float, int, optional, default: None
        If given, only consider the singular values that are ``> threshold``. Must
        be ``>= 0``.
    return_likelihoods : bool, optional, default: False
        If True, returns the all likelihoods associated with each elbow.

    Returns
    -------
    elbows : list
        Elbows indicate subsequent optimal embedding dimensions. Number of
        elbows may be less than ``n_elbows`` if there are not enough singular
        values.
    sing_vals : list
        The singular values associated with each elbow.
    likelihoods : list of array-like
        Array of likelihoods of the corresponding to each elbow. Only returned
        if ``return_likelihoods`` is True.

    References
    ----------
    .. [1] Zhu, M. and Ghodsi, A. (2006).
        Automatic dimensionality selection from the scree plot via the use of
        profile likelihood. Computational Statistics & Data Analysis, 51(2),
        pp.918-930.
    """
    # Handle input data
    if not isinstance(X, (np.ndarray, csr_array)):
        msg = "X must be a numpy array or scipy.sparse.csr_array, not {}.".format(
            type(X)
        )
        raise ValueError(msg)
    if X.ndim > 2:
        msg = "X must be a 1d or 2d-array, not {}d array.".format(X.ndim)
        raise ValueError(msg)
    elif np.min(X.shape) <= 1:
        msg = "X must have more than 1 samples or 1 features."
        raise ValueError(msg)

    # Handle n_elbows
    if not isinstance(n_elbows, int):
        msg = "n_elbows must be an integer, not {}.".format(type(n_elbows))
        raise ValueError(msg)
    elif n_elbows < 1:
        msg = f"number of elbows should be an integer > 1, not {n_elbows}."
        raise ValueError(msg)

    # Handle threshold
    if threshold is not None:
        if not isinstance(threshold, (int, float)):
            msg = "threshold must be an integer or a float, not {}.".format(
                type(threshold)
            )
            raise ValueError(msg)
        elif threshold < 0:
            msg = "threshold must be >= 0, not {}.".format(threshold)
            raise ValueError(msg)

    # Handle n_components
    if n_components is None:
        # per recommendation by Zhu & Godsie
        k = int(np.ceil(np.log2(np.min(X.shape))))
    elif not isinstance(n_components, int):
        msg = "n_components must be an integer, not {}.".format(type(n_components))
        raise ValueError(msg)
    else:
        k = n_components

    # Check to see if svd is needed
    if X.ndim == 1:
        D = np.sort(X)[::-1]
    elif X.ndim == 2:
        # Singular values in decreasing order
        D = scipy.sparse.linalg.svds(A=X, k=k, return_singular_vectors=False)
        D = np.sort(D)[::-1]
        # U, D, V = sklearn.utils.extmath.randomized_svd()

    if threshold is not None:
        D = D[D > threshold]

        if len(D) == 0:
            msg = "No values greater than threshold {}."
            raise IndexError(msg.format(threshold))

    idx = 0
    elbows = []
    values = []
    likelihoods = []
    for _ in range(n_elbows):
        arr = D[idx:]
        if arr.size <= 1:  # Cant compute likelihoods with 1 numbers
            break
        lq = _compute_likelihood(arr)
        idx += np.argmax(lq).item() + 1
        elbows.append(idx)
        values.append(D[idx - 1])
        likelihoods.append(lq)

    if return_likelihoods:
        return elbows, values, likelihoods
    else:
        return elbows, values


def select_svd(
    X: Union[np.ndarray, sp.csr_matrix],
    n_components: Optional[int] = None,
    n_elbows: Optional[int] = 2,
    algorithm: str = "randomized",
    n_iter: int = 5,
    svd_seed: Optional[int] = None,
    device: str = 'cuda'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform SVD using various methods, leveraging GPU with PyTorch when possible.
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Handle the input data format conversion if necessary
    if isinstance(X, sp.csr_matrix):
        if algorithm not in ['truncated', 'eigsh']:
            X = X.todense()
        X = torch.tensor(X, dtype=torch.float32).to(device)
    else:
        X = torch.tensor(X, dtype=torch.float32).to(device)

    if n_components is None:
        if n_elbows is None:
            raise ValueError(
                "both n_components and n_elbows are None. One must be provided."
            )
        else:
            dims = select_dimension(X, n_elbows=n_elbows, threshold=None)
            elbows = dims[0]
            n_components = elbows[-1]

    U, S, V = None, None, None

    # Using PyTorch for GPU-accelerated computation
    if algorithm in ['randomized', 'full', 'truncated']:
        if X.is_sparse:
            X = X.to_dense()  # PyTorch SVD doesn't support sparse matrix directly
        U, S, V = torch.linalg.svd(X, full_matrices=False)
        U, S, V = U[:, :n_components], S[:n_components], V[:, :n_components]

    # Adjust the method based on the algorithm specified
    if algorithm == 'randomized':
        U, S, Vt = randomized_svd(X.cpu().numpy(), n_components=n_components, n_iter=n_iter, random_state=svd_seed)
        V = Vt.T
    elif algorithm == 'truncated':
        U, S, Vt = svds(X.cpu().numpy(), k=n_components)
        V = Vt.T
    elif algorithm == 'eigsh':
        if not isinstance(X, sp.csr_matrix):
            raise ValueError("Eigsh algorithm requires a sparse matrix input.")
        S, U = sp.linalg.eigsh(X, k=n_components)
        S = np.abs(S)
        V = U.T
    elif algorithm == 'full':
        U, S, V = np.linalg.svd(X.cpu().numpy(), full_matrices=False)
        U, S, V = U[:, :n_components], S[:n_components], V[:, :n_components]

    # Sort and slice the components
    if algorithm not in ['eigsh', 'randomized', 'truncated']:
        sorted_indices = torch.argsort(S, descending=True)
        U = U[:, sorted_indices]
        S = S[sorted_indices]
        V = V[:, sorted_indices]

    # Move data back to CPU for numpy processing
    if isinstance(U, torch.Tensor):
        U, S, V = U.cpu().numpy(), S.cpu().numpy(), V.cpu().numpy()

    return U, S, V

# Example usage
if __name__ == '__main__':
    data = np.random.rand(100, 50)
    U, D, V = select_svd(data, n_components=10, algorithm='randomized')
    print("U shape:", U.shape)
    print("D shape:", D.shape)
    print("V shape:", V.shape)
