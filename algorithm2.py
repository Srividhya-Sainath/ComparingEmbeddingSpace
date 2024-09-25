import os
import h5py
import json
import torch
import time
import logging
from tqdm import tqdm
from itertools import combinations
from utils import compute_distance
from latent_pos.latent_strategy import TruncatedSVDEmbedding

def setup_logger(log_file_path):
    logger = logging.getLogger('KernelComparisonLogger')
    logger.setLevel(logging.INFO)
    
    f_handler = logging.FileHandler(log_file_path)
    f_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    f_handler.setFormatter(formatter)

    logger.addHandler(f_handler)
    
    return logger

def extract_k_value(kernel_key):
    """Extracts the numeric K value from the kernel key, assuming format 'kernelK'."""
    return int(kernel_key.replace('kernel', ''))

def compare_kernels_across_models(model_paths, kernel_keys, embedding_strategy, results_path, device, logger, distance_threshold=1e-10):
    start_time = time.time()
    has_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() and has_gpu else "cpu")
    
    logger.info(f"GPU is available: {has_gpu}")
    logger.info(f"Device: {device}")

    kernels = {}
    for model, model_path in model_paths.items():
        kernels[model] = {}
        with h5py.File(model_path, 'r') as hf:
            for kernel_key in kernel_keys:
                if kernel_key in hf:
                    kernels[model][kernel_key] = torch.tensor(hf[kernel_key][...])

    comparison_results = {}
    previous_distances = None 
    for kernel_key in tqdm(kernel_keys, desc="Processing kernels", unit="kernel"):
        logger.info(f"Comparing for {kernel_key}")
        comparison_results[kernel_key] = []
        current_distances = []  
        for (model1, kernel1), (model2, kernel2) in combinations(kernels.items(), 2):
            distance = compute_distance(kernel1[kernel_key].to(device), kernel2[kernel_key].to(device), embedding_strategy, device=device)
            current_distances.append(distance)
            comparison_results[kernel_key].append((model1, model2, distance))
            logger.info(f"Distance between {model1} and {model2} for {kernel_key} is {distance}")

        if previous_distances is not None:
            differences = [abs(current - previous) for current, previous in zip(current_distances, previous_distances)]
            max_difference = max(differences)
            logger.info(f"Max difference between current and previous distances for {kernel_key}: {max_difference}")

            if max_difference < distance_threshold:
                logger.info(f"Distances for {kernel_key} are similar to the previous K. Stopping further analysis.")
                break
        
        previous_distances = current_distances

    with open(results_path, 'w') as f:
        json.dump(comparison_results, f, indent=4)

    elapsed_time = time.time() - start_time
    logger.info(f"Total comparison time: {elapsed_time:.2f} seconds")

def process_files(base_directory, models, logger):
    """Process all .h5 files from the first model and check if they exist in other model folders."""
    first_model = models[0]
    first_model_folder = os.path.join(base_directory, first_model)
    
    h5_files = [f for f in os.listdir(first_model_folder) if f.endswith('.h5')]
    
    if not h5_files:
        logger.error(f"No .h5 files found in the first model folder: {first_model_folder}")
        return

    for h5_file in h5_files:
        results_path = f'{base_directory}/{h5_file}_kernel_comparisons.json'
        if os.path.exists(results_path):
            logger.info(f"Output file {results_path} already exists. Skipping {h5_file}.")
            continue

        logger.info(f"Checking file: {h5_file}")

        model_paths = {}
        for model in models:
            model_path = os.path.join(base_directory, model, h5_file)
            if os.path.exists(model_path):
                model_paths[model] = model_path
            else:
                logger.warning(f"File {h5_file} does not exist in {model}'s folder. Skipping this file.")
                break

        if len(model_paths) == len(models):
            logger.info(f"File {h5_file} exists in all models. Performing kernel comparisons...")

            example_model = next(iter(model_paths.values()))
            with h5py.File(example_model, 'r') as hf:
                kernel_keys = sorted([key for key in hf.keys() if key.startswith('kernel')], key=extract_k_value)

            ase_strategy = TruncatedSVDEmbedding(n_components=1000)

            compare_kernels_across_models(model_paths, kernel_keys, ase_strategy, results_path, device='cuda', logger=logger)

if __name__ == '__main__':
    base_directory = '' # Directory containing all the models with data kernel files
    
    models = [model for model in os.listdir(base_directory) if model != 'logs']
    log_file_path = f'{base_directory}/kernel_comparison_log.txt'
    logger = setup_logger(log_file_path)
    logger.info("Starting kernel comparisons...")
    logger.info("Available Models: {models} ")
    process_files(base_directory, models, logger)

    logger.info("Kernel comparisons completed.")
