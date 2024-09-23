import os
import h5py
import numpy as np
import yaml
from sklearn.metrics.pairwise import cosine_similarity
from helper import load_matrix_from_h5, TOPk
import torch
from tqdm import tqdm
import logging
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def save_data_kernel(feature_paths, output_base_directory, k_list, logger):
    for feature_path in tqdm(feature_paths, desc="Processing models", unit="file"):
        model_type = feature_path.split('/')[-3]
        output_directory = os.path.join(output_base_directory, model_type)
        os.makedirs(output_directory, exist_ok=True)

        filename = os.path.basename(feature_path).replace('.h5', '')
        output_file_path = os.path.join(output_directory, f'{filename}_dataKernel.h5')

        feature = load_matrix_from_h5(feature_path, 'feats')
        similarity_matrix = cosine_similarity(feature)

        for k in k_list:
            kernel_key = f'kernel{k}'
            if file_contains_key(output_file_path, kernel_key):
                logger.info(f"Skipping {output_file_path} as it already contains '{kernel_key}'.")
            else:
                data_kernel = TOPk(similarity_matrix, k=k)
                with h5py.File(output_file_path, 'a') as hf_out:
                    hf_out.create_dataset(kernel_key, data=data_kernel)
                logger.info(f"Saved data kernel {kernel_key} to {output_file_path}")

def process_files(config, logger):
    input_directories = config['input_directories']
    output_directory = config['output_directory']
    k_list = config['K']
    os.makedirs(output_directory, exist_ok=True)
    for input_directory in tqdm(input_directories, desc="Processing directories", unit="directory"):
        feature_paths = [os.path.join(input_directory, f) for f in os.listdir(input_directory) if f.endswith('.h5')]
        logger.info(f"Processing directory: {input_directory}, found {len(feature_paths)} files.")
        save_data_kernel(feature_paths, output_directory, k_list, logger)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process kernel generation pipeline.")
    parser.add_argument('--config', type=str, required=True, help="Path to the config.yaml file.")
    args = parser.parse_args()
    config = load_config(args.config)
    log_file_path = config['log_file_path']
    log_dir = os.path.dirname(log_file_path)
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(log_file_path)
    logger.info("Starting processing...")
    process_files(config, logger)

    logger.info("Processing completed.")
