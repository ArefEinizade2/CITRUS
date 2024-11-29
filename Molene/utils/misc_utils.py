import os
import numpy as np


def check_create_folder(filepath):
    if not os.path.exists(filepath):
        os.makedirs(filepath)


def build_probability_matrix(C, intra_prob, inter_prob):
    diag_matrix_in_prob = np.zeros((C, C), float)
    np.fill_diagonal(diag_matrix_in_prob, intra_prob)
    diag_matrix_between_prob = np.zeros((C, C), float)
    np.fill_diagonal(diag_matrix_between_prob, inter_prob)
    return np.full((C, C), inter_prob) - diag_matrix_between_prob + diag_matrix_in_prob


def get_folders_in_dir(dataset_path: str) -> list:
    root, dirs, files = next(os.walk(dataset_path))
    return [os.path.join(dataset_path, filename) for filename in dirs]
