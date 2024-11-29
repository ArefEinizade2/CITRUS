from typing import Tuple
import os
import math
import numpy as np
import torch


def convert_src_loc_signals_to_product_graph(data: torch.Tensor) -> torch.Tensor:
    """
    :param data: shape is [batch_size x observation_window x n_nodes_spatial_graph]
    :return: reshaped_data. shape is is [batch_size x 1 x n_nodes_space_time_graph]
    """
    assert len(data.shape) == 3
    reshaped_data = data.transpose(1, -1).reshape(data.shape[0], -1).unsqueeze(1)
    return reshaped_data


def permute_src_loc_data(dataset: torch.Tensor, obs_window: int, order: list) -> torch.Tensor:
    """
    :param dataset: original dataset. Shape is: [batch_size x 1 x n_nodes_space_time_graph]
    :param obs_window: number of total observations
    :param order: permutation of the original graph for which 'dataset' was created
    :return: permuted_dataset. permuted dataset according to 'order'. Shape is: [batch_size x 1 x n_nodes_space_time_graph]
    """
    permuted_dataset = dataset\
        .reshape(dataset.shape[0], -1, obs_window) \
        .transpose(1, 2)[:, :, order] \
        .transpose(1, 2) \
        .reshape(dataset.shape[0], 1, -1)
    return permuted_dataset


def create_dataset(sizes: list, observation_windows: int, allow_overlap: bool, gso: torch.Tensor, tMax: int, verbose: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    data = []
    labels = []
    for source_node_idx in range(sum(sizes)):
        label = list(np.cumsum(sizes) > source_node_idx).index(True)
        if verbose:
            print(f"Source node: {source_node_idx}\t Community: {label}.")

        x_0 = torch.zeros(sum(sizes))
        x_0[source_node_idx] = 1

        diffused_signals = diffuse_signal(x_0, gso, tMax)

        data_points = create_training_points(diffused_signals, observation_windows, allow_overlap)

        data += data_points
        labels += [label for _ in data_points]
    return torch.stack(data), torch.tensor(labels)


def diffuse_signal(initial_signal: torch.Tensor, gso: torch.Tensor, n_of_timesteps: int) -> list:
    """
    Takes a signal, a gso, and performs 'n_of_timesteps' shifts using the GSO. All the generated signals are returned
    """
    signals = [initial_signal]
    for _ in range(n_of_timesteps):
        x_diffused = torch.mv(gso, signals[-1])  # shift by one the previously shifted signal
        signals.append(x_diffused)
    signals = signals[1:]
    return signals


def create_training_points(diffused_signals: list, window_length: int, overlap: bool) -> list:
    data_points = []
    for i in range(0, len(diffused_signals), 1 if overlap else window_length):
        packed_signals = torch.stack(diffused_signals[i: i + window_length])
        if packed_signals.shape[0] == window_length:
            data_points.append(packed_signals)
        else:
            # print(f"Datapoint discarded starting from index '{i}'")
            pass
    return data_points


def perform_split(indices: list, data: torch.Tensor, labels: torch.Tensor, split: list, return_indices=False):
    """
    Performs the actual split into train val test
    """
    assert 0.99 < sum(split) < 1.01
    trn_idx, val_idx, tst_idx = np.split(np.random.permutation(indices),
                                         [int(len(indices) * split[0]),
                                          int(len(indices) * sum(split[:-1]))]
                                         )
    trn_data = data[trn_idx]
    trn_labels = labels[trn_idx]
    val_data = data[val_idx]
    val_labels = labels[val_idx]
    tst_data = data[tst_idx]
    tst_labels = labels[tst_idx]

    if return_indices:
        return trn_data, trn_labels, val_data, val_labels, tst_data, tst_labels, trn_idx, val_idx, tst_idx
    else:
        return trn_data, trn_labels, val_data, val_labels, tst_data, tst_labels


def compute_total_number_data_points_src_loc(window: int, n_of_nodes: int, t_max: int, overlap: bool) -> int:
    if overlap:
        assert window < t_max
        total_number_of_datapoints_per_graph = n_of_nodes * (1 + t_max - window)
    else:
        total_number_of_datapoints_per_graph = n_of_nodes * math.floor(t_max / window)
    return total_number_of_datapoints_per_graph


def save_split_data(trn_data: torch.Tensor, trn_labels: torch.Tensor,
                    val_data: torch.Tensor, val_labels: torch.Tensor,
                    tst_data: torch.Tensor, tst_labels: torch.Tensor,
                    path_to_save: str) -> None:
    torch.save(trn_data, os.path.join(path_to_save, 'trn_data.pt'))
    torch.save(trn_labels, os.path.join(path_to_save, 'trn_labels.pt'))
    torch.save(val_data, os.path.join(path_to_save, 'val_data.pt'))
    torch.save(val_labels, os.path.join(path_to_save, 'val_labels.pt'))
    torch.save(tst_data, os.path.join(path_to_save, 'tst_data.pt'))
    torch.save(tst_labels, os.path.join(path_to_save, 'tst_labels.pt'))


def get_dataset(path: str):
    trn_data, trn_labels = load_data_and_labels(path, "trn")
    val_data, val_labels = load_data_and_labels(path, "val")
    tst_data, tst_labels = load_data_and_labels(path, "tst")
    return trn_data, trn_labels, val_data, val_labels, tst_data, tst_labels


def load_data_and_labels(path: str, prefix: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    :param path:
    :param prefix: can be "trn", "val" or "tst"
    """
    data = torch.load(os.path.join(path, f'{prefix}_data.pt'))
    labels = torch.load(os.path.join(path, f'{prefix}_labels.pt'))
    return data, labels
