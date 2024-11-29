import os
import pickle

import torch

from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utils.graph_utils import permutation_by_degree
from utils.misc_utils import check_create_folder


def unpack_molene_dataset(full_dataset, step_ahead):
    dataset_for_step = full_dataset[step_ahead]

    trn_set = dataset_for_step['trn']
    val_set = dataset_for_step['val']
    tst_set = dataset_for_step['tst']

    trn_data = trn_set['data']
    trn_labels = trn_set['labels']
    assert trn_data.shape[0] == trn_labels.shape[0]
    assert trn_data.shape[2] == trn_labels.shape[1]

    val_data = val_set['data']
    val_labels = val_set['labels']
    assert val_data.shape[0] == val_labels.shape[0]
    assert val_data.shape[2] == val_labels.shape[1]

    tst_data = tst_set['data']
    tst_labels = tst_set['labels']
    assert tst_data.shape[0] == tst_labels.shape[0]
    assert tst_data.shape[2] == tst_labels.shape[1]

    return trn_data, trn_labels, val_data, val_labels, tst_data, tst_labels


def get_device(use_gpu=True):
    if use_gpu and torch.cuda.is_available():
        device = 'cuda:0'
        torch.cuda.empty_cache()
    else:
        device = 'cpu'
    print("Device selected: %s" % device)
    return device


def transform_data_to_all_steps_prediction(full_data, node_first, device=None):
    assert 5 in full_data.keys()  # we need the five step-ahead dataset

    trn_data = torch.from_numpy(full_data[5]['trn']['data'])
    val_data = torch.from_numpy(full_data[5]['val']['data'])
    tst_data = torch.from_numpy(full_data[5]['tst']['data'])

    if device:
        trn_data = trn_data.to(device)
        val_data = val_data.to(device)
        tst_data = tst_data.to(device)

    n_trn_samples = trn_data.shape[0]  # we need to take the 5-step ahead since it is the shorter
    trn_indices = list(range(n_trn_samples))

    n_val_samples = val_data.shape[0]  # we need to take the 5-step ahead since it is the shorter
    val_indices = list(range(n_val_samples))

    n_tst_samples = tst_data.shape[0]  # we need to take the 5-step ahead since it is the shorter
    tst_indices = list(range(n_tst_samples))

    trn_labels = torch.stack([
        torch.from_numpy(full_data[1]['trn']['labels'][trn_indices]),
        torch.from_numpy(full_data[2]['trn']['labels'][trn_indices]),
        torch.from_numpy(full_data[3]['trn']['labels'][trn_indices]),
        torch.from_numpy(full_data[4]['trn']['labels'][trn_indices]),
        torch.from_numpy(full_data[5]['trn']['labels'][trn_indices])
    ], dim=1)

    val_labels = torch.stack([
        torch.from_numpy(full_data[1]['val']['labels'][val_indices]),
        torch.from_numpy(full_data[2]['val']['labels'][val_indices]),
        torch.from_numpy(full_data[3]['val']['labels'][val_indices]),
        torch.from_numpy(full_data[4]['val']['labels'][val_indices]),
        torch.from_numpy(full_data[5]['val']['labels'][val_indices])
    ], dim=1)

    tst_labels = torch.stack([
        torch.from_numpy(full_data[1]['tst']['labels'][tst_indices]),
        torch.from_numpy(full_data[2]['tst']['labels'][tst_indices]),
        torch.from_numpy(full_data[3]['tst']['labels'][tst_indices]),
        torch.from_numpy(full_data[4]['tst']['labels'][tst_indices]),
        torch.from_numpy(full_data[5]['tst']['labels'][tst_indices])
    ], dim=1)

    if device:
        trn_labels = trn_labels.to(device)
        val_labels = val_labels.to(device)
        tst_labels = tst_labels.to(device)

    # some checks on the process
    for i in range(n_trn_samples):
        assert (full_data[1]['trn']['data'][i] == full_data[2]['trn']['data'][i]).all()
        assert (full_data[2]['trn']['data'][i] == full_data[3]['trn']['data'][i]).all()
        assert (full_data[3]['trn']['data'][i] == full_data[4]['trn']['data'][i]).all()
        assert (full_data[4]['trn']['data'][i] == full_data[5]['trn']['data'][i]).all()

        assert (full_data[1]['trn']['labels'][i + 1] == full_data[2]['trn']['labels'][i]).all()
        assert (full_data[2]['trn']['labels'][i + 1] == full_data[3]['trn']['labels'][i]).all()
        assert (full_data[3]['trn']['labels'][i + 1] == full_data[4]['trn']['labels'][i]).all()
        assert (full_data[4]['trn']['labels'][i + 1] == full_data[5]['trn']['labels'][i]).all()

    for data, labels in zip([trn_data, val_data, tst_data], [trn_labels, val_labels, tst_labels]):
        assert data.shape[0] == labels.shape[0]
        assert data.shape[2] == labels.shape[2]
        assert labels.shape[1] == 5

    if node_first:
        trn_data = trn_data.transpose(1, 2).unsqueeze(1)  # [batch x features x nodes x timesteps]
        val_data = val_data.transpose(1, 2).unsqueeze(1)  # [batch x features x nodes x timesteps]
        tst_data = tst_data.transpose(1, 2).unsqueeze(1)  # [batch x features x nodes x timesteps]

    return trn_data, val_data, tst_data, trn_labels, val_labels, tst_labels


def convert_data_to_graph_time_pr_graph(data):
    return data.transpose(1, -1).reshape(data.shape[0], -1).unsqueeze(1)


def get_name_string(obs_window,
                    n_feat_per_layer, n_taps_per_layer,
                    time_pooling_ratio_per_layer, pool_reach_per_layer, n_active_nodes_per_timestep_per_layer,
                    weight_decay,
                    cyclic,
                    lambda_reg, time_directed):
    window = f"w={obs_window}"
    layers = f"L={len(n_taps_per_layer)}"
    features = f"F={n_feat_per_layer[1:]}"
    taps = f"T={n_taps_per_layer}"
    ratios = f"P={time_pooling_ratio_per_layer}"
    reach = f"R={pool_reach_per_layer}"
    nodes = f"N={n_active_nodes_per_timestep_per_layer[1:]}"
    weight = f"wd={weight_decay}"
    cyc = f"cyclic={cyclic}"
    lmbda = f"r={lambda_reg}"
    time = f"tdirect={time_directed}"

    string = ""
    for chunk in [window, layers, features, taps, ratios, reach, nodes, weight, cyc, lmbda, time]:
        string += f"_{chunk}"
    return string[1:]


def weight_MOLENE_dataset(coordinates, adjacency_matrix, n_decimals):
    # First, we need to compute the average Euclidean distance between all the stations
    distance_matrix = euclidean_distances(coordinates, coordinates)
    assert distance_matrix.shape == adjacency_matrix.shape

    for j in range(distance_matrix.shape[0]):
        for i in range(distance_matrix.shape[1]):
            if i == j:
                assert distance_matrix[i][j] == 0
            else:
                assert distance_matrix[i][j] != 0

    total_sum = distance_matrix.sum()  # scalar sum of all the distances in the matrix

    # to account for the diagonal elements
    n_pairwise_comparisons = (distance_matrix.shape[0] * distance_matrix.shape[1]) - distance_matrix.shape[0]

    avg_distance = total_sum / n_pairwise_comparisons
    #avg_distance_of_connected_nodes = np.mean(distance_matrix[adjacency_matrix > 0])

    print(f"Average distance: {avg_distance}")

    # Then, we build the weights of the graph by using a Gaussian kernel
    weights_kernel = adjacency_matrix.copy()
    # binarize to keep information about which stations are connected
    weights_kernel[weights_kernel > 0] = 1

    for i in range(weights_kernel.shape[0]):
        for j in range(weights_kernel.shape[1]):
            if weights_kernel[i, j] == 0:
                # there is no edge, no need to compute the weight between the stations
                continue
            else:
                distance_ij = distance_matrix[i, j]
                new_weight = round(np.exp(-distance_ij / avg_distance), n_decimals)
                #new_weight2 = round(np.exp(-distance_ij / avg_distance_of_connected_nodes), n_decimals)
                weights_kernel[i, j] = new_weight

    return weights_kernel


def convert_MOLENE_dataset_to_graph_signals(data_df, ordered_stations):
    graph_signals = []

    date_indices = data_df.index.unique()
    for date_index in date_indices:
        graph_signal = data_df.loc[date_index]
        #     print(graph_signal)
        assert graph_signal.shape[0] == len(ordered_stations)

        sorted_graph_signal = graph_signal.sort_values(by=['numer_sta'])
        #     print(sorted_graph_signal)
        assert list(sorted_graph_signal['numer_sta']) == ordered_stations

        graph_signals.append(sorted_graph_signal['t'].values)

    return np.stack(graph_signals, axis=0)


def generate_data_points_from_timeseries(timeseries_data, window, step):
    samples = []
    targets = []
    n_of_timesteps = timeseries_data.shape[0]
    for i in range(n_of_timesteps):
        start = i
        end = i + window
        data_indices = list(range(start, end))
        target = data_indices[-1] + step

        try:
            sample_data = timeseries_data[data_indices]
            sample_target = timeseries_data[target]

            samples.append(sample_data)
            targets.append(sample_target)
        except:
            #print(f"Finished at timestep: {i}")
            break
    datapoints = np.stack(samples, axis=0)
    labels = np.stack(targets, axis=0)
    assert datapoints.shape[0] == labels.shape[0]

    return datapoints, labels


def generate_dataset_from_graph_timeseries(steps_ahead, observation_window, graph_signals, splits):
    n_of_stations = graph_signals.shape[1]
    n_of_timesteps = graph_signals.shape[0]
    idxs = np.array(list(range(n_of_timesteps)))

    assert len(splits) == 3
    assert sum(splits) == 1
    train_perc = splits[0]
    val_perc = splits[1]
    trn_indices, val_indices, tst_indices = np.split(
        idxs,
        [int(len(idxs) * train_perc),
         int(len(idxs) * (train_perc + val_perc))]
    )

    print(f"{n_of_timesteps} total samples. Train: {len(trn_indices)} - Val: {len(val_indices)} - Test: {len(tst_indices)}")
    assert sum([len(trn_indices), len(val_indices), len(tst_indices)]) == n_of_timesteps

    dataset = {step: {} for step in steps_ahead}

    for step in steps_ahead:
        print(f"\nCreating dataset for {step} step-ahead...")
        for indices, dataset_type in zip([trn_indices, val_indices, tst_indices], ['trn', 'val', 'tst']):
            graph_signals_for_dataset = graph_signals[indices, :]
            datapoints, labels = generate_data_points_from_timeseries(graph_signals_for_dataset, observation_window, step)
            print(f"[{dataset_type}] | N. of obtained samples (and labels): {datapoints.shape[0]}")
            type_step_dataset = {
                'data': datapoints,
                'labels': labels
            }
            dataset[step][dataset_type] = type_step_dataset

    # perform asserts to check the process
    # assertions on step datasets
    for step in steps_ahead:
        assert dataset[step]['trn']['data'].shape[0] == len(trn_indices) - observation_window - step + 1
        assert dataset[step]['trn']['data'].shape[1] == observation_window
        assert dataset[step]['trn']['data'].shape[2] == n_of_stations

        assert dataset[step]['val']['data'].shape[0] == len(val_indices) - observation_window - step + 1
        assert dataset[step]['val']['data'].shape[1] == observation_window
        assert dataset[step]['val']['data'].shape[2] == n_of_stations

        assert dataset[step]['tst']['data'].shape[0] == len(tst_indices) - observation_window - step + 1
        assert dataset[step]['tst']['data'].shape[1] == observation_window
        assert dataset[step]['tst']['data'].shape[2] == n_of_stations

        assert dataset[step]['trn']['labels'].shape[0] == len(trn_indices) - observation_window - step + 1
        assert dataset[step]['trn']['labels'].shape[1] == n_of_stations

        assert dataset[step]['val']['labels'].shape[0] == len(val_indices) - observation_window - step + 1
        assert dataset[step]['val']['labels'].shape[1] == n_of_stations

        assert dataset[step]['tst']['labels'].shape[0] == len(tst_indices) - observation_window - step + 1
        assert dataset[step]['tst']['labels'].shape[1] == n_of_stations

        # to assess the process
        if observation_window > 1:
            assert (
                    dataset[step]['trn']['data'][0][observation_window - 1] ==
                    dataset[step]['trn']['data'][1][observation_window - 2]).all()


    # asserts across step datasets
    for idx in range(len(steps_ahead) - 1):
        assert (
                dataset[steps_ahead[idx]]['trn']['data'][0][0] ==
                dataset[steps_ahead[idx + 1]]['trn']['data'][0][0]).all()
    dataset['all'] = graph_signals



    return dataset, (trn_indices, val_indices, tst_indices)


def obtain_gt_coords(coordinates, n_timesteps, n_spatial_nodes, x_spacing, y_spacing, doSquare, order=None):
    assert coordinates.shape[0] == n_spatial_nodes


    delta_x = max(coordinates[:, 0]) - min(coordinates[:, 0])
    delta_y = max(coordinates[:, 1]) - min(coordinates[:, 1])

    if doSquare:
        assert n_timesteps == 4
        up_left = coordinates.copy()
        up_left[:, 1] = up_left[:, 1] + delta_y + y_spacing

        bottom_right = coordinates.copy()
        bottom_right[:, 0] = bottom_right[:, 0] + delta_x + x_spacing

        up_right = coordinates.copy()
        up_right[:, 1] = up_right[:, 1] + delta_y + y_spacing
        up_right[:, 0] = up_right[:, 0] + delta_x + x_spacing

        gt_coords = np.concatenate((coordinates, up_left, up_right, bottom_right), axis=0)
        for i in range(coordinates.shape[0]):
            gt_coords[i * 4] = coordinates[i]
            gt_coords[i * 4 + 1] = up_left[i]
            gt_coords[i * 4 + 2] = up_right[i]
            gt_coords[i * 4 + 3] = bottom_right[i]
    else:
        # TODO improve flexibility
        assert n_timesteps == 4 or n_timesteps == 5

        up_first = coordinates.copy()
        up_first[:, 1] = up_first[:, 1] + delta_y + y_spacing

        up_second = coordinates.copy()
        up_second[:, 1] = up_first[:, 1] + delta_y + y_spacing

        up_third = coordinates.copy()
        up_third[:, 1] = up_second[:, 1] + delta_y + y_spacing

        up_forth = coordinates.copy()
        up_forth[:, 1] = up_third[:, 1] + delta_y + y_spacing

        if n_timesteps == 4:
            gt_coords = np.concatenate((coordinates, up_first, up_second, up_third), axis=0)
            for i in range(coordinates.shape[0]):
                gt_coords[i * 4] = coordinates[i].copy()
                gt_coords[i * 4 + 1] = up_first[i]
                gt_coords[i * 4 + 2] = up_second[i]
                gt_coords[i * 4 + 3] = up_third[i]
        else:
            gt_coords = np.concatenate((coordinates, up_first, up_second, up_third, up_forth), axis=0)
            for i in range(coordinates.shape[0]):
                gt_coords[i * 5] = coordinates[i].copy()
                gt_coords[i * 5 + 1] = up_first[i]
                gt_coords[i * 5 + 2] = up_second[i]
                gt_coords[i * 5 + 3] = up_third[i]
                gt_coords[i * 5 + 4] = up_forth[i]

        if order:
            initial_coords = gt_coords.copy()
            chunks = [initial_coords[x:x + n_timesteps] for x in range(0, len(initial_coords), n_timesteps)]
            perm_coords = np.array(chunks)[order].reshape(initial_coords.shape[0], -1)
            return gt_coords, perm_coords

    return gt_coords


def run_graph_signals_timeseries_analysis(data, results_folder, splits, do_timeseries, do_distributions):
    # both for original data and with data without in-sample mean:
    #   1) plot and save all the timeseries for each station showing train, validation and test data.
    #   2) plot and save the distribution for each station showing train, validation and test data.


    # data is shaped as [N_stations x N_observations]
    n_stations = data.shape[0]
    n_timesteps = data.shape[1]

    train_val_line_idx = int(n_timesteps * splits[0])
    val_test_line_idx = int(n_timesteps * (splits[0]+splits[1]))

    # TIMESERIES
    if do_timeseries:
        timeseries_folder = os.path.join(results_folder, "timeseries")
        check_create_folder(timeseries_folder)
        for station_idx in range(n_stations):
            station_timeseries = data[station_idx, :]

            # compute in-sample mean
            in_sample_mean = np.mean(station_timeseries[:val_test_line_idx])
            station_timeseries_no_mean = station_timeseries - in_sample_mean

            for data_to_plot, title_label in zip([station_timeseries, station_timeseries_no_mean], ['Original', 'No in-sample mean']):
                title = f"{station_idx} - {title_label}"
                fig, ax = plt.subplots(figsize=(15, 6))
                ax.plot(range(len(data_to_plot)), data_to_plot, color='b')
                plt.axvline(x=train_val_line_idx, c='r', linestyle=':')
                plt.axvline(x=val_test_line_idx, c='r', linestyle=':')
                plt.xlabel("Time index")
                plt.ylabel("Temperature")
                plt.title(title)
                plt.tight_layout()
                plt.savefig(os.path.join(timeseries_folder, f"station={station_idx}_{title_label.replace(' ', '_')}.png"))
                plt.clf()
                plt.close(fig)


            # zoomed visualization

            start = 4380
            end = 4460
            label = f"Zoom {start}-{end}"
            title = f"{station_idx} - {label}"
            fig, ax = plt.subplots(figsize=(15, 6))
            ax.plot(range(len(data_to_plot))[start:end], data_to_plot[start:end], color='b')
            plt.xlabel("Time index")
            plt.ylabel("Temperature")
            plt.title(title)
            plt.tight_layout()
            plt.savefig(os.path.join(timeseries_folder, f"station={station_idx}_{label.replace(' ', '_')}.png"))
            plt.clf()
            plt.close(fig)


    # DISTRIBUTIONS
    if do_distributions:
        distributions_folder = os.path.join(results_folder, "distributions")
        check_create_folder(distributions_folder)

        for station_idx in range(n_stations):
            station_timeseries = data[station_idx, :]

            # compute in-sample mean
            in_sample_mean = np.mean(station_timeseries[:val_test_line_idx])
            station_timeseries_no_mean = station_timeseries - in_sample_mean

            trn_indices = range(train_val_line_idx)
            val_indices = range(train_val_line_idx, val_test_line_idx)
            tst_indices = range(val_test_line_idx, n_timesteps)

            data_to_plot = station_timeseries_no_mean
            pd.Series(data_to_plot[trn_indices]).plot(kind='kde', label="train")
            pd.Series(data_to_plot[val_indices]).plot(kind='kde', label="val")
            pd.Series(data_to_plot[tst_indices]).plot(kind='kde', label="test")

            plt.legend()
            label = "No in-sample mean"
            title = f"{station_idx} - {label}"
            plt.title(title)
            plt.tight_layout()
            plt.savefig(os.path.join(distributions_folder, f"station={station_idx}_{label.replace(' ', '_')}.png"))
            plt.clf()


def get_dataset(ds_type, ds_folder, normalize_adj_matrix, splits, obs_window):
    assert sum(splits) == 1
    if ds_type == 'MOLENE':
        print("\n\nMOLENE is selected\n\n")
        dataset_path = ds_folder + r"/MOLENE/dataset/processed/"
        weighted_adjacency_path = dataset_path + "weighted_adjacency.npy"
        pickle_dataset_path = dataset_path + f"dataset_w={obs_window}_steps=[1, 2, 3, 4, 5]_splits={splits}.pickle"

    elif ds_type == 'NOAA':
        print("\n\nNOAA is selected\n\n")
        dataset_path = ds_folder + r"/NOAA/dataset/processed/"
        weighted_adjacency_path = dataset_path + "weighted_adj.npy"
        pickle_dataset_path = dataset_path + f"NOA_w={obs_window}_steps=[1, 2, 3, 4, 5]_splits={splits}.pickle"
    else:
        raise Exception("Select valid dataset.")

    print(f"\nDataset path: {pickle_dataset_path}")

    in_sample = sum(splits[:-1])
    in_sample_means_dict_path = dataset_path + f"in_sample_means_{in_sample}.pickle"

    print(f"Sample means: {in_sample_means_dict_path}")
    with open(in_sample_means_dict_path, 'rb') as handle:
        in_sample_means = pickle.load(handle)
    with open(pickle_dataset_path, 'rb') as handle:
        data = pickle.load(handle)
        observation_window = data[1]['trn']['data'].shape[1]
        assert observation_window == obs_window
        steps_ahead = [k for k in data.keys() if type(k) == int]

    weighted_adjacency = np.load(file=weighted_adjacency_path)
    N_spatial_nodes = weighted_adjacency.shape[0]

    if normalize_adj_matrix:
        max_eigenvalue = max(np.absolute(np.linalg.eig(weighted_adjacency)[0]))
        weighted_adjacency = weighted_adjacency / max_eigenvalue

    return data, observation_window, N_spatial_nodes, steps_ahead, weighted_adjacency, in_sample_means





def get_NOAA_dataset(ds_folder, splits, obs_window, differenced):
    assert sum(splits) == 1

    print("\n\nNOAA is selected\n\n")
    dataset_path = ds_folder + "./NOAA_data/dataset/processed/"
    weighted_adjacency_path = dataset_path + "weighted_adj.npy"

    pickle_dataset_path = dataset_path + f"NOA_w={obs_window}_steps=[1, 2, 3, 4, 5]_splits={splits}.pickle"

    if differenced:
        pickle_dataset_path = dataset_path + f"NOA_w={obs_window}_steps=[1, 2, 3, 4, 5]_splits={splits}_differenced.pickle"


    print(f"\nDataset path: {pickle_dataset_path}")

    with open(pickle_dataset_path, 'rb') as handle:
        data = pickle.load(handle)
        observation_window = data[1]['trn']['data'].shape[1]
        assert observation_window == obs_window
        steps_ahead = [k for k in data.keys() if type(k) == int]

    weighted_adjacency = np.load(file=weighted_adjacency_path)

    return data, steps_ahead, weighted_adjacency



def get_MOLENE_dataset(ds_folder, splits, obs_window):
    assert sum(splits) == 1

    print("\n\nMOLENE is selected\n\n")
    dataset_path = ds_folder + r"dataset/processed/"
    weighted_adjacency_path = dataset_path + "weighted_adjacency.npy"

    pickle_dataset_path = dataset_path + f"dataset_w={obs_window}_steps=[1, 2, 3, 4, 5]_splits={splits}.pickle"

    print(f"\nDataset path: {pickle_dataset_path}")

    with open(pickle_dataset_path, 'rb') as handle:
        data = pickle.load(handle)
        observation_window = data[1]['trn']['data'].shape[1]
        assert observation_window == obs_window
        steps_ahead = [k for k in data.keys() if type(k) == int]

    weighted_adjacency = np.load(file=weighted_adjacency_path)

    return data, steps_ahead, weighted_adjacency






