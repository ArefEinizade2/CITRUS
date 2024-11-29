import torch
import matplotlib.pyplot as plt
from evaluation import rNMSE
from train_utils import perform_chunk_predictions


def perform_step_ahead_deltas(data, model, steps_ahead, verbose=False):
    data_for_prediction = data.clone()
    delta_predictions_dict = {}
    for step in steps_ahead:  # [1, 2, 3, 4, 5]
        if verbose:
            print(f"Computing delta predictions for {step}-step ahead.")
        step_idx = step - 1
        assert 0 <= step_idx < 5

        with torch.no_grad():
            delta_predictions = perform_chunk_predictions(model, data_for_prediction, 100)  # [4371 x 109]
            delta_predictions_dict[step] = delta_predictions.clone()

        if len(data.shape) == 3:
            # LSTM case
            data_for_prediction = torch.cat((data_for_prediction, delta_predictions.unsqueeze(1)), dim=1)[:, 1:, :]
        elif len(data.shape) == 4:
            # GTCNN case
            data_for_prediction = torch.cat((data_for_prediction, delta_predictions.unsqueeze(1).unsqueeze(-1)), dim=-1)[:, :, :, 1:]

    cumulative_delta_predictions_dict = {}
    for step in delta_predictions_dict.keys():
        if verbose:
            print(f"Building cumulative step predictions for step {step}")
        cumulative_predictions_for_step = torch.zeros(delta_predictions_dict[step].shape)
        if verbose:
            print(f"Summing deltas {list(range(1, step + 1))}")
        for step_to_sum in range(1, step + 1):
            cumulative_predictions_for_step += delta_predictions_dict[step_to_sum].cpu()
        cumulative_delta_predictions_dict[step] = cumulative_predictions_for_step

    return delta_predictions_dict, cumulative_delta_predictions_dict


def invert_difference_graph_signals(orig_data, deltas, interval):
    signals = []
    for i in range(interval, orig_data.shape[0]):
        restored_value = orig_data[i - interval, :] + deltas[i - interval, :]
        signals.append(restored_value)
    stacked_signals = torch.stack(signals)
    return stacked_signals



def compute_iteration_rNMSE_with_deltas(cumulative_deltas_dict, original_one_step_labels, verbose=False):
    gtcnn_values = []
    persistence_values = []

    for step in cumulative_deltas_dict.keys():
        if verbose:
            print(f"Computing rNMSE for step {step}")

        cumulative_predictions_for_step = cumulative_deltas_dict[step]
        persistence_delta_predictions = torch.zeros(cumulative_predictions_for_step.shape)

        restored_predicted_values_gtcnn = invert_difference_graph_signals(original_one_step_labels.cpu(),
                                                                          cumulative_predictions_for_step.cpu(), step)
        restored_predicted_values_persistence = invert_difference_graph_signals(original_one_step_labels.cpu(),
                                                                                persistence_delta_predictions.cpu(),
                                                                                step)
        matched_original_values = original_one_step_labels[step:].cpu()

        if verbose:
            print("\tOriginal labels: \t", matched_original_values[12:18, 0].cpu())
            # print("\tRestored labels: \t", restored_step_labels[:7, 0].cpu())
            print("\tRestored persist: \t", restored_predicted_values_persistence[12:18, 0].cpu())
            print("\tRestored gtcnn: \t", restored_predicted_values_gtcnn[12:18, 0].cpu())
            print("\tPredicted deltas: \t", cumulative_predictions_for_step[12:18, 0].cpu())

        rnmse_gtcnn = round(rNMSE(matched_original_values, restored_predicted_values_gtcnn).item(), 4)
        gtcnn_values.append(rnmse_gtcnn)

        pers_rnmse = round(rNMSE(matched_original_values, restored_predicted_values_persistence).item(), 4)
        persistence_values.append(pers_rnmse)

    return gtcnn_values, persistence_values



def visualize_predictions(cumulative_deltas_dict, original_one_step_values, start, width, node, type_of_data):
    end = start + width
    node_to_visualize = node

    for step in cumulative_deltas_dict.keys():
        cumulative_predictions_for_step = cumulative_deltas_dict[step]
        persistence_delta_predictions = torch.zeros(cumulative_predictions_for_step.shape)

        restored_predicted_values_gtcnn = invert_difference_graph_signals(original_one_step_values.cpu(),
                                                                          cumulative_predictions_for_step.cpu(), step)
        restored_predicted_values_persistence = invert_difference_graph_signals(original_one_step_values.cpu(),
                                                                                persistence_delta_predictions.cpu(),
                                                                                step)
        matched_original_values = original_one_step_values[step:].cpu()

        truth = matched_original_values[start:end, node_to_visualize]
        gtcnn_pred = restored_predicted_values_gtcnn[start:end, node_to_visualize]
        persistence_pred = restored_predicted_values_persistence[start:end, node_to_visualize]

        plt.figure(figsize=(15, 3))

        plt.plot(range(width), truth, 'b', label='truth', alpha=0.8)
        plt.plot(range(width), gtcnn_pred, 'r', label='gtcnn', linestyle='--', linewidth=2, alpha=0.8)
        plt.plot(range(width), persistence_pred, 'r', label='persistence', linestyle='-', linewidth=2, alpha=0.1)

        plt.title(
            f"{type_of_data} data - {step}-step ahead prediction - Node: {node_to_visualize} | start: {start} - end: {end}")
        plt.legend()

        plt.tight_layout()
        plt.show()


def visualize_deltas(delta_predictions_dict, delta_labels, start, width, node, type_of_data):
    end = start+width
    node_to_visualize = node

    for step in delta_predictions_dict.keys():


        truth = delta_labels[start:end, step-1, node_to_visualize].cpu()
        gtcnn_pred = delta_predictions_dict[step][start:end, node_to_visualize].cpu()


        plt.figure(figsize=(15, 3))

        plt.plot(range(width), truth, 'b', label='truth')
        plt.plot(range(width), gtcnn_pred, 'g', label='gtcnn', linestyle='-', linewidth=2, alpha=0.6)

        plt.title(f"{type_of_data} data - {step}-step ahead DELTAS prediction - Node: {node_to_visualize} | start: {start} - end: {end}")
        plt.legend()

        plt.tight_layout()
        plt.show()
