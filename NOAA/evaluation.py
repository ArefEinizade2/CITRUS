import torch
import math
from torch import nn
from torch.nn.functional import mse_loss

from pred_utils import convert_data_to_graph_time_pr_graph
from train_utils import perform_chunk_predictions


def rNMSE(truth: torch.Tensor, pred: torch.Tensor, doLog=False):
    """
    rNMSE computation for predictions on a single step-ahead.

    :param truth: [n_samples x prediction_dimensionality]
    :param pred: [n_samples x prediction_dimensionality]
    :param doLog: whether to log or not
    """

    assert len(truth.shape) == 2
    assert truth.shape == pred.shape

    enum = torch.sum((truth - pred).pow(2))
    denum = torch.sum(truth.pow(2))
    res = torch.sqrt(enum / denum)

    if abs(denum) < 0.1:
        raise Exception("heyhey")
    if doLog:
        print(enum)
        print("\n")
        print(denum)
        print("\n")
        print(res)
        print("\n")
    return res


class rNMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, yhat, y):
        return rNMSE(truth=y, pred=yhat)


class rNMSELossWithSparsityRegularizer(nn.Module):
    def __init__(self, model: list, alpha: float):
        super().__init__()
        self.model = model
        self.alpha = alpha

    def forward(self, yhat, y):
        initial_rNMSE_loss = rNMSE(truth=y, pred=yhat)

        regularization_loss = torch.zeros(1).to(yhat.device)
        parametric_weights = [weight for (name, weight) in self.model.named_parameters() if 's_' in name]
        for tens in parametric_weights:
            regularization_loss += torch.abs(tens)

        final_loss = initial_rNMSE_loss + self.alpha * regularization_loss

        return final_loss


class MSELossWithSparsityRegularizer(nn.Module):
    def __init__(self, model: list, alpha: float):
        super().__init__()
        self.model = model
        self.alpha = alpha

    def forward(self, yhat, y):
        initial_rNMSE_loss = mse_loss(yhat, y)

        regularization_loss = torch.zeros(1).to(yhat.device)
        parametric_weights = [weight for (name, weight) in self.model.named_parameters() if 's_' in name]
        for tens in parametric_weights:
            regularization_loss += torch.abs(tens)

        final_loss = initial_rNMSE_loss + self.alpha * regularization_loss

        return final_loss


def compute_iteration_rNMSE(model, steps, data, labels, device, verbose=False):
    """
    :param model:
    :param steps:
    :param data: [batch x features x nodes x timesteps]
    :param labels: [batch x timesteps x nodes]
    :param device:
    :param verbose:
    :return:
    """


    data = data.to(device)
    labels = labels.to(device)


    rNMSE_dict = {}
    data_for_prediction = data.clone()
    # data_for_prediction_prime = convert_data_to_graph_time_pr_graph(data_for_prediction).squeeze(1)
    data_for_prediction_prime = data_for_prediction
    predictions_dict = {}
    for step in steps:  # [1, 2, 3, 4, 5]
        if verbose:
            print(f"\nComputing predictions for {step}-step ahead.")
        step_idx = step - 1
        assert 0 <= step_idx < 5

        with torch.no_grad():
            prediction = perform_chunk_predictions(model, data_for_prediction_prime, 20)
            predictions_dict[step] = prediction.clone()
        truth = labels[:, step_idx, :]
        assert prediction.shape == truth.shape
        rNMSE_dict[step] = rNMSE(truth, prediction)


        if len(data.shape) == 3:
            # LSTM case
            data_for_prediction = torch.cat((data_for_prediction, prediction.unsqueeze(1)), dim=1)[:, 1:, :]
        elif len(data.shape) == 4:
            # GTCNN case
            data_for_prediction = torch.cat((data_for_prediction, prediction.unsqueeze(1).unsqueeze(-1)), dim=-1)[:, :, :, 1:]
        else:
            raise Exception("Something is wrong.")


    return rNMSE_dict, predictions_dict


def compute_rNMSEs_per_step(steps, flat_predictions, labels):
    result_dict = {}
    for i in range(len(steps)):
        start = i*32
        end = (i+1)*32
        step_predictions = flat_predictions[:, start:end]
        step_truth = labels[:, i, :]

        assert step_predictions.shape == step_truth.shape

        result_dict[i+1] = rNMSE(step_truth, step_predictions, doLog=False)
    return result_dict