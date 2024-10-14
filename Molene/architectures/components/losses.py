import torch
from torch import nn


def ce_loss_with_L1_norm_for_parametric_graph(y_pred: torch.Tensor, y_truth,
                                              model_named_parameters: list, alpha: float):
    loss_function = nn.CrossEntropyLoss()
    ce_loss = loss_function(y_pred, y_truth)

    regularization_loss = torch.zeros(1).to(y_pred.device)
    parametric_weights = [weight for (name, weight) in model_named_parameters if 's_' in name]
    for tens in parametric_weights:
        regularization_loss += torch.abs(tens)

    final_loss = ce_loss + alpha * regularization_loss
    # print(final_loss.item())
    return final_loss
