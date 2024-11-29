import os
import sys

import torch
import numpy as np
from sklearn.metrics import accuracy_score
from tensorboardX import SummaryWriter
if sys.platform == 'win32':
    from win10toast import ToastNotifier
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from architectures.space_time.grnn_ruiz.architecture import GatedGCRNNforClassification, GatedGCRNNforRegression
from utils.plot_utils import print_confusion_matrix


def train_model_regression(Iter, model, training_data, validation_data, single_step_trn_labels, single_step_val_labels,
                           num_epochs, batch_size,
                           loss_criterion, optimizer, scheduler,
                           val_metric_criterion,
                           log_dir, not_learning_limit):

    tensorboard = SummaryWriter(log_dir=log_dir)

    n_trn_samples = training_data.size()[0]
    n_batches_per_epoch = int(n_trn_samples/batch_size)

    best_val_metric = 10e10
    print(f"{n_batches_per_epoch} batches per epoch ({n_trn_samples} trn samples in total | batch_size: {batch_size})")

    not_learning_count = 0
    for epoch in range(num_epochs):
        permutation = torch.randperm(n_trn_samples)  # shuffle the training data

        batch_losses = []
        for batch_idx in range(0, n_trn_samples, batch_size):
            batch_indices = permutation[batch_idx:batch_idx + batch_size]
            batch_trn_data, batch_one_step_trn_labels = training_data[batch_indices, :, :], single_step_trn_labels[batch_indices]

            if isinstance(model, GatedGCRNNforRegression):
                batch_trn_data = batch_trn_data.permute(0, 3, 1, 2)
                h0 = torch.zeros(len(batch_indices), model.F_h, batch_trn_data.shape[3]).to(batch_trn_data.device)
                one_step_pred_trn = model(batch_trn_data, h0)
            else:
                one_step_pred_trn = model(batch_trn_data)

            # obtain the loss function
            batch_trn_loss = loss_criterion(one_step_pred_trn, batch_one_step_trn_labels)
            batch_losses.append(batch_trn_loss.item())

            optimizer.zero_grad()
            batch_trn_loss.backward()
            optimizer.step()

        epoch_trn_loss = np.average(batch_losses)
        tensorboard.add_scalar('train-loss', epoch_trn_loss, epoch)

        val_loss = compute_loss_in_chunks(model, validation_data, single_step_val_labels, loss_criterion)

        if val_metric_criterion:
            val_metric = compute_loss_in_chunks(model, validation_data, single_step_val_labels, val_metric_criterion)
        else:
            val_metric = val_loss
        tensorboard.add_scalar('val-metric', val_metric, epoch)

        if scheduler:
            scheduler.step(val_metric)  # this decides when to decrease the learning rate based on plateaus

        diff_loss = abs(epoch_trn_loss - val_loss)
        tensorboard.add_scalar('val-loss', val_loss, epoch)

        tensorboard.add_scalar('diff-loss', diff_loss, epoch)

        tensorboard.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        # # Then, we also compute the iteration avg rNMSE up to 5 steps ahead
        # rNMSEs_val_dict = compute_iteration_rNMSE(one_step_gtcnn, steps_ahead, val_data, val_labels)
        # avg_val_rNMSE = round(np.average(list(rNMSEs_val_dict.values())), 5)
        # tb.add_scalar('valid-avg_rNMSE', avg_val_rNMSE, epoch)

        # We also log the values of the s_ij parameters at each layer
        names = list(dict(model.named_parameters()).keys())
        s_parameters_names = [name for name in names if str(name).startswith("s_")]
        for name in s_parameters_names:
            tensorboard.add_scalar(
                name.replace(".", "/").replace("GFL/", ""),
                round(dict(model.named_parameters())[name].item(), 3),
                epoch
            )

        print(f"Iter {Iter}"
                f"\n Epoch {epoch}"
              f"\n\t train-loss: {round(epoch_trn_loss, 3)} | valid-loss: {round(val_loss, 3)} \t| valid-metric: {val_metric} | lr: {optimizer.param_groups[0]['lr']}")

        if val_metric < best_val_metric:
            not_learning_count = 0
            print(f"\n\t\t\t\tNew best val_metric: {val_metric}. Saving model...\n")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict()
            }, log_dir + "/best_one_step_gtcnn.pth")

            best_val_metric = val_metric
        else:
            not_learning_count += 1

        if not_learning_count > not_learning_limit:
            print("Training is INTERRUPTED.")
            tensorboard.close()

            checkpoint_best = torch.load(log_dir + "/best_one_step_gtcnn.pth")
            model.load_state_dict(checkpoint_best['model_state_dict'])
            epoch_best = checkpoint_best['epoch']
            model.eval()
            print(f"Best model was at epoch: {epoch_best}")

            return model, epoch_best

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict()
        }, log_dir + "/last_one_step_gtcnn.pth")

    print("Training is finished.")
    tensorboard.close()

    checkpoint_best = torch.load(log_dir + "/best_one_step_gtcnn.pth")
    model.load_state_dict(checkpoint_best['model_state_dict'])
    epoch_best = checkpoint_best['epoch']
    model.eval()
    print(f"Best model was at epoch: {epoch_best}")

    return model, epoch_best
#%%
def train_model_regression(Iter, model, training_data, validation_data, single_step_trn_labels, single_step_val_labels,
                           num_epochs, batch_size,
                           loss_criterion, optimizer, scheduler,
                           val_metric_criterion,
                           log_dir, not_learning_limit):

    tensorboard = SummaryWriter(log_dir=log_dir)

    n_trn_samples = training_data.size()[0]
    n_batches_per_epoch = int(n_trn_samples/batch_size)

    best_val_metric = 10e10
    print(f"{n_batches_per_epoch} batches per epoch ({n_trn_samples} trn samples in total | batch_size: {batch_size})")

    not_learning_count = 0
    for epoch in range(num_epochs):
        permutation = torch.randperm(n_trn_samples)  # shuffle the training data

        batch_losses = []
        for batch_idx in range(0, n_trn_samples, batch_size):
            batch_indices = permutation[batch_idx:batch_idx + batch_size]
            batch_trn_data, batch_one_step_trn_labels = training_data[batch_indices, :, :], single_step_trn_labels[batch_indices]

            if isinstance(model, GatedGCRNNforRegression):
                batch_trn_data = batch_trn_data.permute(0, 3, 1, 2)
                h0 = torch.zeros(len(batch_indices), model.F_h, batch_trn_data.shape[3]).to(batch_trn_data.device)
                one_step_pred_trn = model(batch_trn_data, h0)
            else:
                one_step_pred_trn = model(batch_trn_data)

            # obtain the loss function
            batch_trn_loss = loss_criterion(one_step_pred_trn, batch_one_step_trn_labels)
            batch_losses.append(batch_trn_loss.item())

            optimizer.zero_grad()
            batch_trn_loss.backward()
            optimizer.step()

        epoch_trn_loss = np.average(batch_losses)
        tensorboard.add_scalar('train-loss', epoch_trn_loss, epoch)

        val_loss = compute_loss_in_chunks(model, validation_data, single_step_val_labels, loss_criterion)

        if val_metric_criterion:
            val_metric = compute_loss_in_chunks(model, validation_data, single_step_val_labels, val_metric_criterion)
        else:
            val_metric = val_loss
        tensorboard.add_scalar('val-metric', val_metric, epoch)

        if scheduler:
            scheduler.step(val_metric)  # this decides when to decrease the learning rate based on plateaus

        diff_loss = abs(epoch_trn_loss - val_loss)
        tensorboard.add_scalar('val-loss', val_loss, epoch)

        tensorboard.add_scalar('diff-loss', diff_loss, epoch)

        tensorboard.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        # # Then, we also compute the iteration avg rNMSE up to 5 steps ahead
        # rNMSEs_val_dict = compute_iteration_rNMSE(one_step_gtcnn, steps_ahead, val_data, val_labels)
        # avg_val_rNMSE = round(np.average(list(rNMSEs_val_dict.values())), 5)
        # tb.add_scalar('valid-avg_rNMSE', avg_val_rNMSE, epoch)

        # We also log the values of the s_ij parameters at each layer
        names = list(dict(model.named_parameters()).keys())
        s_parameters_names = [name for name in names if str(name).startswith("s_")]
        for name in s_parameters_names:
            tensorboard.add_scalar(
                name.replace(".", "/").replace("GFL/", ""),
                round(dict(model.named_parameters())[name].item(), 3),
                epoch
            )

        print(f"Iter {Iter}"
                f"\n Epoch {epoch}"
              f"\n\t train-loss: {round(epoch_trn_loss, 3)} | valid-loss: {round(val_loss, 3)} \t| valid-metric: {val_metric} | lr: {optimizer.param_groups[0]['lr']}")

        if val_metric < best_val_metric:
            not_learning_count = 0
            print(f"\n\t\t\t\tNew best val_metric: {val_metric}. Saving model...\n")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict()
            }, log_dir + "/best_one_step_gtcnn.pth")

            best_val_metric = val_metric
        else:
            not_learning_count += 1

        if not_learning_count > not_learning_limit:
            print("Training is INTERRUPTED.")
            tensorboard.close()

            checkpoint_best = torch.load(log_dir + "/best_one_step_gtcnn.pth")
            model.load_state_dict(checkpoint_best['model_state_dict'])
            epoch_best = checkpoint_best['epoch']
            model.eval()
            print(f"Best model was at epoch: {epoch_best}")

            return model, epoch_best

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict()
        }, log_dir + "/last_one_step_gtcnn.pth")

    print("Training is finished.")
    tensorboard.close()

    checkpoint_best = torch.load(log_dir + "/best_one_step_gtcnn.pth")
    model.load_state_dict(checkpoint_best['model_state_dict'])
    epoch_best = checkpoint_best['epoch']
    model.eval()
    print(f"Best model was at epoch: {epoch_best}")

    return model, epoch_best


def train_model_quakes(model, training_data, validation_data, trn_labels, val_labels,
                       num_epochs, batch_size,
                       loss_criterion, optimizer, scheduler,
                       val_metric_criterion,
                       log_dir, not_learning_limit, show_notifications=False, print_cm=True):

    tensorboard = SummaryWriter(log_dir=log_dir)
    toaster = ToastNotifier() if show_notifications else None
    n_trn_samples = training_data.size()[0]
    n_batches_per_epoch = int(n_trn_samples/batch_size)

    best_val_metric = np.inf
    print(f"{n_batches_per_epoch} batches per epoch ({n_trn_samples} trn samples in total | batch_size: {batch_size})")

    not_learning_count = 0
    for epoch in range(num_epochs):
        if toaster:
            if epoch%10 == 0:
                #toaster.show_toast("Epoch number", str(epoch))
                pass
        permutation = torch.randperm(n_trn_samples)  # shuffle the training data

        batch_losses = []
        for batch_idx in range(0, n_trn_samples, batch_size):
            batch_indices = permutation[batch_idx:batch_idx + batch_size]
            batch_trn_data, batch_trn_labels = training_data[batch_indices, :, :], trn_labels[batch_indices]

            if isinstance(model, GatedGCRNNforClassification):
                h0 = torch.zeros(len(batch_indices), model.F_h, batch_trn_data.shape[3]).to(batch_trn_data.device)
                batch_pred = model(batch_trn_data, h0)
            else:
                batch_pred = model(batch_trn_data)

            # obtain the loss function
            batch_trn_loss = loss_criterion(batch_pred, batch_trn_labels.long())
            batch_losses.append(batch_trn_loss.item())

            optimizer.zero_grad()
            batch_trn_loss.backward()
            optimizer.step()

        epoch_trn_loss = np.average(batch_losses)
        tensorboard.add_scalar('train-loss', epoch_trn_loss, epoch)

        val_pred = perform_chunk_predictions(model, validation_data, chunk_size=batch_size)
        val_loss = round(loss_criterion(val_pred, val_labels.long()).item(), 3)


        #val_loss = compute_loss_in_chunks(model, validation_data, val_labels.long(), loss_criterion, chunk_size=batch_size)

        if val_metric_criterion:
            val_metric = compute_loss_in_chunks(model, validation_data, val_labels, val_metric_criterion, chunk_size=batch_size)
        else:
            val_metric = val_loss
        tensorboard.add_scalar('val-metric', val_metric, epoch)

        # this decides when to decrease the learning rate
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_metric)
        elif isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
            scheduler.step()
        else:
            raise ValueError()

        diff_loss = abs(epoch_trn_loss - val_loss)
        tensorboard.add_scalar('val-loss', val_loss, epoch)

        tensorboard.add_scalar('diff-loss', diff_loss, epoch)

        tensorboard.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        # # Then, we also compute the iteration avg rNMSE up to 5 steps ahead
        # rNMSEs_val_dict = compute_iteration_rNMSE(one_step_gtcnn, steps_ahead, val_data, val_labels)
        # avg_val_rNMSE = round(np.average(list(rNMSEs_val_dict.values())), 5)
        # tb.add_scalar('valid-avg_rNMSE', avg_val_rNMSE, epoch)

        # We also log the values of the s_ij parameters at each layer
        names = list(dict(model.named_parameters()).keys())
        s_parameters_names = [name for name in names if str(name).startswith("s_")]
        for name in s_parameters_names:
            tensorboard.add_scalar(
                name.replace(".", "/").replace("GFL/", ""),
                round(dict(model.named_parameters())[name].item(), 3),
                epoch
            )

        print(f"Epoch {epoch}"
              f"\n\t train-loss: {round(epoch_trn_loss, 3)} | valid-loss: {round(val_loss, 3)} \t| valid-metric: {val_metric} | lr: {optimizer.param_groups[0]['lr']}")



        if val_metric < best_val_metric:
            not_learning_count = 0
            print(f"\n\t\t\t\tNew best val_metric: {val_metric}. Saving model...\n")
            cm = compute_confusion_matrix(output=val_pred, target=val_labels.long(), print_cm=print_cm)
            # plot_cm(cm, title=f"Epoch {epoch}, val_metric: {val_metric}")
            np.save(arr=cm, file=os.path.join(log_dir, "best_cm_val.npy"))
            if toaster:
                toaster.show_toast(title="New best val_metric", msg=f"{val_metric}", duration=2)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict()
            }, log_dir + "/best_model.pth")

            best_val_metric = val_metric
        else:
            not_learning_count += 1

        if not_learning_count > not_learning_limit:
            print("Training is INTERRUPTED.")
            tensorboard.close()

            checkpoint_best = torch.load(log_dir + "/best_model.pth")
            model.load_state_dict(checkpoint_best['model_state_dict'])
            epoch_best = checkpoint_best['epoch']
            model.eval()
            print(f"Best model was at epoch: {epoch_best}")

            return model, epoch_best

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict()
        }, log_dir + "/last_model.pth")

    print("Training is finished.")
    tensorboard.close()

    checkpoint_best = torch.load(log_dir + "/best_model.pth")
    model.load_state_dict(checkpoint_best['model_state_dict'])
    epoch_best = checkpoint_best['epoch']
    model.eval()
    print(f"Best model was at epoch: {epoch_best}")

    return model, epoch_best



def compute_loss_in_chunks(model, data, labels, criterion, chunk_size=300):
    predictions = perform_chunk_predictions(model, data, chunk_size)
    val_loss = round(criterion(predictions, labels).item(), 3)
    return val_loss


def perform_chunk_predictions(model, data, chunk_size):
    """
    :param model:
    :param data: [batch x features x nodes x timesteps]
    :param chunk_size:
    :return: predictions: [n_samples x spatial_nodes]
    """
    n_val_samples = data.shape[0]
    val_indices = range(n_val_samples)
    with torch.no_grad():
        predictions = []
        for val_batch_idx in range(0, n_val_samples, chunk_size):
            batch_indices = val_indices[val_batch_idx:val_batch_idx + chunk_size]
            val_batch_data = data[batch_indices]

            if isinstance(model, GatedGCRNNforClassification):
                h0 = torch.zeros(len(batch_indices), model.F_h, val_batch_data.shape[3]).to(val_batch_data.device)
                pred = model(val_batch_data, h0)
            elif isinstance(model, GatedGCRNNforRegression):
                val_batch_data = val_batch_data.permute(0, 3, 1, 2)
                h0 = torch.zeros(len(batch_indices), model.F_h, val_batch_data.shape[3]).to(val_batch_data.device)
                pred = model(val_batch_data, h0)
            else:
                # print(val_batch_data.shape)
                pred = model(val_batch_data)
            predictions.append(pred)

        predictions = torch.cat(predictions, dim=0)
    return predictions



def accuracy_classification(output: torch.Tensor, target: torch.Tensor):
    """
        Args:
            output (Tensor): The tensor that contains the output or our neural network
            target (Tensor): The corresponding true labels
    """
    acc = accuracy_score(target.cpu(), torch.max(output, dim=1)[1].cpu())
    return acc


def compute_confusion_matrix(output: torch.Tensor, target: torch.Tensor, print_cm=False):
    pred = torch.max(output, dim=1)[1].cpu().numpy()
    true = target.cpu().numpy()
    cm = print_confusion_matrix(y_true=true, y_pred=pred, print_cm=print_cm)
    return cm


def plot_cm(cm_array, title, figsize=(10, 7)):
    df_cm = pd.DataFrame(cm_array, index=[i for i in range(cm_array.shape[0])],
                         columns=[i for i in range(cm_array.shape[0])])
    plt.figure(figsize=figsize)
    sns.heatmap(df_cm, annot=True)

    mask = np.eye(df_cm.shape[0]) == 0
    sns.heatmap(df_cm, mask=mask, cbar=False,
                annot=True, annot_kws={"weight": "bold"})
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.title(title)
    plt.show()