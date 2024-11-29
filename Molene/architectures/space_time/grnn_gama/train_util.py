from tensorboardX import SummaryWriter
from win10toast import ToastNotifier
import numpy as np
import torch
import os

from prediction.train_utils import compute_confusion_matrix


def get_gso_over_time(batch_size, timesteps, gso, device):
    gso_over_time = torch.zeros(size=(batch_size, timesteps, 1, gso.shape[0], gso.shape[0])).to(device)
    gso_over_time[:, :, :] = gso  # repeat over other dimensions
    return gso_over_time

def train_grnn_quakes(model, gso, training_data, validation_data, trn_labels, val_labels,
                       num_epochs, batch_size,
                       loss_criterion, optimizer, scheduler,
                       val_metric_criterion,
                       log_dir, not_learning_limit, show_notifications=False):

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

            batch_pred = model(batch_trn_data)

            # obtain the loss function
            batch_trn_loss = loss_criterion(batch_pred, batch_trn_labels.long())
            batch_losses.append(batch_trn_loss.item())

            optimizer.zero_grad()
            batch_trn_loss.backward()
            optimizer.step()

        epoch_trn_loss = np.average(batch_losses)
        tensorboard.add_scalar('train-loss', epoch_trn_loss, epoch)

        val_pred = perform_chunk_predictions_GRNN(model, validation_data, gso, chunk_size=batch_size)
        val_loss = round(loss_criterion(val_pred, val_labels.long()).item(), 3)


        #val_loss = compute_loss_in_chunks(model, validation_data, val_labels.long(), loss_criterion, chunk_size=batch_size)

        if val_metric_criterion:
            val_metric = compute_loss_in_chunks_GRNN(model, validation_data, val_labels, val_metric_criterion, gso, chunk_size=batch_size)
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
            cm = compute_confusion_matrix(output=val_pred, target=val_labels.long(), print_cm=False)
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



def perform_chunk_predictions_GRNN(model, data, gso, chunk_size):
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

            gso_over_time = get_gso_over_time(
                batch_size=val_batch_data.shape[0],
                timesteps=val_batch_data.shape[1],
                gso=gso,
                device=gso.device
            )

            pred = model(val_batch_data, gso_over_time)
            predictions.append(pred)

        predictions = torch.cat(predictions, dim=0)
    return predictions



def compute_loss_in_chunks_GRNN(model, data, labels, criterion, gso, chunk_size=300):
    predictions = perform_chunk_predictions_GRNN(model, data, gso, chunk_size)
    val_loss = round(criterion(predictions, labels).item(), 3)
    return val_loss
