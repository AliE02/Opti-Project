from Vision_models import *
import torch
import torch.nn as tn
import matplotlib.pyplot as plt
import torchvision.transforms as tt
import torch.utils as utils
import torch.func as fc
from typing import Dict, KeysView, ValuesView
from torch import Tensor
from torch.nn import functional as F
from functools import partial
import copy
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import time
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score


def compute_metrics(preds, labels):
    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"Acc": acc, "F1": f1}


def cross_entropy(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Compute cross-entropy loss.

    Args:
        x (torch.Tensor): Output of the model.
        t (torch.Tensor): Targets.

    Returns:
        torch.Tensor: Cross-entropy loss.
    """
    return F.cross_entropy(x, t)


def functional_cross_entropy(
    params: ValuesView,
    buffers: Dict[str, Tensor],
    names: KeysView,
    model: torch.nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Functional cross-entropy loss. Given a pytorch model it computes the cross-entropy loss
    in a functional way.

    Args:
        params: Model parameters.
        buffers: Buffers of the model.
        names: Names of the parameters.
        model: A pytorch model.
        x (torch.Tensor): Input tensor for the PyTorch model.
        t (torch.Tensor): Targets.

    Returns:
        torch.Tensor: Cross-entropy loss.
    """
    y = fc.functional_call(
        model, ({k: v for k, v in zip(names, params)}, buffers), (x,)
    )
    return cross_entropy(y, t)


def train_one_epoch(
    device,
    epoch_index,
    max_epochs,
    model,
    base_model,
    params,
    names,
    named_buffers,
    train_loader,
    input_size,
    optimizer,
    forward,
    gradient_clip_val=None,
):
    train_epoch = {"Loss": [], "Acc": [], "F1": []}
    start_time = time.time()

    # Create a progress bar given than train loader has a sampler
    progress_bar = tqdm(  
        enumerate(train_loader),
        total = len(train_loader),
        desc="\tTraining Epoch {}".format(epoch_index + 1),
    )


    for i, (images, labels) in progress_bar:
        images = torch.autograd.Variable(images).to(device)
        labels = torch.autograd.Variable(labels).to(device)

        optimizer.zero_grad()

        output = model(images)

        if forward:
            v_params = tuple([torch.randn_like(p) for p in params])
            f = partial(
                functional_cross_entropy,
                model=base_model,
                names=names,
                buffers=named_buffers,
                x=images,
                t=labels
            )
            # Forward AD
            loss, jvp = fc.jvp(f, (tuple(params),), (v_params,))
            # Setting gradients
            for v, p in zip(v_params, params):
                p.grad = v * jvp
        else:
            loss = cross_entropy(output, labels)
            # can retain_graph
            loss.backward()

        if gradient_clip_val is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)

        preds = torch.argmax(output, dim=1)
        train_epoch["Loss"].append(loss.item())
        # compute accuracy, and f1 score using pytorch functions
        metrics = compute_metrics(preds, labels)

        for k in metrics:
            train_epoch[k].append(metrics[k])

        optimizer.step()

        # Update the progress bar with the current loss
        progress_bar.set_postfix(
            loss=train_epoch["Loss"][-1],
            acc=train_epoch["Acc"][-1],
            f1=train_epoch["F1"][-1],
            iters_per_sec=(i + 1) / (time.time() - start_time),
        )

    return train_epoch


def train_validate(
    device,
    nb_epochs,
    model,
    model_name,
    forward,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    output_path,
    es_patience,
    es_min_delta,
    gradient_clip_val=None
):

    named_buffers = dict(model.named_buffers())
    named_params = dict(model.named_parameters())
    names = named_params.keys()
    params = named_params.values()
    base_model = copy.deepcopy(model)

    input_size = train_loader.dataset[0][0].shape.numel()

    timestamp = datetime.now().strftime("%d_%m_%Y_start_%Hh%Mm")
    run_path = output_path / "runs/{}Model_start_{}".format(model_name, timestamp)
    writer = SummaryWriter(run_path / "tensorboard_logs")

    best_vloss = float("inf")
    # early_stopper = EarlyStopper(patience=es_patience, min_delta=es_min_delta)
    model = model.to(device)
    train_run = {"Loss": [], "Acc": [], "F1": []}
    val_run = {"Loss": [], "Acc": [], "F1": []}

    for epoch in range(nb_epochs):

        model.train()

        train_epoch_run = train_one_epoch(
            device=device,
            epoch_index=epoch,
            max_epochs=nb_epochs,
            model=model,
            base_model=base_model,
            params=params,
            names=names,
            named_buffers=named_buffers,
            train_loader=train_loader,
            input_size=input_size,
            optimizer=optimizer,
            forward=forward,
            gradient_clip_val=gradient_clip_val,
        )

        val_epoch_run = {"Loss": [], "Acc": [], "F1": []}
        start_time = time.time()
        progress_bar = tqdm(
            enumerate(val_loader),
            total=len(val_loader),
            desc="\tValidating",
        )

        model.eval()
        
        with torch.no_grad():
            for i, (vimage, vlabels) in progress_bar:
                vimage, vlabel = vimage.to(device), vlabels.to(device)
                voutput = model(vimage)
                vloss = cross_entropy(voutput, vlabel)
                vpreds = torch.argmax(voutput, dim=1)
                val_epoch_run["Loss"].append(vloss.item())
                vmetrics = compute_metrics(vpreds, vlabel)
                for k in vmetrics:
                    val_epoch_run[k].append(vmetrics[k])

                # Update the progress bar with the current loss
                progress_bar.set_postfix(
                    vloss=val_epoch_run["Loss"][-1],
                    vacc=val_epoch_run["Acc"][-1],
                    vf1=val_epoch_run["F1"][-1],
                    vauc=val_epoch_run["Auc"][-1],
                    iters_per_sec=(i + 1) / (time.time() - start_time),
                )

        # average train and val running values
        for k in val_epoch_run:
            train_run[k].append(np.array(train_epoch_run[k]).mean())
            val_run[k].append(np.array(val_epoch_run[k]).mean())
            # tensorboard logging
            writer.add_scalars(
                "Training vs Validation {} ".format(k),
                {
                    "Train {}".format(k): train_run[k][-1],
                    "Val {}".format(k): val_run[k][-1],
                },
                epoch + 1,
            )

        curr_lr = optimizer.param_groups[0]["lr"]
        # print the epoch, learning rate, average val loss and metrics, avg training loss and metrics
        print(
            "Epoch {} lr: {} Averages : Train Loss: {:.4f} Val Loss: {:.4f} Train Acc: {:.4f} Val Acc: {:.4f} Train F1: {:.4f} Val F1: {:.4f}".format(
                epoch + 1,
                curr_lr,
                train_run["Loss"][-1],
                val_run["Loss"][-1],
                train_run["Acc"][-1],
                val_run["Acc"][-1],
                train_run["F1"][-1],
                val_run["F1"][-1],
            )
        )

        avg_val_loss = val_run["Loss"][-1]
        scheduler.step(avg_val_loss)

        writer.flush()

        # if early_stopper.early_stop(avg_val_loss):
        #    print("Early stopping activated : epoch {}".format(epoch))
        #    break

        if avg_val_loss < best_vloss:
            best_vloss = avg_val_loss
            best_model_path = (
                run_path
                / "best_{}Model_start_{}_stoppedAtEpoch{}_lr{}".format(
                    model_name, timestamp, epoch, curr_lr
                )
            )
            print("Best model saved")
            torch.save(model, best_model_path)
    last_model_path = run_path / "last_{}Model_start_{}_stoppedAtEpoch{}_lr{}".format(
        model_name, timestamp, epoch, curr_lr
    )
    torch.save(model, last_model_path)
    return train_run, val_run, run_path, best_model_path, last_model_path


def run_model(
    model,
    train_loader,
    valid_dataLoader,
    num_classes,
    num_epochs=5,
    learning_rate=1e-3,
    forward=False,
):
    run = 0
    named_buffers = dict(model.named_buffers())
    named_params = dict(model.named_parameters())
    names = named_params.keys()
    params = named_params.values()
    print(train_loader.dataset.data.shape)
    input_size = train_loader.dataset.data.shape[1] * train_loader.dataset.data.shape[2]
    print(input_size)

    base_model = copy.deepcopy(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for e in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = torch.autograd.Variable(images.view(-1, input_size))
            labels = torch.autograd.Variable(labels)

            optimizer.zero_grad()

            output = model(images)

            if forward:
                v_params = tuple([torch.randn_like(p) for p in params])
                f = partial(
                    functional_cross_entropy,
                    model=base_model,
                    names=names,
                    buffers=named_buffers,
                    x=images,
                    t=labels,
                )

                # Forward AD
                loss, jvp = fc.jvp(f, (tuple(params),), (v_params,))

                # Setting gradients
                for v, p in zip(v_params, params):
                    p.grad = v * jvp
            else:
                loss = cross_entropy(output, labels)
                loss.backward()

            # update the parameters
            optimizer.step()
            run += 1

            for i, (images, labels) in enumerate(valid_dataLoader):
                images = torch.autograd.Variable(images.view(-1, input_size))
                labels = torch.autograd.Variable(labels)

                with torch.no_grad():
                    output = model(images)
                    loss = cross_entropy(output, labels)

                if i % 100 == 0:
                    print(f"Epoch {e} - Iteration {i} - Loss: {loss.item()}")

    print("Final Accuracy:", accuracy_score)
