import copy
import math
import numpy as np
import time
import pickle
import json
import torch
import torch.func as fc
import torch.nn.functional as F
from typing import Dict, KeysView, ValuesView
from functools import partial
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from torch import Tensor
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision import transforms


def compute_metrics(preds, labels):
    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"Acc": acc, "F1": f1}


def get_data(dataset_name, batch_size=64, augment=False, img_size=64):
    data_dir = "../data/"
    train_loader, val_loader, test_loader = get_loaders(
        dataset_name,
        data_dir,
        batch_size=batch_size,
        img_size=img_size,
        augment=augment,
    )
    return train_loader, val_loader, test_loader


def get_loaders(dataset_name, data_dir, batch_size, augment, img_size, valid_size=0.1):

    if dataset_name == "MNIST":
        train_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        test_transform = train_transform

    else:
        test_transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010],
                ),
            ]
        )

        train_transform = (
            test_transform
            if not augment
            else transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.4914, 0.4822, 0.4465],
                        std=[0.2023, 0.1994, 0.2010],
                    ),
                ]
            )
        )

    train_val_dataset = eval(
        "datasets." + dataset_name + "(root=data_dir,train=True,download=True)"
    )
    test_dataset = eval(
        "datasets."
        + dataset_name
        + "(root=data_dir,train=False,download=True,transform=test_transform)"
    )

    valid_size = int(0.1 * len(train_val_dataset))
    train_size = len(train_val_dataset) - valid_size
    train_dataset, valid_dataset = torch.utils.data.random_split(
        train_val_dataset, [train_size, valid_size]
    )

    train_dataset.dataset.transform = train_transform
    valid_dataset.dataset.transform = test_transform

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    print(
        f"Train dataset: {len(train_loader.dataset)} batch size {train_loader.batch_size} of dim {train_loader.dataset[0][0].shape}\nValidation dataset: {len(val_loader.dataset)} batch size {val_loader.batch_size} of dim {val_loader.dataset[0][0].shape}\nTest dataset: {len(test_loader.dataset)} batch size {test_loader.batch_size} of dim {test_loader.dataset[0][0].shape}"
    )

    return train_loader, val_loader, test_loader


def exponential_lr_decay(step: int, k: float):
    return math.e ** (-step * k)

def _xent(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Compute cross-entropy loss.

    Args:
        x (torch.Tensor): Output of the model.
        t (torch.Tensor): Targets.

    Returns:
        torch.Tensor: Cross-entropy loss.
    """
    return F.cross_entropy(x, t)


def functional_xent(
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
    return _xent(y, t)


def train_one_epoch_backward(
    device, epoch_index, model, train_loader, input_size, optimizer, gradient_clip_val, flatten
):
    train_epoch = {"Loss": [], "Acc": [], "F1": [], "Time": []}
    start_time = time.time()

    progress_bar = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc="\tTraining Epoch {}".format(epoch_index + 1),
    )

    for i, (images, labels) in progress_bar:
        images = images.view(-1, input_size).to(device) if flatten else images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        output = model(images)

        loss = _xent(output, labels)
        # can retain_graph
        loss.backward()

        if gradient_clip_val > 0:
            torch.nn.utils.clip_grad.clip_grad_norm_(
                parameters=model.parameters(),
                max_norm=gradient_clip_val,
                error_if_nonfinite=True,
            )

        optimizer.step()
        train_epoch["Time"].append(time.time() - start_time)


        preds = F.softmax(output, dim=-1).argmax(dim=-1)
        train_epoch["Loss"].append(loss.item())
        # compute accuracy and f1 score
        metrics = compute_metrics(preds, labels)

        for k in metrics:
            train_epoch[k].append(metrics[k])

        progress_bar.set_postfix(
            loss=train_epoch["Loss"][-1],
            acc=train_epoch["Acc"][-1],
            f1=train_epoch["F1"][-1],
            iters_per_sec=(i + 1) / (time.time() - start_time),
        )

    return train_epoch


def train_one_epoch_forward(
    device,
    epoch_index,
    input_size,
    base_model,
    params,
    named_params,
    names,
    named_buffers,
    train_loader,
    optimizer,
    gradient_clip_val,
    flatten,
):
    train_epoch = {"Loss": [], "Acc": [], "F1": [], "Time": []}
    start_time = time.time()

    progress_bar = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc="\tTraining Epoch {}".format(epoch_index + 1),
    )

    for i, (images, labels) in progress_bar:
        images = images.view(-1, input_size).to(device) if flatten else images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        v_params = tuple([torch.randn_like(p) for p in params])

        f = partial(
            functional_xent,
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

        if gradient_clip_val > 0:
            torch.nn.utils.clip_grad.clip_grad_norm_(
                parameters=params,
                max_norm=gradient_clip_val,
                error_if_nonfinite=True,
            )
            

        optimizer.step()

        train_epoch["Time"].append(time.time() - start_time)

        
        output = fc.functional_call(
            base_model, (named_params, named_buffers), (images,)
        )
        preds = F.softmax(output, dim=-1).argmax(dim=-1)
        train_epoch["Loss"].append(loss.item())

        metrics = compute_metrics(preds, labels)

        for k in metrics:
            train_epoch[k].append(metrics[k])

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
    input_size,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    output_path,
    gradient_clip_val,
    flatten
):

    timestamp = datetime.now().strftime("%d_%m_%Y_start_%Hh%Mm")
    run_path = output_path / "{}Model_start_{}".format(model_name, timestamp)
    writer = SummaryWriter(run_path / "tensorboard_logs")

    best_vloss = float("inf")
    train_run = {"Loss": [], "Acc": [], "F1": [], "Time": []}
    val_run = {"Loss": [], "Acc": [], "F1": [], "Time": []}
    train_run_its = {"Loss": [], "Acc": [], "F1": [], "Time": []}
    val_run_its = {"Loss": [], "Acc": [], "F1": [], "Time": []}

    if forward:
        named_buffers = dict(model.named_buffers())
        named_params = dict(model.named_parameters())
        names = named_params.keys()
        params = named_params.values()
        base_model = copy.deepcopy(model)
        base_model.to("meta")

    for epoch in range(nb_epochs):
        if forward:
            train_epoch_run = train_one_epoch_forward(
                device=device,
                epoch_index=epoch,
                input_size=input_size,
                base_model=base_model,
                params=params,
                named_params=named_params,
                names=names,
                named_buffers=named_buffers,
                train_loader=train_loader,
                optimizer=optimizer,
                gradient_clip_val=gradient_clip_val,
                flatten=flatten
            )
        else:
            train_epoch_run = train_one_epoch_backward(
                device=device,
                epoch_index=epoch,
                model=model,
                train_loader=train_loader,
                input_size=input_size,
                optimizer=optimizer,
                gradient_clip_val=gradient_clip_val,
                flatten=flatten
            )

        val_epoch_run = {"Loss": [], "Acc": [], "F1": [], "Time": []}
        start_time = time.time()
        progress_bar = tqdm(
            enumerate(val_loader),
            total=len(val_loader),
            desc="\tValidating",
        )

        with torch.no_grad():
            for i, (vimages, vlabels) in progress_bar:
                vimages = vimages.view(-1, input_size).to(device) if flatten else vimages.to(device)
                vlabels = vlabels.to(device)
                if forward:
                    voutput = fc.functional_call(
                        base_model, (named_params, named_buffers), (vimages.to(device),)
                    )
                else:
                    voutput = model(vimages)
                vloss = _xent(voutput, vlabels)
                vpreds = F.softmax(voutput, dim=-1).argmax(dim=-1)
                val_epoch_run["Loss"].append(vloss.item())
                val_epoch_run["Time"].append(time.time() - start_time)
                vmetrics = compute_metrics(vpreds, vlabels)
                for k in vmetrics:
                    val_epoch_run[k].append(vmetrics[k])

                progress_bar.set_postfix(
                    vloss=val_epoch_run["Loss"][-1],
                    vacc=val_epoch_run["Acc"][-1],
                    vf1=val_epoch_run["F1"][-1],
                    iters_per_sec=(i + 1) / (time.time() - start_time),
                )

        # update running values
        for k in val_epoch_run:
            train_run[k].append(np.array(train_epoch_run[k]).mean())
            val_run[k].append(np.array(val_epoch_run[k]).mean())
            train_run_its[k].extend(train_epoch_run[k])
            val_run_its[k].extend(val_epoch_run[k])

            writer.add_scalars(
                "Training vs Validation {} ".format(k),
                {
                    "Train {}".format(k): train_run[k][-1],
                    "Val {}".format(k): val_run[k][-1],
                },
                epoch + 1,
            )

        curr_lr = optimizer.param_groups[0]["lr"]
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
        scheduler.step()

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

    # save lists to pickle files
    for k in train_run:
        with open(run_path / "{}_train_run.pkl".format(k), "wb") as f:
            pickle.dump(train_run[k], f)
        with open(run_path / "{}_val_run.pkl".format(k), "wb") as f:
            pickle.dump(val_run[k], f)
        with open(run_path / "{}_train_run_its.pkl".format(k), "wb") as f:
            pickle.dump(train_run_its[k], f)
        with open(run_path / "{}_val_run_its.pkl".format(k), "wb") as f:
            pickle.dump(val_run_its[k], f)

    return train_run, val_run, train_run_its, val_run_its, run_path


def train_validate_forward(
    device,
    nb_epochs,
    model,
    model_name,
    input_size,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    output_path,
    gradient_clip_val=0,
    flatten=True
):
    with torch.no_grad():
        model.train()
        model.float()
        optimizer.zero_grad(set_to_none=True)
        train_run, val_run, train_run_its, val_run_its, run_path = train_validate(
            device,
            nb_epochs,
            model,
            model_name,
            True,
            input_size,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            output_path,
            gradient_clip_val=gradient_clip_val,
            flatten=flatten
        )
    return train_run, val_run, train_run_its, val_run_its, run_path


def train_validate_backward(
    device,
    nb_epochs,
    model,
    model_name,
    input_size,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    output_path,
    gradient_clip_val=0,
    flatten=True
):
    model.train()
    model.float()
    optimizer.zero_grad(set_to_none=True)
    train_run, val_run, train_run_its, val_run_its, run_path = train_validate(
        device,
        nb_epochs,
        model,
        model_name,
        False,
        input_size,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        output_path,
        gradient_clip_val=gradient_clip_val,
        flatten=flatten
    )
    return train_run, val_run, train_run_its, val_run_its, run_path


def evaluate(device, model, input_size, test_loader, run_path=None, flatten=True):
    model.eval()

    eval_run = {"Acc": [], "F1": []}
    start_time = time.time()
    progress_bar = tqdm(
        enumerate(test_loader), total=len(test_loader), desc="\tEvaluating"
    )

    with torch.no_grad():
        for i, (images, labels) in progress_bar:
            images = images.view(-1, input_size).to(device) if flatten else images.to(device)
            labels = labels.to(device)
            output = model(images)
            preds = F.softmax(output, dim=-1).argmax(dim=-1)
            metrics = compute_metrics(preds, labels)
            for k in metrics:
                eval_run[k].append(metrics[k])

                # Update the progress bar with the current loss
            progress_bar.set_postfix(
                eval_acc=eval_run["Acc"][-1],
                eval_f1=eval_run["F1"][-1],
                iters_per_sec=(i + 1) / (time.time() - start_time),
            )

    for k in eval_run:
        eval_run[k] = np.array(eval_run[k]).mean()

    if run_path is not None:
        json.dump(eval_run, open(run_path / "evaluation.json", "w"))

    return eval_run