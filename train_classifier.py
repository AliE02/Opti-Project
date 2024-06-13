import copy
import math
import os
import time
from functools import partial

import hydra
import torch
import torch.func as fc
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils import tensorboard
from tqdm import tqdm
from datetime import datetime
from importlib import import_module

from src.utils import load_datasets
from sklearn.metrics import precision_score, recall_score, f1_score

import multiprocessing

OmegaConf.register_new_resolver("get_method", hydra.utils.get_method)


@hydra.main(config_path="./configs/", config_name="config.yaml")
def train_model(cfg: DictConfig):
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{cfg.device_id}" if use_cuda else "cpu")
    total_epochs = cfg.epochs
    grad_clipping = cfg.grad_clipping

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

    # Summary
    writer = tensorboard.writer.SummaryWriter(os.path.join(os.getcwd(), "logs/fwdgrad"))

    train_dataset, test_dataset, val_dataset = load_datasets(cfg.dataset_name)

    train_loader = hydra.utils.instantiate(cfg.dataset, dataset=train_dataset)
    test_loader = hydra.utils.instantiate(cfg.dataset, dataset=test_dataset)
    if val_dataset:
        val_loader = hydra.utils.instantiate(cfg.dataset, dataset=val_dataset)


    with torch.no_grad():
        if cfg.dataset_name == "esc10":
            model: torch.nn.Module = hydra.utils.instantiate(cfg.model, 2, len(test_dataset.classes))
        elif cfg.dataset_name == "mnist":
            model: torch.nn.Module = hydra.utils.instantiate(cfg.model, 1, len(test_dataset.classes))

        ##################################################################
        #              TODO: Implement other models loading              #
        ##################################################################






        ##################################################################
        else:
            model: torch.nn.Module = hydra.utils.instantiate(cfg.model)
        model.to(device)
        model.float()
        model.train()

        optimizer: torch.optim.Optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
        optimizer.zero_grad(set_to_none=True)

        scheduler: torch.optim.lr_scheduler._LRScheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)

        loss_function = hydra.utils.get_method(cfg.loss)

        named_buffers = dict(model.named_buffers())
        named_params = dict(model.named_parameters())
        names = named_params.keys()
        params = named_params.values()

        base_model = copy.deepcopy(model)
        base_model.to("meta")

        
        # if it doesn't exist, create directory checkpoints
        if not os.path.exists("checkpoints"):
            os.makedirs("checkpoints")

        # Train
        steps = 0
        t_total = 0.0

        for epoch in tqdm(range(total_epochs)):
            base_model.train()
            t0 = time.perf_counter()
            for batch in train_loader:
                steps += 1
                images, labels = batch

                # Sample perturbation (tangent) vectors for every parameter of the model
                v_params = tuple([torch.randn_like(p) for p in params])
                f = partial(
                    loss_function,
                    model=base_model,
                    names=names,
                    buffers=named_buffers,
                    x=images.to(device),
                    t=labels.to(device),
                )

                # Forward AD
                loss, jvp = fc.jvp(f, (tuple(params),), (v_params,))

                # Setting gradients
                for v, p in zip(v_params, params):
                    p.grad = v * jvp

                # Clip gradients
                if grad_clipping > 0:
                    torch.nn.utils.clip_grad.clip_grad_norm_(
                        parameters=params, max_norm=grad_clipping, error_if_nonfinite=True
                    )

                # get the gradient norm
                grad_norm = torch.linalg.vector_norm(
                    torch.cat([torch.flatten(p.grad) for p in params])
                )

                # Optimizer step
                optimizer.step()

                # Lr scaling
                scheduler.step()

                # Zero out grads
                optimizer.zero_grad(set_to_none=True)

                writer.add_scalar("Loss/train_loss_step", loss, steps)
                writer.add_scalar("Misc/lr", scheduler.get_last_lr()[0], steps)

            # write the last training loss of the epoch
            writer.add_scalar("Loss/train_loss_epoch", loss, epoch+1)

            # add validation accuracy
            if val_dataset:
                acc = 0
                losses = []
                for val_batch in val_loader:
                    val_images, val_labels = val_batch
                    val_images = val_images.contiguous().to(device)
                    val_labels = val_labels.contiguous().to(device)

                    val_out = fc.functional_call(base_model, (named_params, named_buffers), (val_images,))
                    val_loss = loss_function(model, names, named_buffers, val_images, val_labels)
                    losses.append(val_loss)

                    # Importing the activation function
                    module_path, function_name = cfg.activation.rsplit('.', 1)
                    module = import_module(module_path)
                    activation = getattr(module, function_name)

                    val_pred = activation(val_out, dim=-1).argmax(dim=-1)
                    acc += (val_pred == val_labels.to(device)).sum()

                writer.add_scalar("Val/loss_step", sum(losses) / len(losses), steps)
                writer.add_scalar("Val/loss_epoch", sum(losses) / len(losses), epoch+1)
                writer.add_scalar("Val/accuracy_step", acc / len(val_dataset), steps)
                writer.add_scalar("Val/accuracy_epoch", acc / len(val_dataset), epoch+1)
                print(f"Validation accuracy: {(acc / len(val_dataset)).item():.4f}")


            # Save model checkpoint
            now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            torch.save(model.state_dict(), f"checkpoints/{cfg.model_name}_epoch_{epoch}_at_{now}.pth")

            t1 = time.perf_counter()
            t_total += t1 - t0
            writer.add_scalar("Time/batch_time", t1 - t0, steps)
            writer.add_scalar("Time/sps", steps / t_total, steps)
            print(f"Epoch [{epoch+1}/{total_epochs}], Loss: {loss.item():.4f}, Time (s): {t1 - t0:.4f}, grad_norm: {grad_norm:.4f}")
        print(f"Mean time: {t_total / total_epochs:.4f}")

        # Test
        acc = 0
        true = []
        pred = []
        for batch in tqdm(test_loader):
            images, labels = batch
            out = fc.functional_call(base_model, (named_params, named_buffers), (images.to(device),))

            # Importing the activation function
            module_path, function_name = cfg.activation.rsplit('.', 1)
            module = import_module(module_path)
            activation = getattr(module, function_name)

            pred = activation(out, dim=-1).argmax(dim=-1)

            # add predictions and labels to the lists
            true += labels.tolist()
            pred += pred.tolist()

            acc += (pred == labels.to(device)).sum()

        # compute precision, recall and f1 score macro-averages
        precision = precision_score(true, pred, average="macro")
        recall = recall_score(true, pred, average="macro")
        f1 = f1_score(true, pred, average="macro")

        writer.add_scalar("Test/accuracy", acc / len(test_dataset), steps)
        writer.add_scalar("Test/precision", precision, steps)
        writer.add_scalar("Test/recall", recall, steps)
        writer.add_scalar("Test/f1", f1, steps)
        print(f"Test accuracy: {(acc / len(test_dataset)).item():.4f}")





if __name__ == "__main__":
    processes = []
    config_names = ["complex_esc", "complex_mnist", "basic_esc"]
    for name in config_names:
        @hydra.main(config_path="./configs/", config_name=f"{name}.yaml")
        def train_model_bis(cfg: DictConfig):
            train_model(cfg)

        p = multiprocessing.Process(target=train_model_bis)
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    print("Training completed successfully!")