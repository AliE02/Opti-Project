import os
import time

import hydra
import torch
import torch.nn.functional as F
import torchvision
from omegaconf import DictConfig, OmegaConf
from torch.utils import tensorboard

from src.loss import xent
from src.utils import load_datasets

from sklearn.metrics import precision_score, recall_score, f1_score

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
    writer = tensorboard.writer.SummaryWriter(os.path.join(os.getcwd(), "logs/backprop"))

    # Dataset creation
    train_dataset, test_dataset, val_dataset = load_datasets(cfg.dataset_name, cfg.seed)
    if val_dataset:
        val_loader = hydra.utils.instantiate(cfg.dataset, dataset=val_dataset)

    train_loader = hydra.utils.instantiate(cfg.dataset, dataset=train_dataset)
    test_loader = hydra.utils.instantiate(cfg.dataset, dataset=test_dataset)

    if cfg.model_name == "convnet":
        model: torch.nn.Module = hydra.utils.instantiate(cfg.model, input_size=1, output_size=len(test_dataset.classes))

    ##################################################################
    #              TODO: Implement other models loading              #
    ##################################################################






    ##################################################################
    else:
        model: torch.nn.Module = hydra.utils.instantiate(cfg.model)
        
    model.to(device)
    model.float()
    model.train()
    params = model.parameters()

    optimizer: torch.optim.Optimizer = hydra.utils.instantiate(cfg.optimizer, params=params)
    optimizer.zero_grad(set_to_none=True)

    scheduler: torch.optim.lr_scheduler._LRScheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)

    steps = 0
    t_total = 0.0
    for epoch in range(total_epochs):
        t0 = time.perf_counter()
        for batch in train_loader:
            steps += 1
            images, labels = batch
            loss = xent(model, images.to(device), labels.to(device))
            loss.backward()
            if grad_clipping > 0:
                torch.nn.utils.clip_grad.clip_grad_norm_(
                    parameters=params, max_norm=grad_clipping, error_if_nonfinite=True
                )

            # get the norm of the gradients
            grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in model.parameters()]))

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            writer.add_scalar("Loss/train_loss_step", loss, steps)
            writer.add_scalar("Misc/lr", scheduler.get_last_lr()[0], steps)

        # write the last training loss of the epoch
        writer.add_scalar("Loss/train_loss_epoch", loss, epoch+1)

        t1 = time.perf_counter()
        t_total += t1 - t0
        writer.add_scalar("Time/batch_time", t1 - t0, steps)
        writer.add_scalar("Time/sps", steps / t_total, steps)
        print(f"Epoch [{epoch+1}/{total_epochs}], Loss: {loss.item():.4f}, Time (s): {t1 - t0:.4f}")
    print("Mean time:", t_total / total_epochs)

    # Test
    acc = 0
    true = []
    pred = []
    for batch in test_loader:
        images, labels = batch
        out = model(images.to(device))
        pred = F.softmax(out, dim=-1).argmax(dim=-1)

        true += labels.tolist()
        pred += pred.tolist()

        acc += (pred == labels.to(device)).sum()

    precision = precision_score(true, pred, average="macro")
    recall = recall_score(true, pred, average="macro")
    f1 = f1_score(true, pred, average="macro")

    writer.add_scalar("Test/precision", precision, steps)
    writer.add_scalar("Test/recall", recall, steps)
    writer.add_scalar("Test/f1", f1, steps)
    writer.add_scalar("Test/accuracy", acc / len(test_dataset), steps)
    print(f"Test accuracy: {(acc / len(test_dataset)).item():.4f}")


if __name__ == "__main__":
    train_model()