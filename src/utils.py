import math
import hydra



def exponential_lr_decay(step: int, k: float):
    return math.e ** (-step * k)

def load_datasets(dataset: str):
    if dataset == "mnist":
        import torchvision
        transform = [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))]

        train_dataset = torchvision.datasets.MNIST(
            "/tmp/data",
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(transform),
        )
        test_dataset = torchvision.datasets.MNIST(
            "/tmp/data",
            train=False,
            download=True,
            transform=torchvision.transforms.Compose(transform),
        )
        return train_dataset, test_dataset
    ###################################################################
    #                TODO: Implement other datasets                   #
    ###################################################################





    ###################################################################
    else:
        raise ValueError(f"Unknown dataset: {dataset}")