# Project Name

A brief description of your project.

## Installation

install all the dependencies by running the following command:

```bash
pip install -r requirements.txt
```


## Usage

This repository contains a mnist_fwdgrad.py script that you can use for training your own model.
You will have to follow these steps:
- Define how to load your data in the `load_data` function located in the `src/utils.py` file.
This function should return a training and validation dataset all of the type `torch.utils.data.Dataset`.
- Define your model in the `src/models.py` file.
- Define how to load your model in the `load_model` function located in the `src/mnist_fwdgrad.py` file in the specified section using the `model_name` field in the configuration file.
- Define your loss function in the `src/losses.py` file (if it is not already implemented).


## Configuration

Now that you have defined how to load your data, your model and your loss function, you need to define the configuration of your training.
You can do this by modifying the `config.yaml` file located in the `configs` directory.
The configuration structure is as follows:
    
```yaml
defaults:
    - _self_
    - dataset: default
    - model: default
    - optimizer: default
    - scheduler: default

epochs: 200
device_id: 0
grad_clipping: 0.0
model_name: "convnet"
dataset_name: "mnist"
loss: src.loss.functional_xent
activation: torch.nn.functional.softmax
```

The defaults section is used to define the config files related to the dataset, model, optimizer and scheduler, (these are stored in the directories `configs/dataset`, `configs/model`, `configs/optimizer` and `configs/scheduler` respectively).
The other fields are used to define parameters that are also needed for the training process.
Typically:
- `epochs` is the number of epochs you want to train your model.
- `device_id` is the id of the GPU you want to use for training.
- `grad_clipping` is the value of the gradient clipping you want to use.
- `model_name` is the name of the model you want to use (this should be the same as the name used to load the model in the `train_model` function).


## Testing

Instructions on how to run tests, if applicable.

## Contributing

Guidelines for contributing to the project.

## FAQs

Frequently asked questions and their answers.

## License

Specify the license under which your code is released.