# Project Name

Welcome to our Optimization for Machine Learning project.
In this project we are going to answer the following question:
- Forward vs Backward: What are the differences in performance accross different domains, datasets and architectures?

Our goal is to provide a comprehensive analysis of the performance of forward and backward optimization methods in the context of machine learning accross different domains such as **Computer Vision**, **Natural Language Processing** and **Speech Recognition**.

We are going to be testing the forward and backward optimization methods on the following following tasks:
- **Image Classification**: We are going to be using the CIFAR10 dataset.
- **Sentiment Analysis**: We are going to be using the IMDB dataset.
- **Speech Recognition**: We are going to be using the TIMIT dataset.

## Installation

install all the dependencies by running the following command:

```bash
pip install -r requirements.txt
```


## Usage

This repository contains a train_classifier.py script that you can use for training your own model.
You will have to follow these steps:
- Define how to load your data in the `load_data` function located in the `src/utils.py` file.
This function should return a training and validation dataset all of the type `torch.utils.data.Dataset`.
- Define your model in the `src/model.py` file.
- Define how to load your model in the `load_model` function located in the `src/train_classifier.py` file in the specified section using the `model_name` field in the configuration file.
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
- `dataset_name` is the name of the dataset you want to use (this should be the same as the name used to load the dataset in the `load_data` function).
- `loss` is the path to the loss function you want to use.
- `activation` is the path to the activation function you want to use.

## Using the train_classifier.py script

You can use the `train_classifier.py` script to train your model.
You can do this by running the following command:

```bash
python train_classifier.py --config_path configs/config.yaml
```

## Reproducing plots

outputs folder contains audio experiments
src folder contains :
- roberta folder for text experiments
- vision experiments and baselines for image classification

You can train the models from scratch or use this link to download runs to reproduce plots:
https://drive.google.com/drive/folders/1t1iYu2a_VS-_gdZCKAH6sLUYoAZVmC7H?usp=sharing
https://drive.google.com/drive/folders/11zao50qKBFVhMADqdzNI95Hua6337P_n?usp=sharing

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
