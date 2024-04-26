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
    y = fc.functional_call(model, ({k: v for k, v in zip(names, params)}, buffers), (x,))
    return _xent(y, t)

def xent(model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss. Given a pytorch model, it computes the cross-entropy loss.

    Args:
        model (torch.nn.Module): PyTorch model.
        x (torch.Tensor): Input tensor for the PyTorch model.
        t (torch.Tensor): Targets.

    Returns:
        torch.Tensor: Cross-entropy loss.
    """
    y = model(x)
    return _xent(y, t)


class LogisticRegression(tn.Module):
    def __init__(self,input_size,num_classes):
        super(LogisticRegression,self).__init__()
        self.linear = tn.Linear(input_size,num_classes)
    
    def forward(self,feature):
        output = self.linear(feature)
        return output



def run_log_reg(train_dataLoader, test_dataLoader, input_size=28*28, num_classes=10, num_epochs=5, learning_rate=1e-3, forward=False):
    run = 0
    model = LogisticRegression(input_size,num_classes)
    named_buffers = dict(model.named_buffers())
    named_params = dict(model.named_parameters())
    names = named_params.keys()
    params = named_params.values()

    base_model = copy.deepcopy(model)
    #crit_loss = tn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

    for epoch in range(num_epochs):
        for i,(images,labels) in enumerate(train_dataLoader):
            images = torch.autograd.Variable(images.view(-1,input_size))
            labels = torch.autograd.Variable(labels)
            
            # Nullify gradients w.r.t. parameters
            optimizer.zero_grad()
            #forward propagation
            output = model(images)
            # compute loss based on obtained value and actual label

            if forward :
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
            else :
                loss = xent(model,images,labels)
                # backward propagation
                loss.backward()
                
            # update the parameters
            optimizer.step()
            run+=1
            
            if (i+1)%200 == 0:
                # check total accuracy of predicted value and actual label
                accurate = 0
                total = 0
                for images,labels in test_dataLoader:
                    images = torch.autograd.Variable(images.view(-1,input_size))
                    output = model(images)
                    _,predicted = torch.max(output.data, 1)
                    # total labels
                    total+= labels.size(0)
                    
                    # Total correct predictions
                    accurate+= (predicted == labels).sum()
                    accuracy_score = 100 * accurate/total
                
                print('Iteration: {}. Loss: {}. Accuracy: {}'.format(run, loss.item(), accuracy_score))

    print('Final Accuracy:',accuracy_score)