import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def paramters_to_vector(params):

    vec = []

    for param in params:
        vec.append(param.view(-1))
    
    return torch.cat(vec)

def vector_to_parameters(vec, paramters):

    pointer = 0

    for param in paramters:
        num_param = param.numel()
        param.data = vec[pointer : pointer + num_param].view_as(param).data
        pointer += num_param
    
def total_param(model):
    
    number = 0
    
    for param in model.parameters():
        number = number + np.prod(list(param.shape))

    return number

def get_layers(str, input_features=640, output_features=10, isBias=False):

    if str == 'linear':
        return nn.Linear(in_features=input_features, out_features=output_features, bias=isBias)
    elif str == 'MLP':
        return nn.Sequential(
            nn.Linear(in_features=input_features, out_features=100, bias=isBias),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=output_features, bias=isBias)
        )
    else:
        raise Exception('No such method called {}, please recheck !'.format(str))