from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from functorch import vmap, jacrev, grad, make_functional

from data_utils import Dataer

device = 'cuda'

def unit_vec(vec, index):
    if index != 0:
        vec[index - 1] = 0
    vec[index] = 1
    return vec

def vectorize(params):
    arr = []
    for param in params:
        arr.append(param.view(-1))
    return torch.cat(arr)

class simple_model(nn.Module):

    def __init__(self):
        
        super(simple_model, self).__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=10, bias=False)
        # self.fc2 = nn.Linear(in_features=100, out_features=10, bias=False)
    def forward(self, x):
        # x = F.relu(self.fc1(x.view(-1, 784)))
        # return self.fc2(x)
        return self.fc1(x.view(-1, 784))


dataer = Dataer('Mnist')

net = simple_model().to(device)

func, net_params = make_functional(net)

criterion = nn.CrossEntropyLoss()

pass_loader = dataer.get_loader(batch_size=1, isTrain=True)

vec = torch.zeros(784).to(device)

# for (image, label) in tqdm(pass_loader):
#     image, label = image.to(device), label.to(device)
#     image.requires_grad = True
#     output = net(image)
#     loss = criterion(output, label)
#     partial_x = vectorize(torch.autograd.grad(loss, image, create_graph=True, retain_graph=True)[0])
#     partial_xx = [vectorize(torch.autograd.grad(partial_x, image, grad_outputs=unit_vec(vec, index), retain_graph=True)[0]) for index in range(784)]

def compute_loss(param, image, label):
    output = func(param, image)
    return criterion(output, label)

for (image, label) in tqdm(pass_loader):
    image, label = image.to(device), label.to(device)
    image.requires_grad = True
    batch_partial_hessian = jacrev(jacrev(compute_loss, argnums=1), argnums=0)
    partial_xx = batch_partial_hessian(net_params, image, label)

