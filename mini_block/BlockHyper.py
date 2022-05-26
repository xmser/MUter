import torch
from functorch import vmap
from model import TestModel
from utils import paramters_to_vector
from datamarket import Dataer
from Train import Neter
from tqdm import tqdm
class BasicBlock:

    def __init__(self, net, dataer, criterion, device):

        self.net = net
        self.dataer = dataer
        self.criterion = criterion
        self.device = device

        self.W = {}
        for name, param in self.net.named_parameters():
            self.W[name] = param

    def get_layer_info(self):
        
        for key, value in self.W.items():
            print('layer name : {}, layer size : {}'.format(key, value.shape))

    def get_layer_name(self):

        names = []
        
        for key in self.W.keys():
            names.append(key)
        
        return names

    def get_layer_grad(self, layers_names, batch_size=None):
        
        layers_grad = {}
        
        if batch_size == None:
            batch_size = 128
        loader = self.dataer.get_loader(batch_size=batch_size, isTrain=True)

        for (image, label) in tqdm(loader):
            image = image.to(self.device)
            label = label.to(self.device)
            
            output = self.net(image)
            loss = self.criterion(output, label)

            for layer_name in layers_names:
                if layer_name not in layers_grad:
                    layers_grad[layer_name] = paramters_to_vector(torch.autograd.grad(loss, self.W[layer_name], retain_graph=True)[0].detach())
                else:
                    layers_grad[layer_name] += paramters_to_vector(torch.autograd.grad(loss, self.W[layer_name], retain_graph=True)[0].detach())

        return layers_grad



if __name__ == "__main__":

    dataer = Dataer(dataset_name='Cifar10')
    neter = Neter(dataer)



    # neter.training(verbose=True, batch_size=128, epochs=100)

    bb = BasicBlock(neter.net, dataer, criterion=torch.nn.CrossEntropyLoss(reduction='sum'), device='cuda')

    # ret = bb.get_layer_grad(layers_names=bb.get_layer_name())



