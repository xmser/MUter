import os
import torch
from functorch import vmap, jacrev, grad, jacfwd, make_functional
from model.common_model import TestModel
from utils import paramters_to_vector, total_param
from data_utils import Dataer
from Train import Neter
from tqdm import tqdm
from kfac_utils import ComputeCovA, ComputeCovG
import time

class BasicBlock:

    def __init__(self, net, dataer, criterion, device):

        self.net = net
        self.dataer = dataer
        self.criterion = criterion
        self.device = device

        self.func_model, self.params = make_functional(self.net)

        self.W = {}
        for name, param in self.net.named_parameters():
            self.W[name] = param

    def get_sample_grads(self, head=-1, rear=-1, isTrain=True, batch_size=128, isAdv=False):

        def compute_loss(params, image, label):
            single_batch_image = image.unsqueeze(0)
            single_batch_label = label.unsqueeze(0)
            preds = self.func_model(params, single_batch_image)
            return self.criterion(preds, single_batch_label)

        per_sample_grads = vmap(grad(compute_loss), in_dims=(None, 0, 0))
        loader = self.dataer.get_loader(head=head, rear=rear, isTrain=isTrain, batch_size=batch_size, isAdv=isAdv)

        
        for image, label in loader:
            image = image.to(self.device)
            label = label.to(self.device)
            batch_grad = list(per_sample_grads(self.params, image, label))
            
            lenth = len(batch_grad)
            for index in range(lenth):
                batch_grad[index] = batch_grad[index].reshape(batch_grad[index].shape[0], -1)

            batch_grad = torch.cat(batch_grad, dim=1)
            print(batch_grad.shape)

            break

    def test_jacobian(self, ):

        def compute_loss(params, image, label):
            preds = self.func_model(params, image)
            return self.criterion(preds, label)

        jacobian_sample = jacrev(compute_loss, argnums=1)
        jacobian_sample_sample = jacrev(jacobian_sample, argnums=1)

        loader = self.dataer.get_loader(isTrain=True, batch_size=1)

        for image, label in loader:
            image = image.to(self.device)
            label = label.to(self.device)
            image.requires_grad = True
            partial_xx = jacobian_sample_sample(self.params, image, label)

            print(partial_xx.sum())
            break
    
    def test2_jacobian(self):

        def set_one(vec, index):
            vec[index] = 1.0
            if index > 0:
                vec[index-1] = 0.0
            return vec
        
        loader = self.dataer.get_loader(isTrain=True, batch_size=1)

        name = self.get_layer_name()
        for image, label in loader:
            image = image.to(self.device)
            label = label.to(self.device)
            image.requires_grad = True
            output = self.net(image)
            loss = self.criterion(output, label)
            grad_x = paramters_to_vector(torch.autograd.grad(loss, image, create_graph=True, retain_graph=True))
            vec = torch.zeros_like(grad_x).to(self.device)
            size = grad_x.shape[0]
            
            start = time.time()
            print(self.W[name[16]].shape)
            grad_xx = [torch.autograd.grad(grad_x, self.W[name[16]], set_one(vec, index), retain_graph=True)[0] for index in range(size)]
            end = time.time()

            for item in grad_xx:
                print(item.shape)

            print('time {:.2f}'.format(end - start)) 

            break



    def get_layer_info(self):
        
        for key, value in self.W.items():
            print('layer name : {}, layer size : {}'.format(key, value.shape))

    def get_layer_name(self):

        names = []
        
        for key in self.W.keys():
            names.append(key)
        
        return names

    def Kronecker_factor(self):
        pass



if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    dataer = Dataer(dataset_name='Cifar10')
    neter = Neter(dataer=dataer, args=None)
    # blocker = BasicBlock(net=neter.net, dataer=dataer, criterion=neter.criterion, device=neter.device)
    # # blocker.get_sample_grads()
    # blocker.test_jacobian()
