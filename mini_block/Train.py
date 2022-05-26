from turtle import update
import torch
import torch.nn as nn
from model import *
from tqdm import tqdm
import numpy as np
import copy
import time
from torchattacks import FGSM, PGD

class Neter:

    def __init__(self, dataer, args, criterion=nn.CrossEntropyLoss(), device='cuda'):

        self.criterion = criterion
        self.dataer = dataer
        self.device = device
        self.args = args
        self.net = None
        self.default_path = None
        self.atk_info = {
            'Cifar10': (0.3, 0.03, 20),
            'Mnist': (0.3, 0.03, 20),
        }
        
        if dataer.dataset_name == 'Mnist':
            self.net = TestModel(784, 10).to(self.device)
        elif dataer.dataset_name == 'Cifar10':
            self.net = ResNet(ResidualBlock).to(self.device)
        else:
            raise Exception('No suchh dataset called {}'.format(dataer.dataset_name))
    
    def copy(self, basic_neter):
        
        self.criterion = basic_neter.criterion
        self.device = basic_neter.device
        self.net = copy.deepcopy(basic_neter.net).to(self.device)
        self.default_path = None
        
    def training(self, epochs=100, lr=0.1, batch_size=128, isAdv=False, verbose=False, head=-1, rear=-1):

        train_loader = self.dataer.get_loader(batch_size=batch_size, isTrain=True, head=head, rear=rear)
        optimizer = torch.optim.SGD(self.net.parameters(), lr=lr)

        if isAdv:
            atk = PGD(self.net, self.atk_info[self.args.dataset][0], self.atk_info[self.args.dataset][1], self.atk_info[self.args.dataset][2])

        start_time = time.time()
        for epoch in range(1, epochs+1):

            lenth = len(train_loader)
            avg_loss = 0.0
            steps = 1
            with tqdm(total=lenth) as pbar:
                pbar.set_description('Epoch [{}/{}]  Lr {}'.format(epoch, epochs, lr))
                for (image, label) in train_loader:
                    image = image.to(self.device)
                    label = label.to(self.device)

                    if isAdv:
                        image = atk(image, label).to(self.device)

                    output = self.net(image)

                    loss = self.criterion(output, label)
                    avg_loss += loss.item()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    pbar.set_postfix(loss='{:.4f}'.format(avg_loss / steps))
                    pbar.update(1)
                    steps += 1

            if epoch % 10 == 0:
                print('Train acc: {:.2f}%'.format(self.test(isTrainset=True) * 100), end='  ')
                print('Test acc: {:.2f}%'.format(self.test(isTrainset=False) * 100))
        
        end_time = time.time()

        return (end_time - start_time)
        
    def test(self, batch_size=128, isTrainset=False, isAttck=False):
        
        loader = self.dataer.get_loader(batch_size=batch_size, isTrain=isTrainset)
        
        if isAttck:
            atk = PGD(self.net, self.atk_info[self.args.dataset][0], self.atk_info[self.args.dataset][1], self.atk_info[self.args.dataset][2])

        total = 0
        correct = 0

        for (image, label) in loader:
            image = image.to(self.device)
            label = label.to(self.device)

            if isAttck:
                image = atk(image, label).to(self.device)

            output = self.net(image)
            _, pred = torch.max(output.data, 1)
            total += image.shape[0]
            correct += (pred == label).sum()
        
        return float(correct) / total
    
    def save_model(self, path=None):

        if path == None:
            path = self.default_path

        torch.save(self.net.state_dict(), f=path)
        print('save done !')

    def load_model(self, path=None):

        if path == None:
            path = self.default_path

        self.net.load_state_dict(torch.load(f=path))
        print('load done !')
    
    def Reset_model_parameters_by_layers(self, delta_w):
        """
        delta_w is a dict, have the same name like the net itself.
        """
        for name, param in self.net.named_parameters():
            param.data += delta_w[name].view_as(param.data)
        
        print('layers update done !')
    
    def Reset_model_parameters_by_vector(self, delta_w):

        head = 0
        
        for param in self.net.parameters():
            numbers = np.prod(param.data.shape)
            param.data += delta_w[head : head + head + numbers]
            head += numbers

        print('vector update done !')
    
    
