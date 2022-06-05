from ast import Break
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import Sampler

class SelfSampler(Sampler):

    def __init__(self, dataset, head=-1, rear=-1):
        
        self.head = head
        self.rear = rear
        
        if self.head == -1:
            self.head = 0
        if self.rear == -1:
            self.rear = int(len(dataset))

        self.lenth = self.rear - self.head
        self.indices = list(range(self.head, self.rear))

    def __iter__(self):
        
        return iter(self.indices)

    def __len__(self):
        
        return len(self.indices)


class Dataer:

    def __init__(self, dataset_name):
        
        self.dataset_name = dataset_name
        self.default_path = './data'
        if dataset_name == 'Mnist':
            transform = transforms.Compose([transforms.ToTensor(), ])
            self.datasets = [
                torchvision.datasets.MNIST(root='./data/mnist', train=True, transform=transform, download=True),
                torchvision.datasets.MNIST(root='./data/mnist', train=False, transform=transform, download=True)
            ]
        elif dataset_name == 'Cifar10':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
                transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
                transforms.ToTensor(),
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            self.datasets = [
                torchvision.datasets.CIFAR10(root='./data/cifar-10-python', train=True, download=True, transform=transform_train), #训练数据集
                torchvision.datasets.CIFAR10(root='./data/cifar-10-python', train=False, download=True, transform=transform_test)
            ]
        else:
            raise Exception('No such dataset called {}'.format(dataset_name))

    def get_loader(self, head=-1, rear=-1, batch_size=128, isTrain=True, isAdv=False):
        
        if isAdv == False:
            if head == -1 and rear == -1:    
                if isTrain:
                    return DataLoader(self.datasets[0], batch_size=batch_size)
                else:
                    return DataLoader(self.datasets[1], batch_size=batch_size)
            else:
                if isTrain:
                    self_sampler = SelfSampler(self.datasets[0], head=head, rear=rear)
                    return DataLoader(self.datasets[0], batch_size=batch_size, sampler=self_sampler)
                else:
                    self_samper = SelfSampler(self.datasets[1], head=head, rear=rear)
                    return DataLoader(self.datasets[0], batch_size=batch_size, sampler=self_sampler)
        else:
            adv_data = self.get_adv_samples()
            if head == -1 and rear == -1:    
                return DataLoader(adv_data, batch_size=batch_size)
            else:
                self_sampler = SelfSampler(adv_data, head=head, rear=rear)
                return DataLoader(adv_data, batch_size=batch_size, sampler=self_sampler)
      
    
    def get_adv_samples(self):
        
        path = os.path.join(self.default_path, self.dataset_name)
        if os.path.exists(path) == False:
            raise Exception('No such adv samples file path, please open the chosen is_save in training !')
        
        if os.path.exists(os.path.join(path, 'sample.pt')) == False:
            raise Exception('No such adv sample file path, please save adv samples first')

        adv_image, label = torch.load(os.path.join(path, 'sample.pt'))
        adv_data = TensorDataset(adv_image, label)

        return adv_data



    def test(self):

        data = self.datasets[0]
        print(data[0])
    

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    dataer = Dataer(dataset_name='Cifar10')
    
    train_loader = dataer.get_loader(batch_size=1, isTrain=False)

    for index, (image, label) in enumerate(train_loader):
        print(image.shape)
        print(label)
        if index == 100:
            break



