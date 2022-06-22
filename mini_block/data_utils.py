from ast import Break
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import Sampler


# TODO : the classtype loader for inner_output is not done !!!

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

class SubSampler(Sampler):

    def __init__(self, dataset, masked_id):
        """
        Args:
            masked_id (list): point what class to be omit.
        """
        
        lenth = int(len(dataset))
        self.indices = [index for index in range(lenth) if dataset.target not in masked_id]

    def __iter__(self):
        
        return iter(self.indices)

    def __len__(self):

        return len(self.indices)

class CustomSampler(Sampler):

    def __init__(self, sequence):

        self.indices = sequence

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
        
        self.class_num = {
            'Cifar10': 10,
            'Mnist': 10,
        }

        self.train_data_lenth = int(len(self.datasets[0]))
        self.test_data_lenth = int(len(self.datasets[1]))

    def get_loader(self, head=-1, rear=-1, batch_size=128, isTrain=True, isAdv=False, isInner=False, isClassType=False, isGetOne=False, id=[0, ]):

        if isInner:  # TODO need to be update for ClassType... 
            return self.get_inner_output_loader(batch_size=batch_size, head=head, rear=rear, isTrain=isTrain, isAdv=isAdv)
        else:
            if isClassType == False:
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
                            self_sampler = SelfSampler(self.datasets[1], head=head, rear=rear)
                            return DataLoader(self.datasets[0], batch_size=batch_size, sampler=self_sampler)
                else:
                    adv_data = self.get_adv_samples(isTrain=isTrain)
                    if head == -1 and rear == -1:    
                        return DataLoader(adv_data, batch_size=batch_size)
                    else:
                        self_sampler = SelfSampler(adv_data, head=head, rear=rear)
                        return DataLoader(adv_data, batch_size=batch_size, sampler=self_sampler)
            else:
                if isAdv:
                    adv_data = self.get_adv_samples()
                    if isGetOne:
                        return self.get_loader_ForOneClass(present_id=id, batch_size=batch_size, isTrain=isTrain, adv_data=adv_data, isAdv=isAdv)
                    else:
                        return self.get_loader_MaskOneClass(masked_id=id, batch_size=batch_size, isTrain=isTrain, adv_data=adv_data, isAdv=isAdv)
                else:
                    if isGetOne:
                        return self.get_loader_ForOneClass(present_id=id, batch_size=batch_size, isTrain=isTrain)
                    else:
                        return self.get_loader_MaskOneClass(masked_id=id, batch_size=batch_size, isTrain=isTrain)
    
    def get_inner_output_loader(self, batch_size=128, head=-1, rear=-1, isTrain=True, isAdv=False):

        if isAdv:
            adv_str = 'adv'
        else:
            adv_str = 'clean'
        
        if isTrain:
            sub_path = '{}_inner_output.pt'.format(adv_str)
        else:
            sub_path = 'test_{}_inner_output.pt'.format(adv_str)
        
        total_path = os.path.join(self.default_path, '{}'.format(self.dataset_name), sub_path)
        if os.path.exists(total_path) == False:
            raise Exception('No such inner output sample file path <{}>'.format(total_path))
        inner_output, label = torch.load(total_path)
        data = TensorDataset(inner_output, label)

        if head == -1 and rear == -1:
            return DataLoader(data, batch_size=batch_size, shuffle=False)
        else:
            self_sampler = SelfSampler(data, head=head, rear=rear)
            return DataLoader(data, batch_size=batch_size, sampler=self_sampler)

    def get_adv_samples(self, isTrain=True):
        
        path = os.path.join(self.default_path, self.dataset_name)
        if os.path.exists(path) == False:
            raise Exception('No such adv samples file path, please open the chosen is_save in training !')
        
        if isTrain:
            str = 'sample.pt'
        else:
            str = 'test_sample.pt'

        if os.path.exists(os.path.join(path, str)) == False:
            raise Exception('No such adv sample file path, please save adv samples first')

        adv_image, label = torch.load(os.path.join(path, str))
        adv_data = TensorDataset(adv_image, label)

        return adv_data

    def get_loader_MaskOneClass(self, masked_id=[0, ], batch_size=128, isTrain=True, adv_data=None, isAdv=False):
        """
        Args:
            masked_id (list, optional): be a list, point what class or classes to be omit.
        """

        if isAdv == False:
            if isTrain:
                sub_sampler = SubSampler(self.datasets[0], masked_id=masked_id)
                return DataLoader(self.datasets[0], batch_size=batch_size, sampler=sub_sampler)
            else:
                sub_sampler = SubSampler(self.datasets[1], masked_id=masked_id)
                return DataLoader(self.datasets[1], batch_size=batch_size, sampler=sub_sampler)
        else:
            sub_sampler = SubSampler(adv_data, masked_id=masked_id)
            return DataLoader(adv_data, batch_size=batch_size, sampler=sub_sampler)

    def get_loader_ForOneClass(self, present_id=[0, ], batch_size=128, isTrain=True, adv_data=None, isAdv=False):
        """
        Args:
            present_id (list, optional): point chose what class or classes to be present.
        """
        masked_id = [index for index in range(self.class_num[self.dataset_name]) if index not in present_id]
        
        if isAdv == False:
            if isTrain:
                sub_sampler = SubSampler(self.datasets[0], masked_id=masked_id)
                return DataLoader(self.datasets[0], batch_size=batch_size, sampler=sub_sampler)
            else:
                sub_sampler = SubSampler(self.datasets[1], masked_id=masked_id)
                return DataLoader(self.datasets[1], batch_size=batch_size, sampler=sub_sampler)
        else:
            sub_sampler = SubSampler(adv_data, masked_id=masked_id)
            return DataLoader(adv_data, batch_size=batch_size, sampler=sub_sampler)
    
    def get_customized_loader(self, sequence, batch_size=128, isTrain=True):
        
        custom_sampler = CustomSampler(sequence=sequence)    
        if isTrain:
            return DataLoader(self.datasets[0], batch_size=batch_size, sampler=custom_sampler)
        else:
            return DataLoader(self.datasets[1], batch_size=batch_size, sampler=custom_sampler)
            
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



