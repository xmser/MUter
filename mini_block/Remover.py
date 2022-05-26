"""
where we abstract the remove way as class, the mainly fun here about such:
0)attribute: id(about what method I used to calculte the matrix and clean or perturbed?) is need for extend?? 
1)attribute: matrix or sub_matrix or layer_matrix(load and store the matrix, using his inverse with the grad to get the \delta_w)
2)attribute: clean_grad
3)attribute: perturbed_grad
4)method: load/save matrix
5)method: calculate the \delta_w
6)attribute: neter(manager the network), for neter(need to add function used to update paramter)
7)method: test model (get from neter)
"""
from Train import Neter
import torch
import os
from BlockHyper import BasicBlock

class Remover:

    def __init__(self, basic_neter, dataer, isDelta, remove_method, args):
        
        self.basic_neter = basic_neter ## the origin neter
        self.dataer = dataer
        self.isDelta = isDelta
        self.args = args
        self.remove_method = remove_method
        self.matrix = None
        self.grad = None
        self.delta_w = None
        self.path = None
        self.root_path = 'preMatrix/{}'.format(self.args.dataset)

        self.neter = Neter(dataer=dataer)
        self.neter.copy(basic_neter)


        if os.path.exists(self.root_path) == False:
            os.makedirs(self.root_path)
                
        self.path = os.path.join(self.path, '{}.pt'.format(self.remove_method))        

        self.init()  ## for matrix init

    def init(self):
        pass

    def Load_matrix(self, ):
        pass

    def Save_matrix(self, ):
        pass

    def Calculate_delta_w(self, ):
        pass

    def Unlearning(self, ):
        pass

class MUterRemover(Remover):

    def __init__(self, basic_neter, dataer, isDelta, remove_method, args):
        
        super(MUterRemover, self).__init__(basic_neter, dataer, isDelta, remove_method, args)
    
    def init(self):
        pass

    def Unlearning(self, head, rear):
        pass

class NewtonRemover(Remover):

    def __init__(self, basic_neter, dataer, isDelta, remove_method, args):
        super(NewtonRemover, self).__init__(basic_neter, dataer, isDelta, remove_method, args)
    
    def init(self):
        pass

    def Unlearning(self, head, rear):
        pass

class InfluenceRemover(Remover):
    
    def __init__(self, basic_neter, dataer, isDelta, remove_method, args):
        
        super(InfluenceRemover, self).__init__(basic_neter, dataer, isDelta, remove_method, args)
    
    def init(self):
        pass

    def Unlearning(self, head, rear): 
        pass

class FisherRemover(Remover):

    def __init__(self, basic_neter, dataer, isDelta, remove_method, args):
        
        super(FisherRemover, self).__init__(basic_neter, dataer, isDelta, remove_method, args)

    def init(self):
        pass

    def Unlearning(self, head, rear):
        pass

            