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
from json import load
from Train import Neter
import torch
import os
from BlockHyper import BasicBlock

class Remover:
    """
    the basic scheme is by pretrain arch, args.isTuning == True, in this way, we code the code.
    """

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
        self.root_path = 'data/preMatrix/{}'.format(self.args.dataset)


        self.neter = Neter(dataer=self.dataer, args=self.args)
        self.neter.net_copy(self.basic_neter) # deepcopy the basic_net's model parameters, only need to [update_parameters, test] is ok.


        if os.path.exists(self.root_path) == False:
            os.makedirs(self.root_path)
        
        # construct the save matrix path
        if self.isDelta:  
            self.path = os.path.join(self.root_path, '{}_delta.pt'.format(self.remove_method))        
        else:
            self.path = os.path.join(self.root_path, '{}'.format(self.remove_method))

    def init(self):
        pass

    def Load_matrix(self, ):
        
        if os.path.exists(self.path) == False:
            raise Exception('No such path for load matrix <{}>'.format(self.path))
        print('Loading matrix from <{}>'.format(self.path))
        self.matrix = torch,load(self.path)
        print('load done !')

    def Save_matrix(self, ):
        
        if self.matrix == None:
            raise Exception('No init the pre save matrix !')
        if self.isDelta:
            print('saving the {}_delta matrix to the path <{}> ...'.format(self.remove_method, self.path))
        else:
            print('saving the {} matrix to the path <{}> ...'.format(self.remove_method, self.path))
        
        torch.save(self.matrix, f=self.path)
        print('save done !')
        
    def Calculate_delta_w(self, ):
        pass

    def Unlearning(self, ):
        pass

class MUterRemover(Remover):
    """
    MUter using the remove function \delta_w = (\partial_ww - \partial_wx.\partial_xx^{-1}.\partial_xw)^{-1}.g
    method : init() to calculate the sum of total_hessain. For \partial_ww, we use the sum loss for samples to get, for \p_xx, \p_xw(wx),
    we use the vmap, jaccre from functorch to do this. difficulty: the \p_xx and \p_xw need to be replace by \partial_f_{\theta}f_{\theta} and so on to do this.
    """

    def __init__(self, basic_neter, dataer, isDelta, remove_method, args):
        
        super(MUterRemover, self).__init__(basic_neter, dataer, isDelta, remove_method, args)
        self.init()
    
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

            