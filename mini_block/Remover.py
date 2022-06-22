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
import torch.nn as nn
import os
from utils import paramters_to_vector, total_param
from functorch import jacrev, make_functional

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

        self.f_theta = [] # input for the last layer


        self.neter = Neter(dataer=self.dataer, args=self.args)
        self.neter.net_copy(self.basic_neter) # deepcopy the basic_net's model parameters, only need to [update_parameters, test] is ok.


        if os.path.exists(self.root_path) == False:
            os.makedirs(self.root_path)
        
        # construct the save matrix path
        if self.isDelta:  
            self.path = os.path.join(self.root_path, '{}_delta.pt'.format(self.remove_method))        
        else:
            self.path = os.path.join(self.root_path, '{}'.format(self.remove_method))

        if isDelta:
            self.basic_neter.save_adv_sample(isTrain=True)
        self.basic_neter.save_inner_output(isTrain=True, isAdv=True)

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
        


    def get_pure_hessain(self, head=-1, rear=-1):
        """
        the pure hessain is \sum \partial_ww for the sample list [head ,rear)
        detail:
        1) using the sum_loss
        2) using the torch.autograd.grad and unit_vec to do this.
        """
        def get_unit_vec(vec, index):
            
            if index !=0:
                vec[index - 1] = 0
            vec[index] = 1
            return vec

        loader = self.dataer.get_loader( # get the inner output loader for the last layer
            head=head, 
            rear=rear, 
            batch_size=self.dataer.train_data_lenth,  # using the total_size and sum_loss to get the pure_hessain
            isAdv=self.isDelta,
            isInner=True
        )

        classifier = self.neter.net.module.fc.to(self.basic_neter.device) #get the last layer
        params_number = total_param(classifier)
        unit_vec = torch.zeros(params_number).to(self.basic_neter.device)
        criterion = nn.CrossEntropyLoss(reduction='sum')
        
        for inner_out, label in loader:
            inner_out = inner_out.to(self.basic_neter.device)
            label = label.to(self.basic_neter.device)
            output = classifier(inner_out)
            loss = criterion(output, label)

            grad_w = paramters_to_vector(torch.autograd.grad(loss, classifier.parameters(), create_graph=True, retain_graph=True)[0])
            grad_ww = [paramters_to_vector(torch.autograd.grad(grad_w, classifier.parameters(), retain_graph=True, grad_outputs=get_unit_vec(unit_vec, index))[0]) for index in range(params_number)]
        
        grad_ww = torch.cat(grad_ww)

        return grad_ww

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

    def get_indirect_hessian(self, head=-1, rear=-1):

        loader = self.dataer.get_loader( # get the inner output loader for the last layer
            head=head, 
            rear=rear, 
            batch_size=1,  # using the total_size and sum_loss to get the pure_hessain
            isAdv=self.isDelta,
            isInner=True
        )
        classifier = self.neter.net.module.fc.to(self.basic_neter.device) #get the last layer
        params_number = total_param(classifier)

        def compute_loss()

    def Unlearning(self, head, rear):
        pass

class NewtonRemover(Remover):

    def __init__(self, basic_neter, dataer, isDelta, remove_method, args):
        super(NewtonRemover, self).__init__(basic_neter, dataer, isDelta, remove_method, args)


    def Unlearning(self, head, rear):
        pass

class InfluenceRemover(Remover):
    
    def __init__(self, basic_neter, dataer, isDelta, remove_method, args):
        
        super(InfluenceRemover, self).__init__(basic_neter, dataer, isDelta, remove_method, args)
    


    def Unlearning(self, head, rear): 
        pass

class FisherRemover(Remover):

    def __init__(self, basic_neter, dataer, isDelta, remove_method, args):
        
        super(FisherRemover, self).__init__(basic_neter, dataer, isDelta, remove_method, args)


    def Unlearning(self, head, rear):
        pass

            