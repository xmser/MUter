import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
from TempArgs import args
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import MultipleLocator

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


def cg_solve(f_Ax, b, cg_iters=20, callback=None, verbose=False, residual_tol=1e-5, x_init=None):
    """
    Goal: Solve Ax=b equivalent to minimizing f(x) = 1/2 x^T A x - x^T b
    Assumption: A is PSD, no damping term is used here (must be damped externally in f_Ax)
    Algorithm template from wikipedia
    Verbose mode works only with numpy
    """
       
    if type(b) == torch.Tensor:
        x = torch.zeros(b.shape[0]) if x_init is None else x_init
        x = x.to(b.device)
        if b.dtype == torch.float16:
            x = x.half()
        r = b - f_Ax@x
        p = r.clone()
    elif type(b) == np.ndarray:
        x = np.zeros_like(b) if x_init is None else x_init
        r = b - f_Ax(x)
        p = r.copy()
    else:
        print("Type error in cg")

    fmtstr = "%10i %10.3g %10.3g %10.3g"
    titlestr = "%10s %10s %10s %10s"
    if verbose: print(titlestr % ("iter", "residual norm", "soln norm", "obj fn"))

    for i in range(cg_iters):
        if callback is not None:
            callback(x)
        if verbose:
            obj_fn = 0.5*x.dot(f_Ax(x)) - 0.5*b.dot(x)
            norm_x = torch.norm(x) if type(x) == torch.Tensor else np.linalg.norm(x)
            print(fmtstr % (i, r.dot(r), norm_x, obj_fn))

        rdotr = r.dot(r)
        Ap = f_Ax@p
        alpha = rdotr/(p.dot(Ap))
        x = x + alpha * p
        r = r - alpha * Ap
        newrdotr = r.dot(r)
        beta = newrdotr/rdotr
        p = r + beta * p

        if newrdotr < residual_tol:
            # print("Early CG termination because the residual was small")
            print('this is {}, i stoped !!!'.format(i))
            break

    if callback is not None:
        callback(x)
    if verbose: 
        obj_fn = 0.5*x.dot(f_Ax(x)) - 0.5*b.dot(x)
        norm_x = torch.norm(x) if type(x) == torch.Tensor else np.linalg.norm(x)
        print(fmtstr % (i, r.dot(r), norm_x, obj_fn))
    return x

def Transfrom_string(str, args):

    prefix = '{}_{}_'.format(args.adv_type, args.isBatchRemove)

    method_sequence = [
        'MUter',
        'Newton_delta',
        'Influence_delta',
        'Fisher_delta',
        'Newton',
        'Influence',
        'Fisher',
    ]

    for method in method_sequence:
        new_str = prefix + method
        if new_str == str:
            return method
    
    print('No match method !')
    return str

def Transform_to_dataframe(dicter, index_sequence, args):
    """transform the dicter type into dataframe for plot pic.

    Args:
        index_sequence: should be a list
        dicter (_type_): {remove_method: [x_1, x_2, x_3,..., x_n-1, x_n]}
        return : df[coloum_1: method, coloum_2: index, coloum_3: value]
        method: rempve way
        index: remove_number
        value: eval value
    """
    reTrans_dict = {
        'method': [],
        'index': [],
        'value': [],
    }

    for i, dex in enumerate(index_sequence):
        for key, value in dicter.items():
            reTrans_dict['method'].append(Transfrom_string(key, args))
            reTrans_dict['index'].append(dex)
            reTrans_dict['value'].append(value[i])
    
    return pd.DataFrame(reTrans_dict)

def get_random_sequence(total_lenth, resort_lenth, seed=None):

    if seed != None:
        random.seed(seed)
    
    resort_sequence = random.sample(range(0, total_lenth), resort_lenth)
    resort_sequence.sort()
    another_sequence = [i for i in range(total_lenth) if i not in resort_sequence]
    random_sequence = np.concatenate([resort_sequence, another_sequence])

    if len(random_sequence) != total_lenth:
        raise Exception('Random sequence error !')
    
    return list(random_sequence)


def generate_save_name(args, remain_head):
    str = ''
    
    if args.adv_type == 'FGSM':
        str += 'FGSM_'
    else:
        str += 'PGD_'
    
    if args.isBatchRemove == 0:
        str += 'Schur_'
    else:
        str += 'Batch_'
    
    str += 'model_ten_{}_times{}'.format(remain_head, args.times)
    print('The name is : {}'.format(str))
    return str

def line_plot(df, metrics):

    # set plot style
    sns.set_style('darkgrid')

    if metrics == 'distance':
        markers = ['o' for i in range(7)]
    else:
        markers = ['o' for i in range(7)]

    ax = sns.lineplot(
        x='index', 
        y='value', 
        data=df, 
        hue='method', 
        style='method',
        markers=markers,
        dashes=False,
    )

    plt.ylabel(metrics)
    plt.xlabel('Remove Numbers')


    return ax

if __name__ == "__main__":
    args = args()
    print(generate_save_name(args, 5000))