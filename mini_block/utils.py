import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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


def cg_solve(f_Ax, b, cg_iters=1000, callback=None, verbose=False, residual_tol=1e-5, x_init=None):
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
            break

    if callback is not None:
        callback(x)
    if verbose: 
        obj_fn = 0.5*x.dot(f_Ax(x)) - 0.5*b.dot(x)
        norm_x = torch.norm(x) if type(x) == torch.Tensor else np.linalg.norm(x)
        print(fmtstr % (i, r.dot(r), norm_x, obj_fn))
    return x

