from model import vgg16
import torch
from tqdm import tqdm

from data_utils import Dataer
from kfac_utils import ComputeCovA, ComputeCovG

CovAHandler = ComputeCovA()
CovGHandler = ComputeCovG()

inps = []
out_grads = []

aa = []
gg = []

def input_hooker(module, input):
    print(input[0].data.shape)
    aa.append(CovAHandler(input[0].data, module))
    inps.append(input[0].data.cpu())

def grad_output_hooker(module, grad_input, grad_output):
    print(grad_output[0].data.shape)
    gg.append(CovGHandler(grad_output[0].data, module, batch_averaged=True))
    out_grads.append(grad_output[0].data.cpu())

param_dict = {'Linear', 'Conv2d'}

device = 'cuda'
model = vgg16().to(device)

for module in model.modules():

    module_name = module.__class__.__name__

    if module_name in param_dict:
        module.register_forward_pre_hook(input_hooker)
        module.register_backward_hook(grad_output_hooker)


dataer = Dataer('Cifar10')

train_loader = dataer.get_loader(batch_size=128, isTrain=True)

criterion = torch.nn.CrossEntropyLoss()

for (image, label) in train_loader:

    image = image.to(device)
    label = label.to(device)

    output = model(image)

    loss = criterion(output, label)

    loss.backward()
    break

print()
print()
print()

for item in aa:
    print(item.shape)
print('='*100)
for item in gg:
    print(item.shape)    