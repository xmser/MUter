# from model.wrn import WideResNet
# import torch

# temp = 

# def hook(module, input, output):
#     temp = input[0]


# device = 'cuda'

# model = WideResNet(28, 1000, 10, 0.0).to(device)

# model = torch.nn.DataParallel(model, device_ids=list(range(1)))

# image = torch.randn((32, 3, 32, 32)).to(device)

# handle = model.module.fc.register_forward_hook(hook)

# output = model(image)

# print(output.shape)
# print(temp.shape)

# handle.remove()

# import torch
# import torch.nn as nn
# from functorch import make_functional

# fc = nn.Linear(100, 10)

# func, param = make_functional(fc)

# data = torch.randn((32, 100))

# output = func(param, data)
# print(param[0].shape)
# print(output.shape)


