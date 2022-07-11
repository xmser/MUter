from builtins import print
import torch

x = torch.eye(3).reshape(1, 3, 3)
batch_x = x.repeat(5, 1, 1).cuda()
y = torch.randn(5, 3, 3).cuda()
print(batch_x)
print(y)
print(batch_x-y)