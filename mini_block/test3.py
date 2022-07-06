import torch

arr = [torch.tensor([1., 1., 1.]), torch.tensor([2., 2., 2.]), torch.tensor([3., 3., 3.]), torch.tensor([4., 4., 4.])]

arr = torch.stack(arr, 0)

print(arr)
print(arr.sum(0))
