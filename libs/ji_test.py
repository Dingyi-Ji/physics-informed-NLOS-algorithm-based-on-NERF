""" import os
import torch.fft as fft
import torch
import numpy as np
def ji_ifftshift(x, dim=(-2, -1)):
    """
    #Perform inverse FFT shift on the last two dimensions of a tensor.
"""
    num_dims = len(dim)
    shift = [(x.shape[d] + 1) // 2 for d in dim]
    return torch.roll(x, shift, dims=dim)

test_mat = np.random.rand(5,5)
#path = "/hy-tmp/physics to rescue/nlos3d/letters/data/"
#print(os.path.isfile(path))
print(test_mat)
test_tensor  = torch.tensor(test_mat)
a = fft.ifftshift(test_tensor, dim=(-2, -1))
print(a)
b = ji_ifftshift(test_tensor, dim=(-2, -1))
print(b)
#print(b.shape)
#print(b[:2,:2])

 """
import torch   

#print(torch.cuda.is_available())  

#print(torch.cuda.device(0))   

#print(torch.cuda.device_count())   

#print(torch.cuda.get_device_name(0))
input = torch.tensor([1, 2, 3])
result = torch.repeat_interleave(input, 0)
print(result)  # 输出: tensor([1, 1, 2, 2, 2, 3])

