import torch
import matplotlib.pyplot as plt
import copy as copy

torch.manual_seed(0)
x = torch.rand([1,1,10],dtype=torch.float64)
conv = torch.nn.ConvTranspose1d(1,1, kernel_size=7)
conv.bias = torch.nn.parameter.Parameter(0*torch.Tensor(conv.bias).to(torch.float64))
conv.weight = torch.nn.parameter.Parameter(torch.Tensor(conv.weight).to(torch.float64))

y = torch.zeros([16], dtype=torch.float64)
for i in range(10):
    y[i:i+7] += torch.squeeze(conv(x[:,:,i:i+1]))

y1 = torch.squeeze(conv(x)) # 16
y21 = torch.squeeze(conv(x[:,:,:5])) # 11
y22 = torch.squeeze(conv(x[:,:,5:])) # 11
y22[:6] += y21[-6:]
#y22[:6] -= conv.bias

y2 = torch.cat([y21[:-6], y22]) # 16

"""
y = torch.zeros(96)
for i in range(96):
    y[i] = conv(x[:,:,i:i+5])

y1 = torch.squeeze(conv(x))
"""
plt.figure(); 
plt.plot((y1-y).detach().numpy()); 
plt.show()
k=3