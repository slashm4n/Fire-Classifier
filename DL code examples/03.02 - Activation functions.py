import torch
from matplotlib import pyplot as plt

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True) # creates the input vector
y = torch.relu(x) # y = torch.sigmoid(x) # y = torch.tanh(x) # the activation function
plt.plot(x.detach().numpy(), y.detach().numpy()), plt.xlabel("x"), plt.ylabel("relu of x")
# we use detach and numpy because matplotlib works with numpy arrays and not pytorch tensors
plt.show()


y.backward(torch.ones_like(x)) # backward pass
plt.plot(x.detach().numpy(), x.grad.numpy()), plt.xlabel("x"), plt.ylabel("grad of relu of x")
plt.show()