import torch

x = torch.arange(4.0)
x.requires_grad_(True)

# y.backward(gradient=torch.ones(len(y))) # since the output is a vector we have to specify this

x.grad.zero_() # sets the gradient to 0
y = x * x # element-wise product
u = y.detach() # we're making a copy of y and we're detaching u from the computational graph.
# the gradient won't flow in u, so it's like having requires_grad_(False)
z = u * x # since u is detached, the gradient will go directly in x and u will be considered a constant

z.sum().backward() # computes the sum of each element of z and initiates the backward pass.
# Backward works with complex functions too
print(x.grad) # in this specific case it's equal to u, because we've the derivative of x wrt x (1) times u
