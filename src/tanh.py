import autograd.numpy as np
from autograd import grad


# Define a function like normal with Python and Numpy
def tanh(x):
    y = np.exp(-x)
    print("hey")
    return (1.0 - y) / (1.0 + y)
    
# Create a function to compute the gradient
grad_tanh = grad(tanh)

# Evaluate the gradient at x = 1.0
print(grad_tanh(1.0))