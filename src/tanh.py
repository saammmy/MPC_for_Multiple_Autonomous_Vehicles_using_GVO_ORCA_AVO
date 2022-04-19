# import autograd.numpy as np
# from autograd import grad


# # Define a function like normal with Python and Numpy
# def tanh(x):
#     y = np.exp(-x)
#     print("hey")
#     return (1.0 - y) / (1.0 + y)
    
# # Create a function to compute the gradient
# grad_tanh = grad(tanh)

# # Evaluate the gradient at x = 1.0
# print(grad_tanh(1.0))

# import matplotlib.pyplot as plt
# from matplotlib.offsetbox import OffsetImage, AnnotationBbox
# from sympy import rotations
# from PIL import Image


# fig, ax = plt.subplots()
# ax.set_xlim([0, 100])
# ax.set_ylim([0, 100])
# img = Image.open('aventador_y.png')
# img = Image.open(None)
# img = img.rotate(180)
# # im = rotate_image_by_angle(im, get_the_rotation_angle_from_colume)
# aventador = AnnotationBbox(OffsetImage(img, zoom=.05),(50, 50), frameon=False)
# ax.add_artist(aventador)
# # aventador.remove()
# plt.scatter(50,30)
# # plt.xticks(range(10))
# # plt.yticks(range(10))
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

vec = np.random.uniform(0, 10, 50)
f = plt.figure(1)
ax = f.add_subplot(111)
ax.plot(vec, color='blue', marker='s', alpha=0.5)

plt.show()