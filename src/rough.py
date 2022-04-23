
import numpy as np

def convolve3D(X, F, stride=1):
    [channel, width, height] = X.shape
    [channel, k, k, final_channel] = F.shape
    y = (width - k)//stride + 1
    x = (height - k)//stride + 1
    ans = np.zeros((final_channel, y, x))
    for i in range(final_channel):
        for j in range(0,y):
            for l in range(0,x):
                ans [i,j,l] = np.sum(X[:, j*stride:j*stride+k, l*stride:l*stride+k] * F[:,:,:,i])
    return ans

def convolve3D(X, F, stride=1):
    [c, w, h] = X.shape
    [c, k, k, d] = F.shape
    y = (w - k)//stride + 1
    x = (h - k)//stride + 1
    out = np.zeros((d, y, x))
    for i in range(d):
        for j in range(0,y):
            for l in range(0,x):
                out [i,j,l] = np.sum(X[:, jstride:jstride+k, lstride:lstride+k] * F[:,:,:,i])
    return out