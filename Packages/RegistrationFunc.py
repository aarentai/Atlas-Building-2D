import numpy as np
import torch
import torch.nn.functional as F
from IPython.core.debugger import set_trace
import matplotlib.pyplot as plt


def get_div(v):
    v_x = (torch.roll(v[0], shifts=(0, -1), dims=(0, 1)) - torch.roll(v[0], shifts=(0, 1), dims=(0, 1)))/2
    v_y = (torch.roll(v[1], shifts=(-1, 0), dims=(0, 1)) - torch.roll(v[1], shifts=(1, 0), dims=(0, 1)))/2
    return v_x + v_y


def get_det(jac_m): # size_h x size_w
    return jac_m[0, 0] * jac_m[1, 1] - jac_m[0, 1] * jac_m[1, 0] #


def get_jacobian_determinant(diffeo): # diffeo: 2 x size_h x size_w
    jac_m = get_jacobian_matrix(diffeo) # jac_m: 2 x 2 x size_h x size_w
    return jac_m[0, 0] * jac_m[1, 1] - jac_m[0, 1] * jac_m[1, 0] # size_h x size_w


def get_jacobian_matrix(diffeo): # diffeo: 2 x size_h x size_w
#     return torch.stack((get_gradient(diffeo[1]), get_gradient(diffeo[0])))
    return torch.stack((get_gradient(diffeo[0]), get_gradient(diffeo[1])))


def get_gradient(F):  # 2D F: size_h x size_w
    F_padded = torch.zeros((F.shape[0]+2,F.shape[1]+2))
    F_padded[1:-1,1:-1] = F
    F_padded[0,:] = F_padded[1,:]
    F_padded[-1,:] = F_padded[-2,:]
    F_padded[:,0] = F_padded[:,1]
    F_padded[:,-1] = F_padded[:,-2]
    F_x = (torch.roll(F_padded, shifts=(0, -1), dims=(0, 1)) - torch.roll(F_padded, shifts=(0, 1), dims=(0, 1)))/2
    F_y = (torch.roll(F_padded, shifts=(-1, 0), dims=(0, 1)) - torch.roll(F_padded, shifts=(1, 0), dims=(0, 1)))/2
    return torch.stack((F_x[1:-1,1:-1].type(torch.DoubleTensor), F_y[1:-1,1:-1].type(torch.DoubleTensor)))
#     F_x = (torch.roll(F, shifts=(0, -1), dims=(0, 1)) - torch.roll(F, shifts=(0, 1), dims=(0, 1)))/2
#     F_y = (torch.roll(F, shifts=(-1, 0), dims=(0, 1)) - torch.roll(F, shifts=(1, 0), dims=(0, 1)))/2
#     return torch.stack((F_x, F_y))


# get the identity mapping
def get_idty(size_h, size_w): 
    HH, WW = torch.meshgrid([torch.arange(size_h, dtype=torch.double), torch.arange(size_w, dtype=torch.double)])
#     return torch.stack((HH, WW))
    return torch.stack((WW, HH))


# my interpolation function
def compose_function(f, diffeo, mode='periodic'):  # f: N x m x n  diffeo: 2 x m x n
    
    f = f.permute(f.dim()-2, f.dim()-1, *range(f.dim()-2))  # change the size of f to m x n x ...
    
    size_h, size_w = f.shape[:2]
#     Ind_diffeo = torch.stack((torch.floor(diffeo[0]).long()%size_h, torch.floor(diffeo[1]).long()%size_w))
    Ind_diffeo = torch.stack((torch.floor(diffeo[1]).long()%size_h, torch.floor(diffeo[0]).long()%size_w))

    F = torch.zeros(size_h+1, size_w+1, *f.shape[2:], dtype=torch.double)
    
    if mode=='border':
        F[:size_h,:size_w] = f
        F[-1, :size_w] = f[-1]
        F[:size_h, -1] = f[:, -1]
        F[-1, -1] = f[-1,-1]
    elif mode =='periodic':
        # extend the function values periodically (1234 1)
        F[:size_h,:size_w] = f
        F[-1, :size_w] = f[0]
        F[:size_h, -1] = f[:, 0]
        F[-1, -1] = f[0,0]
    
    # use the bilinear interpolation method
    F00 = F[Ind_diffeo[0], Ind_diffeo[1]].permute(*range(2, f.dim()), 0, 1)  # change the size to ...*m*n
    F01 = F[Ind_diffeo[0], Ind_diffeo[1]+1].permute(*range(2, f.dim()), 0, 1)
    F10 = F[Ind_diffeo[0]+1, Ind_diffeo[1]].permute(*range(2, f.dim()), 0, 1)
    F11 = F[Ind_diffeo[0]+1, Ind_diffeo[1]+1].permute(*range(2, f.dim()), 0, 1)

#     C = diffeo[0] - Ind_diffeo[0].type(torch.DoubleTensor)
#     D = diffeo[1] - Ind_diffeo[1].type(torch.DoubleTensor)
    C = diffeo[0] - Ind_diffeo[1].type(torch.DoubleTensor)
    D = diffeo[1] - Ind_diffeo[0].type(torch.DoubleTensor)

    F0 = F00 + (F01 - F00)*C
    F1 = F10 + (F11 - F10)*C
    return F0 + (F1 - F0)*D
#     return (1-D)*(1-C)*F00+C*(1-D)*F01+D*(1-C)*F10+C*D*F11


def plot_diffeo(diffeo, title=None, step_size=1, show_axis=False):
    diffeo = diffeo.detach().numpy()
    import matplotlib.pyplot as plt
    
    plt.figure(num=None, figsize=(5,5), dpi=100, facecolor='w', edgecolor='k')
    if show_axis is False:
        plt.axis('off')
    ax = plt.gca()
    ax.set_aspect('equal')
    for h in range(0, diffeo.shape[1], step_size):
#         plt.plot(diffeo[1, h, :], diffeo[0, h, :], 'b', linewidth=0.5)
        plt.plot(diffeo[0, h, :], diffeo[1, h, :], 'b', linewidth=0.5)
    for w in range(0, diffeo.shape[2], step_size):
#         plt.plot(diffeo[1, :, w], diffeo[0, :, w], 'b', linewidth=0.5)
        plt.plot(diffeo[0, :, w], diffeo[1, :, w], 'b', linewidth=0.5)
        
    if(title):
        plt.title(title)
    plt.show()
    
    
def plotImage(I0, I1):
    import matplotlib.pyplot as plt
    fig = plt.figure(num=None, figsize=(6,3), dpi=100, facecolor='w', edgecolor='k')
    a1 = fig.add_subplot(1, 2, 1)
    imgplot = plt.imshow(I0)
    a1.set_title('I0')
    a2 = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(I1)
    a2.set_title('I1')

    
def plotImg(I):
    import matplotlib.pyplot as plt
    fig = plt.figure(num=None, figsize=(6,3), dpi=100, facecolor='w', edgecolor='k')
    img = plt.imshow(I)
    plt.colorbar(img,fraction=0.046, pad=0.04)