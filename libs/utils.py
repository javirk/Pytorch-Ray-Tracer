import torch
from math import pi
from einops import rearrange
import yaml
import matplotlib.pyplot as plt

FARAWAY = 1.0e3
dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def dot(a, b):
    '''
    Dot product between two vectors
    :return: torch tensor
    '''
    return torch.einsum('...i,...i', a, b).unsqueeze(-1)

def unit_vector(a, dim=None):
    '''
    Returns the unit vector with the direction of a
    :param dim:
    :param a:
    :return:
    '''
    return a / a.norm(dim=dim).unsqueeze(-1)

def degrees_to_radians(d):
    return d * pi / 180.

def repeat_value_tensor(val, reps, device='cpu'):
    if type(val) != torch.Tensor:
        return torch.tile(torch.tensor(val, device=device), (reps, 1))
    else:
        return torch.tile(val, reps)

def random_in_range(a, b, size):
    '''
    Random float tensor in range [a, b)
    :param a: minimum value
    :param b: maximum value
    :param size: size of the tensor
    :return: tensor
    '''
    return (a - b) * torch.rand(size, device=dev) + b

def random_on_unit_sphere(size):
    # We use the method in https://stats.stackexchange.com/questions/7977/how-to-generate-uniformly-distributed-points-on-the-surface-of-the-3-d-unit-sphe
    # to produce vectors on the surface of a unit sphere

    x = torch.randn(size)
    l = torch.sqrt(torch.sum(torch.pow(x, 2), dim=-1)).unsqueeze(1)
    x = (x/l).to(dev)

    return x

def unit_sphere(size):
    v = torch.randn(size)
    v = v / v.norm(2, dim=-1, keepdim=True)
    return v.to(dev)

def random_on_unit_sphere_like(t):
    return unit_sphere(t.shape)

def plot_t(t, width=400, height=225):
    assert width * height == t.shape[0], 'Dimensions mismatch'
    data = rearrange(t, '(h w) c -> h w c', w=width, h=height)
    plt.imshow(data.cpu())

def plot_dir(r):
    if type(r) != torch.Tensor:
        r = r.directions
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    r = r.cpu().numpy()
    ax.scatter(r[:, 0], r[:, 1], r[:, 2])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

def read_config(path):
    with open(path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        return data

def convert_to_float(frac_str):
    try:
        return float(frac_str)
    except ValueError:
        num, denom = frac_str.split('/')
        try:
            leading, num = num.split(' ')
            whole = float(leading)
        except ValueError:
            whole = 0
        frac = float(num) / float(denom)
        return whole - frac if whole < 0 else whole + frac