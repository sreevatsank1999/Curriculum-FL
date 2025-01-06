import torch

from math import sqrt

class ToCUDA(object):
    def __init__(self, device=torch.device("cpu")):
        self.device = device;

    def __call__(self, tensor):
        return tensor.to(self.device);

    def __repr__(self):
        return self.__class__.__name__ + '(device={0})'.format(self.device)