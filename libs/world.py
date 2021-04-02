import torch
from functools import reduce
from libs.utils import FARAWAY, dev

class World:
    def __init__(self):
        self.objects = []

    def add(self, o):
        self.objects.append(o)

    def clear(self):
        self.objects = []

    def hit(self, r_in, t_min, t_max):
        intersections = [obj.intersect(r_in, t_min, t_max) for obj in self.objects] # hit should return t
        nearest = torch.min(torch.stack(intersections, dim=0), dim=0).values

        return intersections, nearest