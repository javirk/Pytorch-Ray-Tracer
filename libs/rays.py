import torch
from libs.utils import dev, FARAWAY
from einops import repeat
import libs.utils as u


class Rays:
    def __init__(self, origin, directions):
        if len(origin) == 1:
            self.origin = repeat(origin, 'c -> copy c', copy=directions.shape[0]).to(dev)
        else:
            self.origin = origin.to(dev)

        self.directions = directions.float().to(dev)

    def n_rays(self):
        return self.directions.shape[0]

    def x(self):
        return self.directions[:, 0]

    def y(self):
        return self.directions[:, 1]

    def z(self):
        return self.directions[:, 2]

    def point_at(self, t):
        return self.origin + self.directions * t

    def squared_length(self):
        return torch.einsum('...i,...i', self.directions, self.directions).unsqueeze(1)

    def length(self):
        return torch.sqrt(self.squared_length())

    def unit_direction(self):
        unit_directions = self.directions / self.length()
        return Rays(self.origin, unit_directions)

    def trace(self, world, depth):
        t_max = u.repeat_value_tensor(FARAWAY, self.n_rays(), device=dev)
        intersections, nearest = world.hit(self, 0.001, t_max)  # Self because we are throwing this ray
        color = torch.zeros_like(self.directions, device=dev)

        for obj, distances in zip(world.objects, intersections):
            hit = (nearest != FARAWAY) & (distances == nearest)
            if hit.any():
                color_obj = obj.get_color(self, distances, world, depth)
                color = color_obj * hit + color * (~hit)

        hit_anything = (nearest != FARAWAY)

        unit = self.unit_direction()
        t = 0.5 * (unit.y() + 1.0).unsqueeze(1)
        color1 = torch.ones((t.shape[0], 3), device=dev)
        color2 = (torch.zeros((t.shape[0], 3)) + torch.tensor((0.5, 0.7, 1.0))).to(dev)
        c2 = (1.0 - t) * color1 + t * color2

        # return color * hit_anything + c2 * (~hit_anything)
        return torch.where(hit_anything, color, c2)
