import torch
from libs.rays import Rays
from libs.image import Image
from libs.utils import dev, degrees_to_radians, unit_vector, convert_to_float
import matplotlib.pyplot as plt
from math import tan


class Camera:
    def __init__(self, lookfrom, lookat, vup, vfov, image_width, aspect_ratio, render_depth):
        aspect_ratio = convert_to_float(aspect_ratio)
        theta = degrees_to_radians(vfov)
        h = tan(theta / 2)
        viewport_height = 2.0 * h
        viewport_width = aspect_ratio * viewport_height

        w = unit_vector(lookfrom - lookat, dim=0)
        u = unit_vector(torch.cross(vup, w), dim=0)
        v = torch.cross(w, u)

        self.origin = lookfrom
        self.horizontal = viewport_width * u
        self.vertical = viewport_height * v
        self.lower_left_corner = self.origin - self.horizontal / 2 - self.vertical / 2 - w

        self.image_width = image_width
        self.image_height = int(self.image_width // aspect_ratio)
        self.out_shape = (self.image_height, self.image_width, 3)

        self.render_depth = render_depth


    def render(self, world, antialiasing=1):
        colors = torch.zeros((self.image_width * self.image_height, 3), device=dev)
        for _ in range(antialiasing):
            x = torch.tile(torch.linspace(0, (self.out_shape[1] - 1) / self.out_shape[1], self.out_shape[1]),
                           (self.out_shape[0],)).unsqueeze(1)
            y = torch.repeat_interleave(torch.linspace(0, (self.out_shape[0] - 1) / self.out_shape[0], self.out_shape[0]),
                                        self.out_shape[1]).unsqueeze(1)
            if antialiasing != 1:
                x += torch.rand(x.shape) / self.out_shape[1]
                y += torch.rand(y.shape) / self.out_shape[0]

            ray = Rays(origin=self.origin, directions=self.lower_left_corner + x * self.horizontal + y * self.vertical - self.origin)
            colors += ray.trace(world, self.render_depth)

        scale = 1 / antialiasing
        colors = torch.sqrt(scale * colors)
        return Image.from_flat(colors, self.image_width, self.image_height)


