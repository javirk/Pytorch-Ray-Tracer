import torch
import libs.utils as u
from libs.utils import dev


class Sphere:
    def __init__(self, center, radius, material):
        super().__init__()
        self.center = center.to(dev)
        self.radius = radius
        self.material = material

    def intersect(self, r, t_min, t_max):
        oc = r.origin - self.center
        a = r.squared_length()
        half_b = u.dot(oc, r.directions)
        c = u.dot(oc, oc) - self.radius * self.radius
        discriminant = half_b * half_b - a * c

        sqrtd = torch.sqrt(torch.maximum(torch.zeros_like(discriminant), discriminant))
        h0 = (-half_b - sqrtd) / a
        h1 = (-half_b + sqrtd) / a

        # TODO: There should be t_min there. But let's wait for that
        h = torch.where((h0 > 0) & (h0 < h1), h0, h1)
        pred = (discriminant > 0) & (h > 0)
        root = torch.where(pred, h, t_max)
        return root

    def get_color(self, r_in, t, world, depth=1):
        color = torch.zeros_like(r_in.directions, device=dev)
        p = r_in.point_at(t)
        outward_normal = (p - self.center) / self.radius
        r_scattered = self.material.scatter(outward_normal, p, r_in=r_in)
        if depth > 0:
            color += self.material.albedo * r_scattered.trace(world, depth - 1)

        return color

