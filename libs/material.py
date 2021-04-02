from libs.utils import random_on_unit_sphere_like, dot, unit_vector
from libs.rays import Rays

class Material:
    def __init__(self, mat, attenuation):
        self.albedo = attenuation
        self.material = mat

        if self.material == 'lambertian':
            self.scatter = self.scatter_lambertian
        elif self.material == 'metal':
            self.scatter = self.scatter_metal
        else:
            raise ValueError(self.material + ' unknown material')

    def scatter_lambertian(self, normal, p, **kwargs):
        scatter_direction = normal + random_on_unit_sphere_like(p)
        ray = Rays(p, scatter_direction)
        return ray

    def scatter_metal(self, normal, p, **kwargs):
        r_in = kwargs['r_in']
        reflected = self._reflect(unit_vector(r_in.directions, dim=1), normal)
        # reflected = torch.where(dot(reflected, normal) > 0, reflected, torch.zeros_like(reflected))
        ray = Rays(p, reflected)
        # hit_world = (hit_world & dot(reflected, normal) > 0)

        return ray

    @staticmethod
    def _reflect(v, n):
        return v - 2 * dot(v, n) * n