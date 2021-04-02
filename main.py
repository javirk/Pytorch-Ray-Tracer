import torch
from libs.sphere import Sphere
from libs.world import World
from libs.camera import Camera
from libs.material import Material
from libs.utils import read_config

if __name__ == '__main__':
    dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    config = read_config('config.yml')

    material_ground = Material('lambertian', torch.tensor((0.5, 0.5, 0.5), device=dev))
    material_center = Material('lambertian', torch.tensor((0.1, 0.2, 0.5), device=dev))
    material_left = Material('lambertian', torch.tensor((224, 90, 90), device=dev))
    material_right = Material('metal', torch.tensor((250, 248, 202), device=dev))

    # World
    world = World()
    world.add(Sphere(torch.tensor((0.0, -1000., 0)), 1000., material_ground))
    world.add(Sphere(torch.tensor((0.0, 1.0, 0.)), 1, material_center))
    # world.add(Sphere(torch.tensor((-0.25, 0.5, 1.5)), 0.5, material_left))
    # world.add(Sphere(torch.tensor((-0.5, 0.5, -1.5)), 0.5, material_right))

    lookfrom = torch.tensor((-2, 1, 1.))
    lookat = torch.tensor((0., 0., 0.))
    vup = torch.tensor((0., 1., 0.))

    cam = Camera(lookfrom, lookat, vup, config['fov'], config['image_width'], config['aspect_ratio'], config['render_depth'])

    with torch.no_grad():
        image = cam.render(world, antialiasing=config['antialiasing'])

    image.show(flip=True)
    image.save('output/test_1.png', flip=True)