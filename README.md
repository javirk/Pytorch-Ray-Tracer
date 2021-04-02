# PyTorch Ray Tracer

This is a ray tracer built in PyTorch and based on [Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html).
Even if it uses GPU, it is not the most efficient ray tracer, as it computes a vast amount of information that is
disregarded later. It calculates all rays in parallel.

![test image](https://github.com/javirk/Pytorch-Ray-Tracer/blob/master/output/test.png?raw=true)

There is something wrong with the shadows, but it's enough for me at the moment. I will try to improve it in the future.

I haven't calculated how much of a speed improvement (if any) one can get using this ray tracer instead of the tutorial's version.
This is part of a bigger project, so I focused on having a working version and not delving into the details too much.
Also, dielectric materials are not implemented because they are not necessary for my use case, but their implementation
should be straightforward adding another method `scatter_dielectric()` in `libs/material.py`.
 
## How to make it work
Just change your parameters in `config.yml`. The number of spheres, their material (metal or lambertian) and position,
as well as the direction and position of the camera can be modified in `main.py`. I think the code is self-explanatory, 
but you can write an issue if it's not.
