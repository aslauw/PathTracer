# PathTracer
*Note: Work in progress*

PathTracer is an attempt to create a real time GPU path tracing renderer, using [CUDA](http://www.nvidia.com/object/cuda_home_new.html).
It currently uses [SDL 2](https://www.libsdl.org/) to handle inputs and render window.
The README will be updated as new features are implemented.

## Installation
*Prerequisite :*
* Mac OS 10.9 or above
* [Xcode](https://developer.apple.com/xcode/) and Xcode command line tools
* [SDL 2 library](https://www.libsdl.org/)
* [CUDA LLVM Compiler (NVCC)](https://developer.nvidia.com/cuda-llvm-compiler)
Simply run **make**, and launch **./pathtracer**.

## Current features
* Full CUDA kernel computation
* Real time camera movement
* Direct illumination
* Basic primitives
* Diffuse/Specular shading
* Fresnel reflection/refraction

## Coming features
* Path tracing and global illumination (Color bleeding, etc)
* Data structure optimization
* Complex primitives / Meshes
* Texturing
* Antialiasing
* Depth of field
* Motion blur
* Caustics
* Animated scene

## Controls
* **W** - Move forward
* **A** - Move left
* **S** - Move backward
* **D** - Move right
* **Spacebar** - Move up
* **Left Shift** - Move down
* **Left Arrow** - Rotate camera left
* **Right Arrow** - Rotate camera right
* **Up Arrow** - Rotate camera up
* **Down Arrow** - Rotate camera down
* **K** - Decrease FOV
* **L** - Increase FOV
* **ESC** - Quit
