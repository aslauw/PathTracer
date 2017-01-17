#ifndef SPHERE_HPP
#define SPHERE_HPP

#include "Object.hpp"

class Sphere : public Object
{
public:
    __host__ __device__ Sphere(Vec3<float> origin, float rad, uint8_t r = 255, uint8_t g = 255, uint8_t b = 255, uint8_t a = 255);
    __host__ __device__ Sphere(float x, float y, float z, float rad, uint8_t r = 255, uint8_t g = 255, uint8_t b = 255, uint8_t a = 255);

    __host__ __device__ float           r()             { return _r; }
    __host__ __device__ void            setR(float r)   { _r = r; }

    __host__ __device__ float           intersect(Ray& ray);

    __host__ __device__ ~Sphere();

private:
    float   _r;
};

#endif
