#ifndef PLANE_HPP
#define PLANE_HPP

#include "Object.hpp"

class Plane : public Object
{
public:
    __host__ __device__ Plane(Vec3<float> origin, Vec3<float> n, uint8_t r = 255, uint8_t g = 255, uint8_t b = 255, uint8_t a = 255);
    __host__ __device__ Plane(float x, float y, float z, float nx, float ny, float nz, uint8_t r = 255, uint8_t g = 255, uint8_t b = 255, uint8_t a = 255);

    __host__ __device__ Vec3<float>     n()                 { return _n; }
    __host__ __device__ void            setN(Vec3<float> n) { _n = n; }

    __host__ __device__ float           intersect(Ray& ray);

    __host__ __device__ ~Plane();

private:
    Vec3<float> _n;
};


#endif
