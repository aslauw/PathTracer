#ifndef RAY_HPP
#define RAY_HPP

#include <cuda_runtime.h>

#include "Vector.hpp"

class Ray
{
public:
    __host__ __device__ Ray();
    __host__ __device__ Ray(Vec3<float> origin, Vec3<float> direction);
    __host__ __device__ Ray(float x, float y, float z, float dx, float dy, float dz);

    __host__ __device__ Vec3<float>     origin()    { return _origin; }
    __host__ __device__ Vec3<float>     direction() { return _direction; }

    __host__ __device__ float           x()         { return _origin.x(); }
    __host__ __device__ float           y()         { return _origin.y(); }
    __host__ __device__ float           z()         { return _origin.z(); }
    __host__ __device__ float           dx()        { return _direction.x(); }
    __host__ __device__ float           dy()        { return _direction.y(); }
    __host__ __device__ float           dz()        { return _direction.z(); }

    __host__ __device__ void            setOrigin(Vec3<float> origin)               { _origin = origin; }
    __host__ __device__ void            setOrigin(float x, float y, float z)        { _origin = Vec3<float>(x, y, z); }
    __host__ __device__ void            setDirection(Vec3<float> direction)         { _direction = direction; }
    __host__ __device__ void            setDirection(float dx, float dy, float dz)  { _direction = Vec3<float>(dx, dy, dz); }

    __host__ __device__ float           norm()          { return _direction.norm(); }
    __host__ __device__ void            normalize()     { _direction.normalize(); }
    __host__ __device__ void            unit()          { _direction.unit(); }

    __host__ __device__ ~Ray();

private:
    Vec3<float> _origin;
    Vec3<float> _direction;
};

#endif
