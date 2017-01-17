#include "Ray.hpp"

// Ctors
__host__ __device__ Ray::Ray()
: _origin(Vec3<float>(0, 0, 0))
, _direction(Vec3<float>(0, 0, -1))
{
}

__host__ __device__ Ray::Ray(Vec3<float> origin, Vec3<float> direction)
: _origin(origin)
, _direction(direction)
{
}

__host__ __device__ Ray::Ray(float x, float y, float z, float dx, float dy, float dz)
: _origin(Vec3<float>(x, y, z))
, _direction(Vec3<float>(dx, dy, dz))
{
}

// Dtor
__host__ __device__ Ray::~Ray()
{
}
