#include "Plane.hpp"

// Ctors
__host__ __device__
Plane::Plane(Vec3<float> origin, Vec3<float> n, uint8_t r, uint8_t g, uint8_t b, uint8_t a)
: Object(ObjectType::Plane, origin, r, g, b, a)
, _n(n)
{
}

__host__ __device__
Plane::Plane(float x, float y, float z, float nx, float ny, float nz, uint8_t r, uint8_t g, uint8_t b, uint8_t a)
: Object(ObjectType::Plane, x, y, z, r, g, b, a)
, _n(Vec3<float>(nx, ny, nz))
{
}

// Functions
__host__ __device__ float
Plane::intersect(Ray& ray)
{
    float       dot;
    float       t;
    Vec3<float> rayToPlane;

    dot = ray.direction().dot(_n);
    // If parallel, no intersection
    if (fabs(dot) <= 0.0001)
        return INFINITY;

    rayToPlane = _origin - ray.origin();
    t = rayToPlane.dot(_n) / dot;

    // Intersection
    if (t > 0)
        return t;

    // No intersection
    return INFINITY;
}

// Dtor
__host__ __device__
Plane::~Plane()
{
}
