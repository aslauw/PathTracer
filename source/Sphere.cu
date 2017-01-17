#include "Sphere.hpp"
#include <cmath>
#include <iostream>

// Ctors
__host__ __device__
Sphere::Sphere(Vec3<float> origin, float rad, uint8_t r, uint8_t g, uint8_t b, uint8_t a)
: Object(ObjectType::Sphere, origin, r, g, b, a)
, _r(rad)
{
}

__host__ __device__
Sphere::Sphere(float x, float y, float z, float rad, uint8_t r, uint8_t g, uint8_t b, uint8_t a)
: Object(ObjectType::Sphere, x, y, z, r, g, b, a)
, _r(rad)
{
}

// Functions
__host__ __device__ float
Sphere::intersect(Ray& ray)
{
    float ox, oy, oz;
    float dx, dy, dz;
    float cx, cy, cz;
    float cox, coy, coz;
    ox = ray.origin().x();
    oy = ray.origin().y();
    oz = ray.origin().z();
    dx = ray.direction().x();
    dy = ray.direction().y();
    dz = ray.direction().z();
    cx = _origin.x();
    cy = _origin.y();
    cz = _origin.z();
    cox = ox - cx;
    coy = oy - cy;
    coz = oz - cz;

    float b, c;
    b = 2.0 * (dx * cox + dy * coy + dz * coz);
    c = (cox * cox + coy * coy + coz * coz) - _r * _r;

    float delta;
    delta = b * b - 4.0 * c;
    if (delta > 0.0)
    {
        float t1, t2;
        float sqrt_d = sqrt(delta);
        t1 = (-b - sqrt_d) / 2.0;
        t2 = (-b + sqrt_d) / 2.0;
        if (t1 < 0.0 && t2 < 0.0)
            return INFINITY;
        if (t1 < 0.0)
            return t2;
        if (t2 < 0.0)
            return t1;
        return (t1 < t2) ? t1 : t2;
    }
    else if (delta == 0.0)
    {
        float t;
        t = -b / 2.0;
        if (t > 0.0)
            return t;
        return INFINITY;
    }
    else
    {
        return INFINITY;
    }
}

// Dtor
__host__ __device__
Sphere::~Sphere()
{
}
