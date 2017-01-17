#include "Light.hpp"

// Ctors
__host__ __device__
Light::Light(Vec3<float> origin, uint8_t r, uint8_t g, uint8_t b, float intensity)
: _origin(origin)
{
    _color.r = r;
    _color.g = g;
    _color.b = b;
    _color.a = 255;
    _intensity = intensity;
}

__host__ __device__
Light::Light(float x, float y, float z, uint8_t r, uint8_t g, uint8_t b, float intensity)
: _origin(Vec3<float>(x, y, z))
{
    _color.r = r;
    _color.g = g;
    _color.b = b;
    _color.a = 255;
    _intensity = intensity;
}

// Dtor
__host__ __device__
Light::~Light()
{
}
