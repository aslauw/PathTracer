#include "Object.hpp"

// Ctors
__host__ __device__
Object::Object(ObjectType type, Vec3<float> origin, uint8_t r, uint8_t g, uint8_t b, uint8_t a)
: _type (type)
, _origin(origin)
, _diffuse(1.0)
, _specular(0.8)
, _shininess(5.0)
, _reflectance(0.0)
, _transmittance(0.0)
, _refractive(1.0)
{
    _color.r = r;
    _color.g = g;
    _color.b = b;
    _color.a = a;
}

__host__ __device__
Object::Object(ObjectType type, float x, float y, float z, uint8_t r, uint8_t g, uint8_t b, uint8_t a)
: _type (type)
, _origin(Vec3<float>(x, y, z))
, _diffuse(1.0)
, _specular(0.8)
, _shininess(5.0)
, _reflectance(0.0)
, _transmittance(0.0)
, _refractive(1.0)
{
    _color.r = r;
    _color.g = g;
    _color.b = b;
    _color.a = a;
}

// Dtor
__host__ __device__
Object::~Object()
{
}
