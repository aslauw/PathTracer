#ifndef OBJECT_HPP
#define OBJECT_HPP

#include <cstdint>

#include "Ray.hpp"
#include "SDLAssets.hpp"

enum class ObjectType
{
    Sphere = 0,
    Plane
};

class Object
{
public:
    __host__ __device__ Object(ObjectType type, Vec3<float> origin, uint8_t r = 255, uint8_t g = 255, uint8_t b = 255, uint8_t a = 255);
    __host__ __device__ Object(ObjectType type, float x, float y, float z, uint8_t r = 255, uint8_t g = 255, uint8_t b = 255, uint8_t a = 255);

    __host__ __device__ uint64_t        id()                            { return _id; }
    __host__ __device__ ObjectType      type()                          { return _type; }
    __host__ __device__ Vec3<float>     origin()                        { return _origin; }
    __host__ __device__ float           diffuse()                       { return _diffuse; }
    __host__ __device__ float           specular()                      { return _specular; }
    __host__ __device__ float           shininess()                     { return _shininess; }
    __host__ __device__ float           reflectance()                   { return _reflectance; }
    __host__ __device__ float           transmittance()                 { return _transmittance; }
    __host__ __device__ float           refractive()                    { return _refractive; }
    __host__ __device__ Color           color()                         { return _color; }

    __host__ __device__ void            setId(uint64_t id)                      { _id = id; }
    __host__ __device__ void            setOrigin(Vec3<float> origin)           { _origin = origin; }
    __host__ __device__ void            setDiffuse(float diffuse)               { _diffuse = diffuse; }
    __host__ __device__ void            setSpecular(float specular)             { _specular = specular; }
    __host__ __device__ void            setShininess(float shininess)           { _shininess = shininess; }
    __host__ __device__ void            setReflectance(float reflectance)       { _reflectance = reflectance; }
    __host__ __device__ void            setTransmittance(float transmittance)   { _transmittance = transmittance; }
    __host__ __device__ void            setRefractive(float refractive)         { _refractive = refractive; }
    __host__ __device__ void            setColor(Color color)                   { _color = color; }

    __host__ __device__ ~Object();

protected:
    uint64_t        _id;
    ObjectType      _type;
    Vec3<float>     _origin;
    float           _diffuse;
    float           _specular;
    float           _shininess;
    float           _reflectance;
    float           _transmittance;
    float           _refractive;
    Color           _color;
};

#endif
