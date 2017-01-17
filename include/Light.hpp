#ifndef LIGHT_HPP
#define LIGHT_HPP

#include <cstdint>

#include "SDLAssets.hpp"
#include "Vector.hpp"

class Light
{
public:
    __host__ __device__ Light(Vec3<float> origin, uint8_t r = 255, uint8_t g = 255, uint8_t b = 255, float intensity = 1.0);
    __host__ __device__ Light(float x, float y, float z, uint8_t r = 255, uint8_t g = 255, uint8_t b = 255, float intensity = 1.0);

    __host__ __device__ uint64_t        id()            { return _id; }
    __host__ __device__ Vec3<float>     origin()        { return _origin; }
    __host__ __device__ Color&          color()         { return _color; }
    __host__ __device__ float           intensity()     { return _intensity; }

    __host__ __device__ void            setId(uint64_t id)              { _id = id; }
    __host__ __device__ void            setOrigin(Vec3<float> origin)   { _origin = origin; }
    __host__ __device__ void            setColor(Color color)           { _color = color; }
    __host__ __device__ void            setIntensity(float intensity)   { _intensity = intensity; }

    __host__ __device__ ~Light();

private:
    uint64_t        _id;
    Vec3<float>     _origin;
    Color           _color;
    float           _intensity;
};

#endif
