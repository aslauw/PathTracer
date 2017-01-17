#ifndef SCENE_HPP
#define SCENE_HPP

#include <vector>

#include "Object.hpp"
#include "Sphere.hpp"
#include "Plane.hpp"
#include "Light.hpp"

#define MAX_NB_OBJECTS  64
#define MAX_NB_LIGHTS   32

class Scene
{
public:
    __host__ __device__ Scene();

    __host__ __device__ Object**    objects()       { return _objects; }
    __host__ __device__ uint64_t    objectsSize()   { return _objectsSize; }
    __host__ __device__ Light**     lights()        { return _lights; }
    __host__ __device__ uint64_t    lightsSize()    { return _lightsSize; }

    __host__ __device__ void        addObject(Object* object);
    __host__ __device__ void        addLight(Light* light);

    __host__ __device__ void        setObjects(Object** objects)    { _objects = objects; }
    __host__ __device__ void        setObjectsSize(uint64_t size)   { _objectsSize = size; }
    __host__ __device__ void        setLights(Light** lights)       { _lights = lights; }
    __host__ __device__ void        setLightsSize(uint64_t size)    { _lightsSize = size; }

    __host__ __device__ ~Scene();

private:
    Object**                _objects;
    uint64_t                _objectsSize;
    Light**                 _lights;
    uint64_t                _lightsSize;
};

#endif
