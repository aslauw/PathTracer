#include "Scene.hpp"

// Ctors
__host__ __device__
Scene::Scene()
{
    _objects = new Object*[MAX_NB_OBJECTS];
    _lights = new Light*[MAX_NB_LIGHTS];
    memset(_objects, 0, MAX_NB_OBJECTS * sizeof(Object*));
    memset(_lights, 0, MAX_NB_LIGHTS * sizeof(Light*));
    _objectsSize = 0;
    _lightsSize = 0;
}

// Functions

__host__ __device__ void
Scene::addObject(Object* object)
{
    object->setId(_objectsSize);
    if (_objectsSize < MAX_NB_OBJECTS)
    {
        _objects[_objectsSize] = object;
        _objectsSize++;
    }
}

__host__ __device__ void
Scene::addLight(Light* light)
{
    light->setId(_lightsSize);
    if (_lightsSize < MAX_NB_LIGHTS)
    {
        _lights[_lightsSize] = light;
        _lightsSize++;
    }
}

// Dtor
__host__ __device__
Scene::~Scene()
{
}
