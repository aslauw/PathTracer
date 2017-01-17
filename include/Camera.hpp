#ifndef CAMERA_HPP
#define CAMERA_HPP

#include <iostream>

#include <cstdint>
#include <ostream>
#include <cuda_runtime.h>

#include "SDLAssets.hpp"
#include "Vector.hpp"
#include "Ray.hpp"
#include "Scene.hpp"

#define MOVE_SPEED  0.5
#define ROT_SPEED   0.01
#define EPSILON     0.001
#define DEPTH       4

class Camera
{
public:
    __host__ __device__     Camera();
    __host__ __device__     Camera(const Vec3<float> position, const Vec3<float> forward, const Vec3<float> right);
    __host__ __device__     Camera(float px, float py, float pz, float fx, float fy, float fz, float rx, float ry, float rz);

    __host__ __device__ Vec3<float>     position()          { return _position; }
    __host__ __device__ Vec3<float>     forward()           { return _forward; }
    __host__ __device__ Vec3<float>     right()             { return _right; }
    __host__ __device__ uint32_t        width() const       { return _width; }
    __host__ __device__ uint32_t        height() const      { return _height; }
    __host__ __device__ float           fov() const         { return _fov; }
    __host__ __device__ float           ratio() const       { return _ratio; }
    __host__ __device__ Vec3<float>*    cache() const       { return _cache; }
    __host__ __device__ float           x() const           { return _position.x(); }
    __host__ __device__ float           y() const           { return _position.y(); }
    __host__ __device__ float           z() const           { return _position.z(); }
    __host__ __device__ float           fx() const          { return _forward.x(); }
    __host__ __device__ float           fy() const          { return _forward.y(); }
    __host__ __device__ float           fz() const          { return _forward.z(); }
    __host__ __device__ float           rx() const          { return _right.x(); }
    __host__ __device__ float           ry() const          { return _right.y(); }
    __host__ __device__ float           rz() const          { return _right.z(); }

    __host__ __device__ void    setPosition(const Vec3<float> position)     { _position = position; }
    __host__ __device__ void    setPosition(float x, float y, float z)      { _position = Vec3<float>(x, y, z); }
    __host__ __device__ void    setForward(const Vec3<float> forward)       { _forward = forward; }
    __host__ __device__ void    setForward(float x, float y, float z)       { _forward = Vec3<float>(x, y, z); }
    __host__ __device__ void    setRight(const Vec3<float> right)           { _right = right; }
    __host__ __device__ void    setRight(float x, float y, float z)         { _right = Vec3<float>(x, y, z); }
    __host__ __device__ void    setWidth(uint32_t width)                    { _width = width; _ratio = (float)_width / _height; }
    __host__ __device__ void    setHeight(uint32_t height)                  { _height = height; _ratio = (float)_width / _height; }
    __host__ __device__ void    setPlane(uint32_t width, uint32_t height)   { _width = width; _height = height; _ratio = (float)_width / _height; }
    __host__ __device__ void    setFov(float fov)                           { _fov = fov; }

    __host__ __device__ void            setCache();
    __host__ __device__ void            updatePosition(struct s_Keys& keys);
    __host__ __device__ Vec3<float>&    toWorld(Vec3<float>& v);
    __host__ __device__ void            moveForward();
    __host__ __device__ void            moveBack();
    __host__ __device__ void            moveLeft();
    __host__ __device__ void            moveRight();
    __host__ __device__ void            moveUp();
    __host__ __device__ void            moveDown();
    __host__ __device__ void            rotateUp();
    __host__ __device__ void            rotateDown();
    __host__ __device__ void            rotateLeft();
    __host__ __device__ void            rotateRight();
    __host__ __device__ void            barrelLeft();
    __host__ __device__ void            barrelRight();

    __host__ __device__                 ~Camera();

private:
    Vec3<float>     _position;
    Vec3<float>     _forward;     
    Vec3<float>     _right;
    uint32_t        _width;
    uint32_t        _height;
    float           _fov;
    float           _ratio;
    Vec3<float>*    _cache;
};

std::ostream&           operator<<(std::ostream& os, Camera& rhs);

#endif
