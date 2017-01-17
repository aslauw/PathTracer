#ifndef VECTOR_HPP
#define VECTOR_HPP

#include <cmath>
#include <cuda_runtime.h>

template <class T>
class Vec3
{
public:
    // Ctors
    __host__ __device__ Vec3<T>() : _x(0), _y(0), _z(0), _unit(false) {}
    __host__ __device__ Vec3<T>(T x, T y, T z, bool unit = false): _x(x), _y(y), _z(z), _unit(unit) {}
    __host__ __device__ Vec3<T>(const T& rhs) { *this = rhs; }

    // Geters
    __host__ __device__ T       x() const   { return _x; }
    __host__ __device__ T       y() const   { return _y; }
    __host__ __device__ T       z() const   { return _z; }

    __host__ __device__ bool    unit()
    {
        if (_unit)
            return true;
        if (squaredNorm() == 1)
        {
            _unit = true;
            return true;
        }
        return false;
    }

    // Seters
    __host__ __device__ void    setX(T x)   { _x = x; _unit = false; }
    __host__ __device__ void    setY(T y)   { _y = y; _unit = false; }
    __host__ __device__ void    setZ(T z)   { _z = z; _unit = false; }

    // Otors
    __host__ __device__ Vec3<T>&    operator=(const Vec3<T> rhs)
    {
        _x = rhs._x;
        _y = rhs._y;
        _z = rhs._z;
        _unit = rhs._unit;
        return *this;
    }
    // Vector/Vector operations
    __host__ __device__ Vec3<T>     operator+(const Vec3<T>& rhs)
    {
        return Vec3<T>(_x + rhs._x, _y + rhs._y, _z + rhs._z);
    }
    __host__ __device__ Vec3<T>     operator-(const Vec3<T>& rhs)
    {
        return Vec3<T>(_x - rhs._x, _y - rhs._y, _z - rhs._z);
    }
    __host__ __device__ Vec3<T>     operator*(const Vec3<T>& rhs)
    {
        return Vec3<T>(_x * rhs._x, _y * rhs._y, _z * rhs._z);
    }
    __host__ __device__ Vec3<T>     operator/(const Vec3<T>& rhs)
    {
        return Vec3<T>(_x / rhs._x, _y / rhs._y, _z / rhs._z);
    }

    // Vector scalar operations
    __host__ __device__ Vec3<T>     operator+(float s)
    {
        return Vec3<T>(_x + s, _y + s, _z + s);
    }
    __host__ __device__ Vec3<T>     operator-(float s)
    {
        return Vec3<T>(_x - s, _y - s, _z - s);
    }
    __host__ __device__ Vec3<T>     operator*(float s)
    {
        return Vec3<T>(_x * s, _y * s, _z * s);
    }
    __host__ __device__ Vec3<T>     operator/(float s)
    {
        return Vec3<T>(_x / s, _y / s, _z / s);
    }

    // Vector/Vector operations
    __host__ __device__ Vec3<T>&    operator+=(const Vec3<T>& v)
    {
        _x += v._x;
        _y += v._y;
        _z += v._z;
        _unit = false;
        return *this;
    }
    __host__ __device__ Vec3<T>&    operator-=(const Vec3<T>& v)
    {
        _x -= v._x;
        _y -= v._y;
        _z -= v._z;
        _unit = false;
        return *this;
    }
    __host__ __device__ Vec3<T>&    operator*=(const Vec3<T>& v)
    {
        _x *= v._x;
        _y *= v._y;
        _z *= v._z;
        _unit = false;
        return *this;
    }
    __host__ __device__ Vec3<T>&    operator/=(const Vec3<T>& v)
    {
        _x /= v._x;
        _y /= v._y;
        _z /= v._z;
        _unit = false;
        return *this;
    }

    // Vector/Scalar operations
    __host__ __device__ Vec3<T>&    operator+=(const T v)
    {
        _x += v;
        _y += v;
        _z += v;
        _unit = false;
        return *this;
    }
    __host__ __device__ Vec3<T>&    operator-=(const T v)
    {
        _x -= v;
        _y -= v;
        _z -= v;
        _unit = false;
        return *this;
    }
    __host__ __device__ Vec3<T>&    operator*=(const T v)
    {
        _x *= v;
        _y *= v;
        _z *= v;
        _unit = false;
        return *this;
    }
    __host__ __device__ Vec3<T>&    operator/=(const T v)
    {
        _x /= v;
        _y /= v;
        _z /= v;
        _unit = false;
        return *this;
    }

    // Functions
    __host__ __device__ T       norm()              { return ((_unit) ? 1 : sqrt(_x * _x + _y * _y + _z * _z)); }
    __host__ __device__ T       squaredNorm()       { return ((_unit) ? 1 : (_x * _x + _y * _y + _z * _z)); }

    __host__ __device__ void    normalize()
    {
        if (_unit)
            return;
        T norm = this->norm();
        _x /= norm;
        _y /= norm;
        _z /= norm;
        _unit = true;
    }

    __host__ __device__ float   dot(const Vec3<T>& rhs)
    {
        return (_x * rhs._x + _y * rhs._y + _z * rhs._z);
    }

    __host__ __device__ Vec3<T> cross(const Vec3<T>& rhs)
    {
        return Vec3<T>(_y * rhs._z - _z * rhs._y, _z * rhs._x - _x * rhs._z, _x * rhs._y - _y * rhs._x);
    }

    // Dtor
    __host__ __device__ ~Vec3<T>() {}

private:
    T       _x;
    T       _y;
    T       _z;
    bool    _unit;
};

#endif
