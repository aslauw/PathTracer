#include "Camera.hpp"
#include "Matrix.hpp"
#include "Sphere.hpp"
#include "Plane.hpp"

#include <cmath>
#include <iostream>

// Ctors
Camera::Camera()
{
    _position = Vec3<float>(0, 0, 0);
    _forward = Vec3<float>(0, 0, -1);
    _right = Vec3<float>(-1, 0, 0);
    _fov = (45.0 * M_PI / 180.0);
}

Camera::Camera(const Vec3<float> position, const Vec3<float> forward, const Vec3<float> right)
{
    _position = position;
    _forward = forward;
    _right = right;
}

Camera::Camera(float px, float py, float pz, float fx, float fy, float fz, float rx, float ry, float rz)
{
    _position = Vec3<float>(px, py, pz);
    _forward = Vec3<float>(fx, fy, fz);
    _right = Vec3<float>(rx, ry, rz);
}

// Functions
void
Camera::setCache()
{
    _cache = new Vec3<float>[_width * _height];
    for (uint32_t i = 0; i < _height; i++)
    {
        for (uint32_t j = 0; j < _width; j++)
        {
            float px = (2.0 * ((j + 0.5) / _width) - 1.0) * tan(_fov / 2.0) * _ratio; 
            float py = (1.0 - 2.0 * ((i + 0.5) / _height)) * tan(_fov / 2.0);
            Vec3<float> camPixel(px, py, 1.0);
            _cache[i * _width + j] = camPixel;
        }
    }
}

void
Camera::updatePosition(struct s_Keys& keys)
{
    // Camera Z-Axis translation
    if (keys.moveForward)
        moveForward();
    else if (keys.moveBack)
        moveBack();
    // Camera X-Axis translation
    if (keys.moveLeft)
        moveLeft();
    else if (keys.moveRight)
        moveRight();
    // Camera Y-Axis translation
    if (keys.moveUp)
        moveUp();
    else if (keys.moveDown)
        moveDown();
    
    // Camera X-Axis rotation
    if (keys.rotateUp)
        rotateUp();
    else if (keys.rotateDown)
        rotateDown();
    // Camera Y-Axis rotation
    if (keys.rotateLeft)
        rotateLeft();
    else if (keys.rotateRight)
        rotateRight();
    // Camera Z-Axis rotation
    if (keys.barrelLeft)
        barrelLeft();
    else if (keys.barrelRight)
        barrelRight();
    
    // Fov updatePosition
    if (keys.fovUp)
        _fov = fmin(_fov + 0.01, 150.0 * M_PI / 180.0);
    else if (keys.fovDown)
        _fov = fmax(_fov - 0.01, 20.0 * M_PI / 180.0);
}

Vec3<float>&
Camera::toWorld(Vec3<float>& v)
{
    Vec3<float> l;
    l = _right;
    l *= -1;
    Matrix4<float>  mat(l, _forward.cross(_right), _forward, _position);

    v = mat * v;
    return v;
}

void
Camera::moveForward()
{
    Vec3<float> scaled(_forward);

    scaled *= MOVE_SPEED;
    _position = _position + scaled;
}

void
Camera::moveBack()
{
    Vec3<float> scaled(_forward);

    scaled *= -MOVE_SPEED;
    _position = _position + scaled;
}

void
Camera::moveLeft()
{
    Vec3<float> scaled(_right);

    scaled *= MOVE_SPEED;
    _position = _position + scaled;
}

void
Camera::moveRight()
{
    Vec3<float> scaled(_right);

    scaled *= -MOVE_SPEED;
    _position = _position + scaled;
}

void
Camera::moveUp()
{
    Vec3<float> scaled(0.0, 1.0, 0.0);

    scaled *= MOVE_SPEED;
    _position = _position + scaled;
}

void
Camera::moveDown()
{
    Vec3<float> scaled(0.0, 1.0, 0.0);

    scaled *= -MOVE_SPEED;
    _position = _position + scaled;
}

void
Camera::rotateUp()
{
   Vec3<float>  q;
   double costheta, sintheta;

   costheta = cos(-M_PI * ROT_SPEED);
   sintheta = sin(-M_PI * ROT_SPEED);

   q.setX(q.x() + (costheta + (1 - costheta) * _right.x() * _right.x()) * _forward.x());
   q.setX(q.x() + ((1 - costheta) * _right.x() * _right.y() - _right.z() * sintheta) * _forward.y());
   q.setX(q.x() + ((1 - costheta) * _right.x() * _right.z() + _right.y() * sintheta) * _forward.z());

   q.setY(q.y() + ((1 - costheta) * _right.x() * _right.y() + _right.z() * sintheta) * _forward.x());
   q.setY(q.y() + (costheta + (1 - costheta) * _right.y() * _right.y()) * _forward.y());
   q.setY(q.y() + ((1 - costheta) * _right.y() * _right.z() - _right.x() * sintheta) * _forward.z());

   q.setZ(q.z() + ((1 - costheta) * _right.x() * _right.z() - _right.y() * sintheta) * _forward.x());
   q.setZ(q.z() + ((1 - costheta) * _right.y() * _right.z() + _right.x() * sintheta) * _forward.y());
   q.setZ(q.z() + (costheta + (1 - costheta) * _right.z() * _right.z()) * _forward.z());

   _forward = q;
}

void
Camera::rotateDown()
{
    Vec3<float>  q;
   double costheta, sintheta;

   costheta = cos(M_PI * ROT_SPEED);
   sintheta = sin(M_PI * ROT_SPEED);

   q.setX(q.x() + (costheta + (1 - costheta) * _right.x() * _right.x()) * _forward.x());
   q.setX(q.x() + ((1 - costheta) * _right.x() * _right.y() - _right.z() * sintheta) * _forward.y());
   q.setX(q.x() + ((1 - costheta) * _right.x() * _right.z() + _right.y() * sintheta) * _forward.z());

   q.setY(q.y() + ((1 - costheta) * _right.x() * _right.y() + _right.z() * sintheta) * _forward.x());
   q.setY(q.y() + (costheta + (1 - costheta) * _right.y() * _right.y()) * _forward.y());
   q.setY(q.y() + ((1 - costheta) * _right.y() * _right.z() - _right.x() * sintheta) * _forward.z());

   q.setZ(q.z() + ((1 - costheta) * _right.x() * _right.z() - _right.y() * sintheta) * _forward.x());
   q.setZ(q.z() + ((1 - costheta) * _right.y() * _right.z() + _right.x() * sintheta) * _forward.y());
   q.setZ(q.z() + (costheta + (1 - costheta) * _right.z() * _right.z()) * _forward.z());

   _forward = q;
}

void
Camera::rotateLeft()
{
    Matrix4<float>  mat;
    mat[0] = cos(M_PI * ROT_SPEED);
    mat[2] = sin(M_PI * ROT_SPEED);
    mat[8] = -sin(M_PI * ROT_SPEED);
    mat[10] = cos(M_PI * ROT_SPEED);

    _forward = mat * _forward;
    _right = mat * _right;
}

void
Camera::rotateRight()
{
    Matrix4<float>  mat;
    mat[0] = cos(-M_PI * ROT_SPEED);
    mat[2] = sin(-M_PI * ROT_SPEED);
    mat[8] = -sin(-M_PI * ROT_SPEED);
    mat[10] = cos(-M_PI * ROT_SPEED);

    _forward = mat * _forward;
    _right = mat * _right;
}

void
Camera::barrelLeft()
{
}

void
Camera::barrelRight()
{
}

// Otors
std::ostream&
operator<<(std::ostream& os, Camera& rhs)
{
    os << "Camera: " << std::endl;
    os << "\tposition (W): " << rhs.x() << " " << rhs.y() << " " << rhs.z() << std::endl;
    os << "\tforward  (W): " << rhs.fx() << " " << rhs.fy() << " " << rhs.fz() << std::endl;
    os << "\tright    (W): " << rhs.rx() << " " << rhs.ry() << " " << rhs.rz() << std::endl;
    os << "\tfov         : " << rhs.fov() << std::endl;
    os << "\tratio       : " << rhs.ratio() << std::endl;
    return os;
}

// Dtor
Camera::~Camera()
{
    delete[] _cache;
}
