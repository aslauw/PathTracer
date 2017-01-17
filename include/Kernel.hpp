#ifndef KERNEL_HPP
#define KERNEL_HPP

#include <cuda_runtime.h>

#include "Camera.hpp"
#include "Scene.hpp"
#include "Matrix.hpp"
#include "Ray.hpp"
#include "Object.hpp"
#include "Sphere.hpp"
#include "Plane.hpp"
#include "Light.hpp"

#define MAX_RAY_QUEUE_SIZE  1024

typedef struct  s_QRay
{
    Ray             ray;        // Ray
    uint8_t         depth;      // Ray depth
    float           k;          // Fading coefficient (reflectance, transmittance, ...)
}               QRay;


/*
** Templated ray tracing function.
**
** Templating forces compiler to create different versions of rayTrace allowing recursion-like calls,
** thus avoiding memory allocation of a ray queue.
*/
template <int depth>
__device__ Color    rayTrace(Ray ray, Object** objects, uint64_t objectsSize, Light** lights, uint64_t lightsSize)
{
    Color           color;
    Object*         hitObject;
    float           t, distance;

    color.v = BLACK_VISIBLE;

    // Find intersection
    t = INFINITY;
    distance = INFINITY;
    for (size_t i = 0; i < objectsSize; i++)
    {
        if (objects[i]->type() == ObjectType::Sphere)
        {
            t = static_cast<Sphere*>(objects[i])->intersect(ray);
        }
        else if (objects[i]->type() == ObjectType::Plane)
        {
            t = static_cast<Plane*>(objects[i])->intersect(ray);
        }

        if (t < distance)
        {
            distance = t;
            hitObject = objects[i];
        }
    }
    if (distance == INFINITY)
        return color;

    // Compute shading
    Vec3<float>     rayOrigin(ray.origin());
    Vec3<float>     hitPoint(rayOrigin + (ray.direction() * distance));
    Vec3<float>     toLight;
    float           distToLight;
    Ray             shadowRay;
    bool            shadow;
    float           dot;
    float           k;

    // Get normal
    Vec3<float>     normal;
    if (hitObject->type() == ObjectType::Sphere)
    {
        normal = hitPoint - hitObject->origin();
        normal /= static_cast<Sphere*>(hitObject)->r();
    }
    // Plane
    else if (hitObject->type() == ObjectType::Plane)
    {
        normal = static_cast<Plane*>(hitObject)->n();
        Vec3<float> i = hitPoint - rayOrigin;
        if (i.dot(normal) > 0)
            normal = normal * -1.0;
    }
    // Avoid self intersections
    hitPoint += normal * EPSILON;

    // Loop through every light
    for (uint64_t i = 0; i < lightsSize; i++)
    {
        toLight = lights[i]->origin() - hitPoint;
        distToLight = toLight.norm();
        toLight /= distToLight;
        shadowRay.setOrigin(hitPoint);
        shadowRay.setDirection(toLight);

        // Cast shadow ray
        shadow = false;
        for (uint64_t j = 0; j < objectsSize; j++)
        {
            float shadowT;
            if (objects[j]->type() == ObjectType::Sphere)
            {
                shadowT = static_cast<Sphere*>(objects[j])->intersect(shadowRay);
            }
            else if (objects[j]->type() == ObjectType::Plane)
            {
                shadowT = static_cast<Plane*>(objects[j])->intersect(shadowRay);
            }
            if (shadowT != INFINITY)
            {
                if (shadowT < distToLight)
                {
                    shadow = true;
                    break;
                }
            }
        }

        if (!shadow)
        {
            Color hc;
            Color lc;
            // Compute diffuse
            if (hitObject->diffuse() > 0)
            {
                dot = normal.dot(toLight);
                if (dot > 0)
                {
                    k = lights[i]->intensity() * 10.0 / distToLight * hitObject->diffuse();
                    hc = hitObject->color();
                    lc = lights[i]->color();
                    color.r = fmin(255, color.r + dot * k * fmax((float)0, (float)hc.r - (255 - lc.r)));
                    color.g = fmin(255, color.g + dot * k * fmax((float)0, (float)hc.g - (255 - lc.g)));
                    color.b = fmin(255, color.b + dot * k * fmax((float)0, (float)hc.b - (255 - lc.b)));
                }
            }
            // Compute specular
            if (hitObject->specular() > 0)
            {
                Vec3<float> view(hitPoint - rayOrigin);
                view.normalize();
                Vec3<float> half(view + toLight);

                dot = normal.dot(half);
                if (dot > 0)
                {
                    dot = powf(dot, hitObject->shininess());

                    k = lights[i]->intensity() * 10.0 / distToLight * hitObject->specular();
                    hc = hitObject->color();
                    lc = lights[i]->color();
                    color.r = fmin(255, color.r + dot * k * fmax((float)0, (float)hc.r - (255 - lc.r)));
                    color.g = fmin(255, color.g + dot * k * fmax((float)0, (float)hc.g - (255 - lc.g)));
                    color.b = fmin(255, color.b + dot * k * fmax((float)0, (float)hc.b - (255 - lc.b)));
                }
            }
        }
    }

    // Compute reflection
    if (hitObject->reflectance() > 0)
    {
        Vec3<float> view;
        Vec3<float> n;
        float       dot;

        view = hitPoint - rayOrigin;
        view.normalize();
        n = normal;
        dot = -normal.dot(view);
        n *= 2 * dot;

        Ray     reflectedRay(hitPoint, view + n);
        Color   reflectedColor;
        reflectedColor = rayTrace<depth + 1>(reflectedRay, objects, objectsSize, lights, lightsSize);
        color.r = fmin(255, color.r + hitObject->reflectance() * reflectedColor.r);
        color.g = fmin(255, color.g + hitObject->reflectance() * reflectedColor.g);
        color.b = fmin(255, color.b + hitObject->reflectance() * reflectedColor.b);
    }

    // Compute refraction
    if (hitObject->transmittance() > 0)
    {
        if (hitObject->type() == ObjectType::Plane)
        {
            Ray     refractedRay(hitPoint, ray.direction());
            Color   refractedColor;
            refractedColor = rayTrace<depth + 1>(refractedRay, objects, objectsSize, lights, lightsSize);
            color.r = fmin(255, color.r + hitObject->transmittance() * refractedColor.r);
            color.g = fmin(255, color.g + hitObject->transmittance() * refractedColor.g);
            color.b = fmin(255, color.b + hitObject->transmittance() * refractedColor.b);
        }
        else
        {
            Vec3<float> view;
            Vec3<float> n;
            float       c1, c2, n1, n2, nr;

            view = hitPoint - rayOrigin;
            view.normalize();
            // Entering medium
            if (view.dot(normal) < 0)
            {
                hitPoint -= normal * 2.0 * EPSILON;
                n1 = 1.0;
                n2 = hitObject->refractive();
                nr = n1 / n2;
                n = normal;
            }
            // Leaving medium
            else
            {
                n1 = hitObject->refractive();
                n2 = 1.0;
                nr = n1 / n2;
                n = normal * -1.0;
            }
            c1 = -n.dot(view);
            c2 = sqrtf(1.0 - nr * nr * (1.0 - c1 * c1));
            Vec3<float> refractedDir = (view * nr) + n * (nr * c1 - c2);

            Ray     refractedRay(hitPoint, refractedDir);
            Color   refractedColor;
            refractedColor = rayTrace<depth + 1>(refractedRay, objects, objectsSize, lights, lightsSize);
            color.r = fmin(255, color.r + hitObject->transmittance() * refractedColor.r);
            color.g = fmin(255, color.g + hitObject->transmittance() * refractedColor.g);
            color.b = fmin(255, color.b + hitObject->transmittance() * refractedColor.b);
        }
    }

    return color;
}

/*
** Maximum depth rayTrace version
*/
template <>
__device__ Color
rayTrace<DEPTH>(Ray ray, Object** objects, uint64_t objectsSize, Light** lights, uint64_t lightsSize)
{
    Color color;

    color.v = BLACK_VISIBLE;
    return color;
}

/*
** Main rendering kernel
*/
__global__ void     renderKernel(Camera* camera, Scene* scene, uint32_t* pixels)
{
    int pi, pj;

    // Get pixel corresponding to thread
    pi = blockIdx.y * blockDim.y + threadIdx.y;
    pj = blockIdx.x * blockDim.x + threadIdx.x;

    // Get pixel in camera coordinates
    float px = (2.0 * ((pj + 0.5) / camera->width()) - 1.0) * tan(camera->fov() / 2.0) * camera->ratio();
    float py = (1.0 - 2.0 * ((pi + 0.5) / camera->height())) * tan(camera->fov() / 2.0);
    Vec3<float> pixel(px, py, 1.0);

    // Pixel from camera to world coordinates
    Vec3<float> orig = camera->position();
    Vec3<float> l;
    l = camera->right();
    l *= -1;
    Matrix4<float> mat(l, camera->forward().cross(camera->right()), camera->forward(), camera->position());
    pixel = mat * pixel;

    // Setup ray
    Ray ray(orig, pixel - orig);
    ray.normalize();

    // Raytrace
    Object**    objects = scene->objects();
    uint64_t    objectsSize = scene->objectsSize();;
    Light**     lights = scene->lights();
    uint64_t    lightsSize = scene->lightsSize();
    Color       color;

    color = rayTrace<0>(ray, objects, objectsSize, lights, lightsSize);

    // Set pixel color
    pixels[pi * camera->width() + pj] = color.v;
}

#endif
