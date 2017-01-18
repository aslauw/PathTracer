#include <iostream>

#include "Window.hpp"
#include "Camera.hpp"
#include "Sphere.hpp"
#include "Plane.hpp"
#include "Scene.hpp"
#include "Kernel.hpp"

#define RETURN_SUCCESS 0
#define RETURN_FAILURE 1

#define WINDOW_WIDTH 1024
#define WINDOW_HEIGHT 768

int
main(int ac, char** av)
{
    (void)ac;
    (void)av;

    // Create window
    Window      window(WINDOW_WIDTH, WINDOW_HEIGHT);
    if (!window.valid())
    {
        return RETURN_FAILURE;
    }

    // Create camera
    Camera      camera;
    camera.setPosition(0.0, 0.0, 15.0);
    camera.setPlane(WINDOW_WIDTH, WINDOW_HEIGHT);
    camera.setCache();

    // Create Scene
    Scene       scene;
    // Walls
    Plane*      backWall = new Plane(0.0, 0.0, -20.0, 0.0, 0.0, 1.0);
    backWall->setDiffuse(0.5);
    backWall->setSpecular(0.4);
    // backWall->setReflectance(0.5);
    Plane*      floorWall = new Plane(0.0, -5.0, 0.0, 0.0, 1.0, 0.0, 50, 50, 50);
    floorWall->setDiffuse(0.9);
    floorWall->setSpecular(0.1);
    floorWall->setReflectance(0.05);
    Plane*      ceilWall = new Plane(0.0, 10.0, 0.0, 0.0, -1.0, 0.0);
    Plane*      leftWall = new Plane(-10.0, 0.0, 0.0, 1.0, 0.0, 0.0, 200, 75, 75);
    Plane*      rightWall = new Plane(10.0, 0.0, 0.0, -1.0, 0.0, 0.0, 100, 100, 200);
    // Spheres
    // Reflective sphere
    Sphere*     leftSphere = new Sphere(-5.0, -2.0, -15.0, 3.0);
    leftSphere->setDiffuse(0.0);
    leftSphere->setSpecular(0.4);
    leftSphere->setReflectance(1.0);
    // Transmittive sphere
    Sphere*     rightSphere = new Sphere(3.0, -2.0, -8.0, 3.0, 200, 200, 200);
    rightSphere->setDiffuse(0.2);
    rightSphere->setSpecular(0.0);
    rightSphere->setTransmittance(0.8);
    rightSphere->setRefractive(1.4);
    // Lights
    Light*      ceilLight = new Light(0.0, 8.0, -15.0);
    Light*      backLight = new Light(0.0, 8.0, 10.0);
    // Add items to scene
    scene.addObject(backWall);
    scene.addObject(floorWall);
    scene.addObject(ceilWall);
    scene.addObject(leftWall);
    scene.addObject(rightWall);
    scene.addObject(leftSphere);
    scene.addObject(rightSphere);
    scene.addLight(ceilLight);
    scene.addLight(backLight);

    // Cuda device variables initialization
    Camera*                 d_camera;
    uint32_t*               d_pixels;
    cudaMalloc(&d_camera, sizeof(Camera));
    cudaMalloc(&d_pixels, WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(uint32_t));
    cudaMemcpy(d_pixels, window.pixels(), WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Copy scene from host to device
    Scene*                  d_scene;
    Object**                d_objects;
    Light**                 d_lights;
    cudaMalloc(&d_scene, sizeof(Scene));
    cudaMalloc(&d_objects, MAX_NB_OBJECTS * sizeof(Object*));
    cudaMalloc(&d_lights, MAX_NB_LIGHTS * sizeof(Light*));

    Object**                objects;
    Light**                 lights;
    objects = new Object*[MAX_NB_OBJECTS];
    lights = new Light*[MAX_NB_LIGHTS];
    // Copy objects
    for (uint64_t i = 0; i < scene.objectsSize(); i++)
    {
        if (scene.objects()[i]->type() == ObjectType::Sphere)
        {
            cudaMalloc(&objects[i], sizeof(Sphere));
            cudaMemcpy(objects[i], scene.objects()[i], sizeof(Sphere), cudaMemcpyHostToDevice);
        }
        else if (scene.objects()[i]->type() == ObjectType::Plane)
        {
            cudaMalloc(&objects[i], sizeof(Plane));
            cudaMemcpy(objects[i], scene.objects()[i], sizeof(Plane), cudaMemcpyHostToDevice);
        }
    }
    cudaMemcpy(d_objects, objects, MAX_NB_OBJECTS * sizeof(Object*), cudaMemcpyHostToDevice);
    // Copy lights
    for (uint64_t i = 0; i < scene.lightsSize(); i++)
    {
        cudaMalloc(&lights[i], sizeof(Light));
        cudaMemcpy(lights[i], scene.lights()[i], sizeof(Light), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(d_lights, lights, MAX_NB_LIGHTS * sizeof(Light*), cudaMemcpyHostToDevice);
    for (uint64_t i = 0; i < scene.objectsSize(); i++)
    {
        delete scene.objects()[i];
    }
    for (uint64_t i = 0; i < scene.lightsSize(); i++)
    {
        delete scene.lights()[i];
    }
    delete[] scene.objects();
    delete[] scene.lights();
    scene.setObjects(d_objects);
    scene.setLights(d_lights);
    cudaMemcpy(d_scene, &scene, sizeof(scene), cudaMemcpyHostToDevice);

    // Main loop
    while (!window.over())
    {
        // Handle updates
        window.pollEvents();

        // Upate camera position
        window.update(camera);
        cudaMemcpy(d_camera, &camera, sizeof(Camera), cudaMemcpyHostToDevice);

        // Rendering kernel
        dim3 threadsPerBlock(16, 16);
        dim3 grid(WINDOW_WIDTH / threadsPerBlock.x, WINDOW_HEIGHT / threadsPerBlock.y);
        renderKernel<<<grid, threadsPerBlock>>>(d_camera, d_scene, d_pixels);

        // Copy pixels from device to renderer
        cudaMemcpy(window.pixels(), d_pixels, WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        window.render();
    }

    // Host clean
    for (uint64_t i = 0; i < scene.objectsSize(); i++)
    {
        cudaFree(objects[i]);
    }
    for (uint64_t i = 0; i < scene.lightsSize(); i++)
    {
        cudaFree(lights[i]);
    }
    delete[] objects;
    delete[] lights;

    // Device clean
    cudaFree(d_pixels);
    cudaFree(d_camera);
    cudaFree(d_scene);
    cudaFree(d_objects);
    cudaFree(d_lights);

    return RETURN_SUCCESS;
}
