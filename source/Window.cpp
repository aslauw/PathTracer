#include "Window.hpp"
#include <iostream>
#include <exception>

// Ctor
Window::Window(uint32_t width, uint32_t height)
: _width(width)
, _height(height)
, _window(0)
, _renderer(0)
, _texture(0)
, _pixels(0)
, _valid(true)
, _over(false)
{
    if (SDL_Init(SDL_INIT_EVERYTHING) != 0)
    {
        std::cerr << "SDL_Init: " << SDL_GetError() << std::endl;
        _valid = false;
    }
    // Create SDL Window
    _window = SDL_CreateWindow("PathTracer", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, _width, _height, 0);
    if (_window == NULL)
    {
        std::cerr << "SDL_CreateWindow: " << SDL_GetError() << std::endl;
        SDL_Quit();
        _valid = false;
    }
    // Create SDL Renderer
    _renderer = SDL_CreateRenderer(_window, -1, SDL_RENDERER_ACCELERATED);
    if (_renderer == NULL)
    {
        std::cerr << "SDL_CreateRenderer: " << SDL_GetError() << std::endl;
        SDL_DestroyWindow(_window);
        SDL_Quit();
        _valid = false;
    }
    // Create SDL Texture
    _texture = SDL_CreateTexture(_renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, _width, _height);
    if (_texture == NULL)
    {
        std::cerr << "SDL_CreateTexture: " << SDL_GetError() << std::endl;
        SDL_DestroyRenderer(_renderer);
        SDL_DestroyWindow(_window);
        SDL_Quit();
        _valid = false;
    }
    // Allocate pixels
    try
    {
        _pixels = new Color[_width * _height];
    }
    catch (std::exception e)
    {
        std::cerr << e.what() << std::endl;
        _valid = false;
    }
    memset(_pixels, BLACK_VISIBLE, _width * _height * sizeof(Color));
    _pitch = _width * sizeof(Color);
    std::memset(&_keys, false, sizeof(Keys));
}

// Functions
void
Window::pollEvents()
{
    while (SDL_PollEvent(&_event))
    {
        if (_event.type == SDL_QUIT)
        {
            _over = true;
        }
        else if (_event.type == SDL_KEYDOWN)
        {
            switch (_event.key.keysym.sym)
            {
                case SDLK_ESCAPE:
                    _over = true;
                    break;
                case SDLK_w:
                    _keys.moveForward = true;
                    break;
                case SDLK_s:
                    _keys.moveBack = true;
                    break;
                case SDLK_a:
                    _keys.moveLeft = true;
                    break;
                case SDLK_d:
                    _keys.moveRight = true;
                    break;
                case SDLK_SPACE:
                    _keys.moveUp = true;
                    break;
                case SDLK_LSHIFT:
                    _keys.moveDown = true;
                    break;
                case SDLK_UP:
                    _keys.rotateUp = true;
                    break;
                case SDLK_DOWN:
                    _keys.rotateDown = true;
                    break;
                case SDLK_LEFT:
                    _keys.rotateLeft = true;
                    break;
                case SDLK_RIGHT:
                    _keys.rotateRight = true;
                    break;
                case SDLK_LEFTBRACKET:
                    _keys.barrelLeft = true;
                    break;
                case SDLK_RIGHTBRACKET:
                    _keys.barrelRight = true;
                    break;
                case SDLK_k:
                    _keys.fovDown = true;
                    break;
                case SDLK_l:
                    _keys.fovUp = true;
                    break;
                default:
                    break;
            }
        }
        else if (_event.type == SDL_KEYUP)
        {
            switch (_event.key.keysym.sym)
            {
                case SDLK_w:
                    _keys.moveForward = false;
                    break;
                case SDLK_s:
                    _keys.moveBack = false;
                    break;
                case SDLK_a:
                    _keys.moveLeft = false;
                    break;
                case SDLK_d:
                    _keys.moveRight = false;
                    break;
                case SDLK_SPACE:
                    _keys.moveUp = false;
                    break;
                case SDLK_LSHIFT:
                    _keys.moveDown = false;
                    break;
                case SDLK_UP:
                    _keys.rotateUp = false;
                    break;
                case SDLK_DOWN:
                    _keys.rotateDown = false;
                    break;
                case SDLK_LEFT:
                    _keys.rotateLeft = false;
                    break;
                case SDLK_RIGHT:
                    _keys.rotateRight = false;
                    break;
                case SDLK_LEFTBRACKET:
                    _keys.barrelLeft = false;
                    break;
                case SDLK_RIGHTBRACKET:
                    _keys.barrelRight = false;
                    break;
                case SDLK_k:
                    _keys.fovDown = false;
                    break;
                case SDLK_l:
                    _keys.fovUp = false;
                    break;
                default:
                    break;
            }
        }
    }
}

void
Window::update(Camera& camera)
{
    camera.updatePosition(_keys);
}

void
Window::render()
{
    // uint64_t    idHit;
    // float       dist;
    // Color       color;

    // dist = INFINITY;

    // memset(_pixels, BLACK_VISIBLE, _width * _height * sizeof(Color));
    // for (uint32_t i = 0; i < _height; i++)
    // {
    //     for (uint32_t j = 0; j < _width; j++)
    //     {
    //         // Setup ray
    //         Vec3<float> orig = camera.position();
    //         Vec3<float> pixel = camera.cache()[i * _width + j];
    //         pixel = camera.toWorld(pixel);
    //         Ray ray(orig, pixel - orig);
    //         ray.normalize();

    //         // Throw ray
    //         dist = camera.throwRay(scene, ray, &idHit);
    //         // Hit nothing, continue
    //         if (dist == INFINITY)
    //             continue;

    //         // Compute shading
    //         Vec3<float> dir(ray.direction());
    //         dir *= dist;
    //         color = camera.computeShading(orig, orig + dir, idHit, scene, 0);
    //         _pixels[i * _width + j] = color;
    //     }
    // }

    SDL_UpdateTexture(_texture, NULL, _pixels, _width * sizeof(uint32_t));
    SDL_RenderCopy(_renderer, _texture, NULL, NULL);
    SDL_RenderPresent(_renderer);
}

// Dtor
Window::~Window()
{
    SDL_DestroyTexture(_texture);
    SDL_DestroyRenderer(_renderer);
    SDL_DestroyWindow(_window);
    SDL_Quit();
}
