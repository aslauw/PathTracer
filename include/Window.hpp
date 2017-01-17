#ifndef WINDOW_HPP
#define WINDOW_HPP

#include <cstdint>
#include <SDL2/SDL.h>

#include "SDLAssets.hpp"
#include "Camera.hpp"

class Window
{
public:
    Window(uint32_t width, uint32_t height);

    // Geters
    uint32_t        width()     { return _width; }
    uint32_t        height()    { return _height; }
    SDL_Window*     window()    { return _window; }
    SDL_Renderer*   renderer()  { return _renderer; }
    SDL_Texture*    texture()   { return _texture; }
    SDL_Event&      event()     { return _event; }
    Color*          pixels()    { return _pixels; }
    int             pitch()     { return _pitch; }
    bool            valid()     { return _valid; }
    bool            over()      { return _over; }

    // Functions
    void            pollEvents();    
    void            update(class Camera& camera);
    void            render();

    ~Window();

private:
    uint32_t        _width;
    uint32_t        _height;
    SDL_Window*     _window;
    SDL_Renderer*   _renderer;
    SDL_Texture*    _texture;
    SDL_Event       _event;
    Color*          _pixels;
    int             _pitch;
    bool            _valid;
    bool            _over;
    Keys            _keys;
};

#endif
