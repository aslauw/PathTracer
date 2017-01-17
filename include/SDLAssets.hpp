#ifndef SDLASSETS_HPP
#define SDLASSETS_HPP

#define BLACK_VISIBLE   4278190080
#define RED_VISIBLE     4294901760

typedef union u_Color
{
    struct
    {
        uint8_t b;
        uint8_t g;
        uint8_t r;
        uint8_t a;
    };
    uint32_t    v;
} Color;

typedef struct s_Keys
{
    bool    moveForward;
    bool    moveBack;
    bool    moveLeft;
    bool    moveRight;
    bool    moveUp;
    bool    moveDown;
    bool    rotateUp;
    bool    rotateDown;
    bool    rotateLeft;
    bool    rotateRight;
    bool    barrelLeft;
    bool    barrelRight;
    bool    fovDown;
    bool    fovUp;
} Keys;

#endif
