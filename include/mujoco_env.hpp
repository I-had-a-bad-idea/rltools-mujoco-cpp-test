#pragma once

#include <mujoco/mujoco.h>

#include <array>
#include <stdexcept>

// Pos of cart, velocity of cart, velocity of pole, rotational velocity of hinge
struct Observation {
    std::array<float, 4> values;
};

class MujocoEnv {
    mjModel* model = nullptr;
    mjData* data = nullptr;

    int slider_id = -1;
    int hinge_id = -1;

    public:
        void load(const char* xml_path);
        Observation observe() const;
        void step(float action);
        float reward(float cart_x, float pole_angle, float action);

        void close();
};