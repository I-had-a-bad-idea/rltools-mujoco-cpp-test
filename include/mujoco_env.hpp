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
    double accumulated_simulation_time = 0.0;

    int slider_id = -1;
    int hinge_id = -1;

    public:
        MujocoEnv() = default;
        ~MujocoEnv();
        MujocoEnv(const MujocoEnv&) = delete;
        MujocoEnv& operator=(const MujocoEnv&) = delete;

        void load(const char* xml_path);
        void reset(float cart_position = 0.0f, float cart_velocity = 0.0f,
                   float pole_angle = 0.0f, float pole_velocity = 0.0f);
        Observation observe() const;
        void step(float action);
        float reward(float cart_x, float pole_angle, float action);
        float timestep() const;

        void close();
};