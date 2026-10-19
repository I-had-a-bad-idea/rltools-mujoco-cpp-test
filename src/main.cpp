#include <mujoco/mujoco.h>

#include "nn.hpp"
#include "mujoco_env.hpp"

#include <stdio.h>

// See https://mujoco.readthedocs.io/en/stable/programming/simulation.html

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: test <model.xml>\n");
        return 1;
    }

    MujocoEnv mujoco_env;
    mujoco_env.load(argv[1]);
    
    NeuralNetwork nn;
    nn.init();

    for (int i = 0; i < 1000; ++i) {
        mujoco_env.step(0.0f);
    }

    mujoco_env.close();
    return 0;
}