#pragma once

#include <memory>
#include <string>

class MujocoEnv;

class NeuralNetwork {
    struct Impl;
    std::unique_ptr<Impl> impl;

    public:
        NeuralNetwork();
        ~NeuralNetwork();
        NeuralNetwork(const NeuralNetwork&) = delete;
        NeuralNetwork& operator=(const NeuralNetwork&) = delete;

        void init();
        float train(MujocoEnv& environment, unsigned int episodes, unsigned int max_steps);
        float run(MujocoEnv& environment, unsigned int episodes, unsigned int max_steps);
        void save(const std::string& path) const;
        void load(const std::string& path);
};
