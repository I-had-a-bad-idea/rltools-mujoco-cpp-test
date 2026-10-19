#pragma once

#include <memory>

class NeuralNetwork {
    struct Impl;
    std::unique_ptr<Impl> impl;

    public:
        NeuralNetwork();
        ~NeuralNetwork();

        void init();
};