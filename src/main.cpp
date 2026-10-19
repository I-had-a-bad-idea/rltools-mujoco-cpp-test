#include "nn.hpp"
#include "mujoco_env.hpp"

#include <exception>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2 || argc > 5) {
        std::cerr << "Usage: test <model.xml> [train|run] [episodes] [policy-file]\n";
        return 1;
    }

    const std::string mode = argc > 2 ? argv[2] : "train";
    if (mode != "train" && mode != "run") {
        std::cerr << "Mode must be either 'train' or 'run'\n";
        return 1;
    }

    unsigned int episodes = mode == "train" ? 100 : 5;
    if (argc > 3) {
        try {
            std::size_t parsed_characters = 0;
            const unsigned long parsed = std::stoul(argv[3], &parsed_characters);
            if (parsed_characters != std::string(argv[3]).size()
                || parsed == 0 || parsed > 1000000UL) {
                throw std::out_of_range("episodes");
            }
            episodes = static_cast<unsigned int>(parsed);
        } catch (const std::exception&) {
            std::cerr << "Episodes must be an integer between 1 and 1000000\n";
            return 1;
        }
    }
    const std::string policy_path = argc > 4 ? argv[4] : "policy.dat";

    try {
        MujocoEnv environment;
        environment.load(argv[1]);

        NeuralNetwork nn;
        if (mode == "train") {
            nn.init();
            const float average_return = nn.train(environment, episodes, 1000);
            nn.save(policy_path);
            std::cout << "Training average return: " << average_return << '\n'
                      << "Saved policy to " << policy_path << '\n';
        } else {
            nn.load(policy_path);
            const float average_return = nn.run(environment, episodes, 1000);
            std::cout << "Evaluation average return: " << average_return << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}