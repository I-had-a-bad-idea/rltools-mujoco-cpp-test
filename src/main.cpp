#include "nn.hpp"
#include "mujoco_env.hpp"

#include <exception>
#include <iostream>
#include <string>
#include <chrono>

int main(int argc, char** argv) {
    if (argc < 2 || argc > 5) {
        std::cerr << "Usage: test <model.xml> [train|run] [episodes] [policy-file]\n"
                  << "       test <model.xml> run [policy-file]\n";
        return 1;
    }

    const std::string mode = argc > 2 ? argv[2] : "train";
    if (mode != "train" && mode != "run") {
        std::cerr << "Mode must be either 'train' or 'run'\n";
        return 1;
    }

    unsigned int episodes = mode == "train" ? 100 : 5;
    int policy_argument = 3;
    if (argc > 3) {
        const std::string third_argument = argv[3];
        const bool is_episode_count = !third_argument.empty()
            && third_argument.find_first_not_of("0123456789") == std::string::npos;
        if (is_episode_count) {
            try {
                std::size_t parsed_characters = 0;
                const unsigned long parsed = std::stoul(argv[3], &parsed_characters);
                if (parsed_characters != third_argument.size()
                    || parsed == 0 || parsed > 1000000UL) {
                    throw std::out_of_range("episodes");
                }
                episodes = static_cast<unsigned int>(parsed);
                policy_argument = 4;
            } catch (const std::exception&) {
                std::cerr << "Episodes must be an integer between 1 and 1000000\n";
                return 1;
            }
        } else if (argc > 4) {
            std::cerr << "Specify either an episode count and policy file, or a policy file alone\n";
            return 1;
        }
    }
    const std::string policy_path = argc > policy_argument ? argv[policy_argument] : "policy.dat";

    try {
        MujocoEnv environment;
        environment.load(argv[1]);

        NeuralNetwork nn;
        if (mode == "train") {
            const auto start = std::chrono::steady_clock::now();
            nn.init();
            const float average_return = nn.train(environment, episodes, 1000);
            const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start);
            nn.save(policy_path);
            std::cout << "Training average return: " << average_return << '\n'
                      << "Training wall time: " << elapsed.count() << " s\n"
                      << "Saved policy to " << policy_path << '\n';
        } else {
            const auto start = std::chrono::steady_clock::now();
            nn.load(policy_path);
            const float average_return = nn.run(environment, episodes, 1000);
            const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start);
            std::cout << "Evaluation average return: " << average_return << '\n'
                      << "Evaluation wall time: " << elapsed.count() << " s\n";
        }
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}