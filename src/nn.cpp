#include "nn.hpp"
#include "mujoco_env.hpp"

#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/random_uniform/operations_generic.h>
#include <rl_tools/rl/algorithms/td3/operations_generic.h>
#include <rl_tools/rl/components/off_policy_runner/operations_generic.h>
#include <rl_tools/rl/utils/evaluation/operations_generic.h>

#include "nn_environment.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace rlt = rl_tools;

using T = float;
using TYPE_POLICY = rlt::numeric_types::Policy<T>;
using DEVICE = rlt::devices::DEVICE_FACTORY<rlt::devices::DefaultCPUSpecification>;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using TI = DEVICE::index_t;

using ENVIRONMENT_PARAMETERS = CartPoleParameters<T>;
using ENVIRONMENT_SPEC = CartPoleSpecification<T, TI, ENVIRONMENT_PARAMETERS>;
using ENVIRONMENT = CartPole<ENVIRONMENT_SPEC>;

constexpr TI EPISODE_STEP_LIMIT = ENVIRONMENT::EPISODE_STEP_LIMIT;
constexpr TI REPLAY_BUFFER_CAPACITY = 100000;
constexpr TI ACTOR_NUM_LAYERS = 3;
constexpr TI ACTOR_HIDDEN_DIM = 64;
constexpr TI CRITIC_NUM_LAYERS = 3;
constexpr TI CRITIC_HIDDEN_DIM = 64;

struct TD3_CARTPOLE_PARAMETERS : rlt::rl::algorithms::td3::DefaultParameters<TYPE_POLICY, TI> {
    static constexpr TI ACTOR_BATCH_SIZE = 100;
    static constexpr TI CRITIC_BATCH_SIZE = 100;
};

using TD3_PARAMETERS = TD3_CARTPOLE_PARAMETERS;
using ACTOR_INPUT_SHAPE = rlt::tensor::Shape<TI, 1, TD3_PARAMETERS::ACTOR_BATCH_SIZE, ENVIRONMENT::Observation::DIM>;
using CRITIC_INPUT_SHAPE = rlt::tensor::Shape<TI, 1, TD3_PARAMETERS::CRITIC_BATCH_SIZE, ENVIRONMENT::Observation::DIM + ENVIRONMENT::ACTION_DIM>;
using ACTOR_CONFIG = rlt::nn_models::mlp::Configuration<TYPE_POLICY, TI, ENVIRONMENT::ACTION_DIM, ACTOR_NUM_LAYERS, ACTOR_HIDDEN_DIM, rlt::nn::activation_functions::RELU, rlt::nn::activation_functions::TANH>;
using CRITIC_CONFIG = rlt::nn_models::mlp::Configuration<TYPE_POLICY, TI, 1, CRITIC_NUM_LAYERS, CRITIC_HIDDEN_DIM, rlt::nn::activation_functions::RELU, rlt::nn::activation_functions::IDENTITY>;
using OPTIMIZER_SPEC = rlt::nn::optimizers::adam::Specification<TYPE_POLICY, TI>;
using OPTIMIZER = rlt::nn::optimizers::Adam<OPTIMIZER_SPEC>;
using PARAMETER_TYPE = rlt::nn::parameters::Adam;
using CAPABILITY_ACTOR = rlt::nn::capability::Gradient<PARAMETER_TYPE>;
using CAPABILITY_CRITIC = rlt::nn::capability::Gradient<PARAMETER_TYPE>;
using CAPABILITY_TARGET = rlt::nn::capability::Forward<>;

using ACTOR_TYPE = rlt::nn_models::mlp::Build<ACTOR_CONFIG, CAPABILITY_ACTOR, ACTOR_INPUT_SHAPE>;
using ACTOR_TARGET_TYPE = rlt::nn_models::mlp::NeuralNetwork<ACTOR_CONFIG, CAPABILITY_TARGET, ACTOR_INPUT_SHAPE>;
using CRITIC_TYPE = rlt::nn_models::mlp::Build<CRITIC_CONFIG, CAPABILITY_CRITIC, CRITIC_INPUT_SHAPE>;
using CRITIC_TARGET_TYPE = rlt::nn_models::mlp::NeuralNetwork<CRITIC_CONFIG, CAPABILITY_TARGET, CRITIC_INPUT_SHAPE>;

using TD3_SPEC = rlt::rl::algorithms::td3::Specification<TYPE_POLICY, TI, ENVIRONMENT, ACTOR_TYPE, ACTOR_TARGET_TYPE, CRITIC_TYPE, CRITIC_TARGET_TYPE, OPTIMIZER, TD3_PARAMETERS>;
using ACTOR_CRITIC_TYPE = rlt::rl::algorithms::td3::ActorCritic<TD3_SPEC>;

struct OFF_POLICY_RUNNER_PARAMETERS : rlt::rl::components::off_policy_runner::ParametersDefault<T, TI> {
    static constexpr TI REPLAY_BUFFER_CAPACITY = ::REPLAY_BUFFER_CAPACITY;
    static constexpr TI EPISODE_STEP_LIMIT = ::EPISODE_STEP_LIMIT;
    static constexpr T EXPLORATION_NOISE = 0.1f;
};

using EXPLORATION_POLICY_SPEC = rlt::nn_models::random_uniform::Specification<TYPE_POLICY, TI, ENVIRONMENT::Observation::DIM, ENVIRONMENT::ACTION_DIM, rlt::nn_models::random_uniform::Range::MINUS_ONE_TO_ONE>;
using EXPLORATION_POLICY = rlt::nn_models::RandomUniform<EXPLORATION_POLICY_SPEC>;
using POLICIES = rlt::utils::Tuple<TI, EXPLORATION_POLICY, ACTOR_TYPE>;
using OFF_POLICY_RUNNER_SPEC = rlt::rl::components::off_policy_runner::Specification<TYPE_POLICY, TI, ENVIRONMENT, POLICIES, OFF_POLICY_RUNNER_PARAMETERS>;
using OFF_POLICY_RUNNER_TYPE = rlt::rl::components::OffPolicyRunner<OFF_POLICY_RUNNER_SPEC>;

constexpr TI SEQUENCE_LENGTH = 1;
using CRITIC_BATCH_SPEC = rlt::rl::components::off_policy_runner::SequentialBatchSpecification<OFF_POLICY_RUNNER_SPEC, SEQUENCE_LENGTH, TD3_PARAMETERS::CRITIC_BATCH_SIZE>;
using ACTOR_BATCH_SPEC = rlt::rl::components::off_policy_runner::SequentialBatchSpecification<OFF_POLICY_RUNNER_SPEC, SEQUENCE_LENGTH, TD3_PARAMETERS::ACTOR_BATCH_SIZE>;
using EVAL_SPEC = rlt::rl::utils::evaluation::Specification<TYPE_POLICY, TI, ENVIRONMENT, 1, EPISODE_STEP_LIMIT>;
using ACTOR_EVAL_TYPE = ACTOR_TYPE::CHANGE_BATCH_SIZE<TI, OFF_POLICY_RUNNER_PARAMETERS::N_ENVIRONMENTS>;

struct NeuralNetwork::Impl {
    DEVICE device;
    RNG rng;
    OPTIMIZER actor_optimizer;
    OPTIMIZER critic_optimizers[2];
    TI seed = 1;
    ACTOR_CRITIC_TYPE actor_critic;
    OFF_POLICY_RUNNER_TYPE off_policy_runner;
    EXPLORATION_POLICY exploration_policy;
    EXPLORATION_POLICY::Buffer<> exploration_buffer;
    rlt::rl::components::off_policy_runner::SequentialBatch<CRITIC_BATCH_SPEC> critic_batch;
    rlt::rl::algorithms::td3::CriticTrainingBuffers<rlt::rl::algorithms::td3::CriticTrainingBuffersSpecification<ACTOR_CRITIC_TYPE::SPEC>> critic_training_buffers;
    CRITIC_TYPE::Buffer<> critic_buffer;
    rlt::rl::components::off_policy_runner::SequentialBatch<ACTOR_BATCH_SPEC> actor_batch;
    rlt::rl::algorithms::td3::ActorTrainingBuffers<rlt::rl::algorithms::td3::ActorTrainingBuffersSpecification<ACTOR_CRITIC_TYPE::SPEC>> actor_training_buffers;
    ACTOR_TYPE::Buffer<> actor_buffer;
    ACTOR_TARGET_TYPE::Buffer<> actor_target_buffer;
    CRITIC_TARGET_TYPE::Buffer<> critic_target_buffer;
    ACTOR_EVAL_TYPE::Buffer<> actor_buffer_eval;
    bool initialized = false;

    ~Impl() {
        release();
    }

    void release();
    void init();
    float train(MujocoEnv& environment, unsigned int episodes, unsigned int max_steps);
    float run(MujocoEnv& environment, unsigned int episodes, unsigned int max_steps);
    void save(const std::string& path) const;
    void load(const std::string& path);
};

namespace {
    constexpr std::array<char, 8> POLICY_MAGIC{{'R', 'L', 'T', 'D', '3', 'P', '1', '\0'}};
    constexpr TI WARMUP_STEPS = TD3_PARAMETERS::ACTOR_BATCH_SIZE;

    template <typename LAYER>
    void write_layer(std::ostream& output, const DEVICE& device, const LAYER& layer) {
        using VALUE = typename LAYER::SPEC::TYPE_POLICY::DEFAULT;
        for (TI output_i = 0; output_i < LAYER::SPEC::OUTPUT_DIM; ++output_i) {
            for (TI input_i = 0; input_i < LAYER::SPEC::INPUT_DIM; ++input_i) {
                const VALUE value = rlt::get(device, layer.weights.parameters, output_i, input_i);
                output.write(reinterpret_cast<const char*>(&value), sizeof(value));
            }
        }
        for (TI output_i = 0; output_i < LAYER::SPEC::OUTPUT_DIM; ++output_i) {
            const VALUE value = rlt::get(device, layer.biases.parameters, output_i);
            output.write(reinterpret_cast<const char*>(&value), sizeof(value));
        }
        if (!output) {
            throw std::runtime_error("Failed to write policy parameters");
        }
    }

    template <typename LAYER>
    void read_layer(std::istream& input, const DEVICE& device, LAYER& layer) {
        using VALUE = typename LAYER::SPEC::TYPE_POLICY::DEFAULT;
        for (TI output_i = 0; output_i < LAYER::SPEC::OUTPUT_DIM; ++output_i) {
            for (TI input_i = 0; input_i < LAYER::SPEC::INPUT_DIM; ++input_i) {
                VALUE value{};
                input.read(reinterpret_cast<char*>(&value), sizeof(value));
                if (!input) {
                    throw std::runtime_error("Policy file is incomplete");
                }
                rlt::set(device, layer.weights.parameters, value, output_i, input_i);
            }
        }
        for (TI output_i = 0; output_i < LAYER::SPEC::OUTPUT_DIM; ++output_i) {
            VALUE value{};
            input.read(reinterpret_cast<char*>(&value), sizeof(value));
            if (!input) {
                throw std::runtime_error("Policy file is incomplete");
            }
            rlt::set(device, layer.biases.parameters, value, output_i);
        }
    }
}

void NeuralNetwork::Impl::release() {
    if (!initialized) {
        return;
    }

    rlt::free(device, actor_buffer_eval);
    rlt::free(device, critic_target_buffer);
    rlt::free(device, actor_target_buffer);
    rlt::free(device, actor_buffer);
    rlt::free(device, actor_training_buffers);
    rlt::free(device, actor_batch);
    rlt::free(device, critic_buffer);
    rlt::free(device, critic_training_buffers);
    rlt::free(device, critic_batch);
    rlt::free(device, exploration_buffer);
    rlt::free(device, off_policy_runner);
    rlt::free(device, exploration_policy);
    rlt::free(device, actor_critic);
    rlt::free(device, critic_optimizers[1]);
    rlt::free(device, critic_optimizers[0]);
    rlt::free(device, actor_optimizer);
    initialized = false;
}

void NeuralNetwork::Impl::init() {
    if (initialized) {
        return;
    }

    rlt::init(device, rng, seed);
    rlt::malloc(device, actor_optimizer);
    rlt::malloc(device, critic_optimizers[0]);
    rlt::malloc(device, critic_optimizers[1]);
    rlt::init(device, actor_optimizer);
    rlt::init(device, critic_optimizers[0]);
    rlt::init(device, critic_optimizers[1]);

    rlt::malloc(device, actor_critic);
    rlt::init(device, actor_critic, rng);
    rlt::reset_optimizer_state(device, actor_optimizer, actor_critic.actor);
    rlt::reset_optimizer_state(device, critic_optimizers[0], actor_critic.critics[0]);
    rlt::reset_optimizer_state(device, critic_optimizers[1], actor_critic.critics[1]);

    rlt::malloc(device, exploration_policy);
    rlt::malloc(device, off_policy_runner);
    rlt::init(device, off_policy_runner);

    rlt::malloc(device, critic_batch);
    rlt::malloc(device, critic_training_buffers);
    rlt::malloc(device, critic_buffer);
    rlt::malloc(device, actor_batch);
    rlt::malloc(device, actor_training_buffers);
    rlt::malloc(device, actor_buffer);
    rlt::malloc(device, actor_target_buffer);
    rlt::malloc(device, critic_target_buffer);
    rlt::malloc(device, actor_buffer_eval);
    rlt::malloc(device, exploration_buffer);

    initialized = true;
}

float NeuralNetwork::Impl::train(MujocoEnv& environment, unsigned int episodes, unsigned int max_steps) {
    if (episodes == 0 || max_steps == 0 || max_steps > EPISODE_STEP_LIMIT
        || max_steps > static_cast<unsigned int>(std::numeric_limits<TI>::max())) {
        throw std::invalid_argument("Training episodes and max steps must be positive; max steps cannot exceed 1000");
    }

    init();
    auto& training_environment = rlt::get(off_policy_runner.envs, 0, 0);
    training_environment.simulator = &environment;
    training_environment.max_episode_steps = static_cast<TI>(max_steps);

    TI step_i = 0;
    unsigned int completed_episodes = 0;
    double total_return = 0;
    while (completed_episodes < episodes) {
        if (step_i < WARMUP_STEPS) {
            rlt::step<0>(device, off_policy_runner, exploration_policy, exploration_buffer, rng);
        } else {
            rlt::step<1>(device, off_policy_runner, actor_critic.actor, actor_buffer_eval, rng);
        }

        if (step_i >= WARMUP_STEPS
            && step_i >= TD3_PARAMETERS::CRITIC_BATCH_SIZE) {
            for (TI critic_i = 0; critic_i < 2; ++critic_i) {
                auto target_action_noise = rlt::matrix_view(device, critic_training_buffers.target_next_action_noise);
                rlt::target_action_noise(device, actor_critic, target_action_noise, rng);
                rlt::gather_batch(device, off_policy_runner, critic_batch, rng);
                rlt::train_critic(
                    device, actor_critic, actor_critic.critics[critic_i], critic_batch,
                    critic_optimizers[critic_i], actor_buffer, actor_target_buffer,
                    critic_buffer, critic_target_buffer, critic_training_buffers, rng);
            }

            if (step_i % 2 == 0) {
                rlt::gather_batch(device, off_policy_runner, actor_batch, rng);
                rlt::train_actor(
                    device, actor_critic, actor_batch, actor_optimizer,
                    actor_buffer, critic_buffer, actor_training_buffers, rng);
                rlt::update_critic_targets(device, actor_critic);
                rlt::update_actor_target(device, actor_critic);
            }
        }

        if (rlt::get(off_policy_runner.truncated, 0, 0)) {
            total_return += rlt::get(off_policy_runner.episode_return, 0, 0);
            ++completed_episodes;
        }
        ++step_i;
    }
    return static_cast<float>(total_return / episodes);
}

float NeuralNetwork::Impl::run(MujocoEnv& environment, unsigned int episodes, unsigned int max_steps) {
    if (episodes == 0 || max_steps == 0 || max_steps > EPISODE_STEP_LIMIT
        || max_steps > static_cast<unsigned int>(std::numeric_limits<TI>::max())) {
        throw std::invalid_argument("Evaluation episodes and max steps must be positive; max steps cannot exceed 1000");
    }

    init();
    ENVIRONMENT evaluation_environment;
    evaluation_environment.simulator = &environment;
    evaluation_environment.max_episode_steps = static_cast<TI>(max_steps);
    rlt::rl::environments::DummyUI ui;
    rlt::rl::utils::evaluation::Result<EVAL_SPEC> result;

    double total_return = 0;
    for (unsigned int episode_i = 0; episode_i < episodes; ++episode_i) {
        rlt::evaluate(
            device, evaluation_environment, ui, actor_critic.actor, result, rng,
            rlt::Mode<rlt::mode::Evaluation<>>{});
        total_return += result.returns_mean;
    }
    return static_cast<float>(total_return / episodes);
}

void NeuralNetwork::Impl::save(const std::string& path) const {
    if (!initialized) {
        throw std::logic_error("Cannot save a policy before initializing the network");
    }

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("Could not open policy file for writing: " + path);
    }

    output.write(POLICY_MAGIC.data(), POLICY_MAGIC.size());
    const std::uint32_t dimensions[] = {
        static_cast<std::uint32_t>(ENVIRONMENT::Observation::DIM),
        static_cast<std::uint32_t>(ACTOR_HIDDEN_DIM),
        static_cast<std::uint32_t>(ENVIRONMENT::ACTION_DIM),
        static_cast<std::uint32_t>(ACTOR_NUM_LAYERS)
    };
    output.write(reinterpret_cast<const char*>(dimensions), sizeof(dimensions));
    write_layer(output, device, actor_critic.actor.input_layer);
    for (TI layer_i = 0; layer_i < ACTOR_TYPE::NUM_HIDDEN_LAYERS; ++layer_i) {
        write_layer(output, device, actor_critic.actor.hidden_layers[layer_i]);
    }
    write_layer(output, device, actor_critic.actor.output_layer);
}

void NeuralNetwork::Impl::load(const std::string& path) {
    init();
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Could not open policy file for reading: " + path);
    }

    std::array<char, POLICY_MAGIC.size()> magic{};
    input.read(magic.data(), magic.size());
    if (!input || magic != POLICY_MAGIC) {
        throw std::runtime_error("Policy file has an unsupported format");
    }

    std::uint32_t dimensions[4]{};
    input.read(reinterpret_cast<char*>(dimensions), sizeof(dimensions));
    const std::uint32_t expected_dimensions[] = {
        static_cast<std::uint32_t>(ENVIRONMENT::Observation::DIM),
        static_cast<std::uint32_t>(ACTOR_HIDDEN_DIM),
        static_cast<std::uint32_t>(ENVIRONMENT::ACTION_DIM),
        static_cast<std::uint32_t>(ACTOR_NUM_LAYERS)
    };
    if (!input || !std::equal(std::begin(dimensions), std::end(dimensions), std::begin(expected_dimensions))) {
        throw std::runtime_error("Policy file architecture does not match this build");
    }

    read_layer(input, device, actor_critic.actor.input_layer);
    for (TI layer_i = 0; layer_i < ACTOR_TYPE::NUM_HIDDEN_LAYERS; ++layer_i) {
        read_layer(input, device, actor_critic.actor.hidden_layers[layer_i]);
    }
    read_layer(input, device, actor_critic.actor.output_layer);
    if (input.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("Policy file contains unexpected trailing data");
    }
}

NeuralNetwork::NeuralNetwork() : impl(std::make_unique<Impl>()) {}

NeuralNetwork::~NeuralNetwork() = default;

void NeuralNetwork::init() {
    impl->init();
}

float NeuralNetwork::train(MujocoEnv& environment, unsigned int episodes, unsigned int max_steps) {
    return impl->train(environment, episodes, max_steps);
}

float NeuralNetwork::run(MujocoEnv& environment, unsigned int episodes, unsigned int max_steps) {
    return impl->run(environment, episodes, max_steps);
}

void NeuralNetwork::save(const std::string& path) const {
    impl->save(path);
}

void NeuralNetwork::load(const std::string& path) {
    impl->load(path);
}
