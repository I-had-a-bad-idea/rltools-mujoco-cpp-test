#pragma once

#include "mujoco_env.hpp"

#include <rl_tools/rl/environments/operations_generic.h>

template <typename T>
struct CartPoleParameters {};

template <typename T_TYPE, typename TI_TYPE, typename PARAMETERS_TYPE = CartPoleParameters<T_TYPE>>
struct CartPoleSpecification {
    using T = T_TYPE;
    using TI = TI_TYPE;
    using PARAMETERS = PARAMETERS_TYPE;
};

template <typename T, typename TI>
struct CartPoleState {
    static constexpr TI DIM = 4;

    T x = 0;
    T x_dot = 0;
    T theta = 0;
    T theta_dot = 0;
    TI steps = 0;
};

template <typename TI, typename T>
struct CartPoleObservation {
    static constexpr TI DIM = 4;
};

template <typename SPEC>
struct CartPole {
    using SPEC_TYPE = SPEC;
    using T = typename SPEC::T;
    using TI = typename SPEC::TI;
    using Parameters = typename SPEC::PARAMETERS;
    using State = CartPoleState<T, TI>;
    using Observation = CartPoleObservation<TI, T>;
    using ObservationPrivileged = Observation;

    static constexpr TI ACTION_DIM = 1;
    static constexpr TI N_AGENTS = 1;
    static constexpr TI EPISODE_STEP_LIMIT = 1000;

    MujocoEnv* simulator = nullptr;
    TI max_episode_steps = EPISODE_STEP_LIMIT;
};

namespace rl_tools {
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE&, const CartPole<SPEC>&) {}

    template <typename DEVICE, typename SPEC>
    void free(DEVICE&, const CartPole<SPEC>&) {}

    template <typename DEVICE, typename SPEC>
    void init(DEVICE&, const CartPole<SPEC>&) {}

    template <typename DEVICE, typename SPEC, typename RNG>
    void sample_initial_parameters(DEVICE&, const CartPole<SPEC>&,
                                   typename CartPole<SPEC>::Parameters&, RNG&) {}

    template <typename DEVICE, typename SPEC>
    void initial_parameters(DEVICE&, const CartPole<SPEC>&,
                            typename CartPole<SPEC>::Parameters&) {}

    template <typename DEVICE, typename SPEC, typename RNG>
    void sample_initial_state(DEVICE& device, const CartPole<SPEC>& env,
                              const typename CartPole<SPEC>::Parameters&,
                              typename CartPole<SPEC>::State& state, RNG& rng) {
        if (!env.simulator) {
            throw std::runtime_error("MuJoCo environment is not connected to the RLTools adapter");
        }

        using T = typename SPEC::T;
        const T x = random::uniform_real_distribution(device.random, (T)-0.05, (T)0.05, rng);
        const T x_dot = random::uniform_real_distribution(device.random, (T)-0.05, (T)0.05, rng);
        const T theta = random::uniform_real_distribution(device.random, (T)-0.05, (T)0.05, rng);
        const T theta_dot = random::uniform_real_distribution(device.random, (T)-0.05, (T)0.05, rng);
        env.simulator->reset((float)x, (float)x_dot, (float)theta, (float)theta_dot);

        state.x = x;
        state.x_dot = x_dot;
        state.theta = theta;
        state.theta_dot = theta_dot;
        state.steps = 0;
    }

    template <typename DEVICE, typename SPEC>
    void initial_state(DEVICE&, const CartPole<SPEC>& env,
                       const typename CartPole<SPEC>::Parameters&,
                       typename CartPole<SPEC>::State& state) {
        if (!env.simulator) {
            throw std::runtime_error("MuJoCo environment is not connected to the RLTools adapter");
        }

        env.simulator->reset();
        state.x = 0;
        state.x_dot = 0;
        state.theta = 0;
        state.theta_dot = 0;
        state.steps = 0;
    }

    template <typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    typename SPEC::T step(DEVICE&, const CartPole<SPEC>& env,
                               typename CartPole<SPEC>::Parameters&,
                               const typename CartPole<SPEC>::State& state,
                               const Matrix<ACTION_SPEC>& action,
                               typename CartPole<SPEC>::State& next_state, RNG&) {
        if (!env.simulator) {
            throw std::runtime_error("MuJoCo environment is not connected to the RLTools adapter");
        }

        const auto control = get(action, 0, 0);
        env.simulator->step((float)control);
        const auto observation = env.simulator->observe();
        next_state.x = observation.values[0];
        next_state.x_dot = observation.values[1];
        next_state.theta = observation.values[2];
        next_state.theta_dot = observation.values[3];
        next_state.steps = state.steps + 1;
        return (typename SPEC::T)env.simulator->timestep();
    }

    template <typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    typename SPEC::T reward(DEVICE&, const CartPole<SPEC>&,
                                typename CartPole<SPEC>::Parameters&,
                                const typename CartPole<SPEC>::State&,
                                const Matrix<ACTION_SPEC>& action,
                                const typename CartPole<SPEC>::State& next_state, RNG&) {
        const auto normalized_action = get(action, 0, 0);
        return (typename SPEC::T)1
            - next_state.theta * next_state.theta
            - (typename SPEC::T)0.1 * next_state.x * next_state.x
            - (typename SPEC::T)0.001 * normalized_action * normalized_action;
    }

    template <typename DEVICE, typename SPEC, typename OBS_TYPE_SPEC,
              typename OBS_SPEC, typename RNG>
    void observe(DEVICE&, const CartPole<SPEC>&,
                 const typename CartPole<SPEC>::Parameters&,
                 const typename CartPole<SPEC>::State& state,
                 const CartPoleObservation<OBS_TYPE_SPEC, typename SPEC::T>&,
                 Matrix<OBS_SPEC>& observation, RNG&) {
        static_assert(OBS_SPEC::ROWS == 1);
        static_assert(OBS_SPEC::COLS == CartPole<SPEC>::Observation::DIM);
        set(observation, 0, 0, state.x);
        set(observation, 0, 1, state.x_dot);
        set(observation, 0, 2, state.theta);
        set(observation, 0, 3, state.theta_dot);
    }

    template <typename DEVICE, typename SPEC, typename RNG>
    bool terminated(DEVICE&, const CartPole<SPEC>& env,
                    const typename CartPole<SPEC>::Parameters&,
                    const typename CartPole<SPEC>::State& state, RNG&) {
        return state.steps >= env.max_episode_steps
            || state.x <= (typename SPEC::T)-2.4
            || state.x >= (typename SPEC::T)2.4
            || state.theta <= (typename SPEC::T)-0.4
            || state.theta >= (typename SPEC::T)0.4;
    }
}
