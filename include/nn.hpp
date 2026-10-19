#define RL_TOOLS_BACKEND_ENABLE_OPENBLAS
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/layers/standardize/operations_generic.h>
#include <rl_tools/nn/layers/dense/operations_cpu_openblas.h>
#include <rl_tools/nn_models/operations_cpu.h>


namespace rlt = rl_tools;


template <typename T>
struct CartPoleParameters {
    constexpr static T G = 9.81;
    constexpr static T CART_MASS = 1.0;
    constexpr static T POLE_MASS = 0.1;
    constexpr static T POLE_HALF_LENGTH = 0.5;
    constexpr static T MAX_FORCE = 15.0;
    constexpr static T MAX_CART_POSITION = 2.5;
    constexpr static T MAX_SPEED = 1.0;
    constexpr static T DT = 0.05;
    constexpr static T INITIAL_STATE_MIN_CART_POSITION = -MAX_CART_POSITION;
    constexpr static T INITIAL_STATE_MAX_CART_POSITION = MAX_CART_POSITION;
    constexpr static T INITIAL_STATE_MIN_ANGLE = -rlt::math::PI<T>;
    constexpr static T INITIAL_STATE_MAX_ANGLE = rlt::math::PI<T>;
    constexpr static T INITIAL_STATE_MIN_SPEED = -MAX_SPEED;
    constexpr static T INITIAL_STATE_MAX_SPEED = MAX_SPEED;
};

template <typename T_T, typename T_TI, typename T_PARAMETERS = CartPoleParameters<T_T>>
struct CartPoleSpecification{
    using T = T_T;
    using TI = T_TI;
    using PARAMETERS = T_PARAMETERS;
};

template <typename T, typename TI>
struct CartPoleState{
    static constexpr TI DIM = 4; // x, x_dot, theta, theta_dot
    T x;
    T x_dot;
    T theta;
    T theta_dot;
};
template <typename TI, typename T = float>
struct CartPoleFourierObservation{
    static constexpr TI DIM = 4; // x, x_dot, theta, theta_dot
    T x;
    T x_dot;
    T theta;
    T theta_dot;
};

template <typename T_SPEC>
struct CartPole{
    using SPEC = T_SPEC;
    using T = typename SPEC::T;
    using TI = typename SPEC::TI;
    using Parameters = typename SPEC::PARAMETERS;
    using State = CartPoleState<T, TI>;
    using Observation = CartPoleFourierObservation<TI, T>;
    using ObservationPrivileged = Observation;
    static constexpr TI ACTION_DIM = 1;
    static constexpr TI N_AGENTS = 1;
};



namespace rl_tools{
    template<typename DEVICE, typename SPEC>
    static void malloc(DEVICE& device, const CartPole<SPEC>& env){}
    template<typename DEVICE, typename SPEC>
    static void free(DEVICE& device, const CartPole<SPEC>& env){}
    template<typename DEVICE, typename SPEC>
    static void init(DEVICE& device, const CartPole<SPEC>& env){}
    template<typename DEVICE, typename SPEC, typename RNG>
    static void sample_initial_parameters(DEVICE& device, const CartPole<SPEC>& env, typename CartPole<SPEC>::Parameters& parameters, RNG& rng){ }
    template<typename DEVICE, typename SPEC>
    static void initial_parameters(DEVICE& device, const CartPole<SPEC>& env, typename CartPole<SPEC>::Parameters& parameters){ }
    template<typename DEVICE, typename SPEC, typename RNG>
    static void sample_initial_state(DEVICE& device, const CartPole<SPEC>& env, const typename CartPole<SPEC>::Parameters& parameters, typename CartPole<SPEC>::State& state, RNG& rng){
        state.x = rlt::random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), SPEC::PARAMETERS::INITIAL_STATE_MIN_CART_POSITION, SPEC::PARAMETERS::INITIAL_STATE_MAX_CART_POSITION, rng);
        state.x_dot = rlt::random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), SPEC::PARAMETERS::INITIAL_STATE_MIN_SPEED, SPEC::PARAMETERS::INITIAL_STATE_MAX_SPEED, rng);
        state.theta = rlt::random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), SPEC::PARAMETERS::INITIAL_STATE_MIN_ANGLE, SPEC::PARAMETERS::INITIAL_STATE_MAX_ANGLE, rng);
        state.theta_dot = rlt::random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), SPEC::PARAMETERS::INITIAL_STATE_MIN_SPEED, SPEC::PARAMETERS::INITIAL_STATE_MAX_SPEED, rng);
    }
    template<typename DEVICE, typename SPEC>
    static void initial_state(DEVICE& device, const CartPole<SPEC>& env, const typename CartPole<SPEC>::Parameters& parameters, typename CartPole<SPEC>::State& state){
        state.x = 0;
        state.x_dot = 0;
        state.theta = 0;
        state.theta_dot = 0;
    }
}

template <typename T>
T clip(T x, T min, T max){
    x = x < min ? min : (x > max ? max : x);
    return x;
}
template <typename DEVICE, typename T>
T f_mod_python(const DEVICE& dev, T a, T b){
    return a - b * rlt::math::floor(dev, a / b);
}

template <typename DEVICE, typename T>
T angle_normalize(const DEVICE& dev, T x){
    return f_mod_python(dev, (x + rlt::math::PI<T>), (2 * rlt::math::PI<T>)) - rlt::math::PI<T>;
}

namespace rl_tools{
    template<typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    typename SPEC::T step(DEVICE& device, const CartPole<SPEC>& env, const typename CartPole<SPEC>::Parameters& parameters, const typename CartPole<SPEC>::State& state, const Matrix<ACTION_SPEC>& action, typename CartPole<SPEC>::State& next_state, RNG& rng) {
        static_assert(ACTION_SPEC::ROWS == 1);
        static_assert(ACTION_SPEC::COLS == 1);
        typedef typename SPEC::T T;
        typedef typename SPEC::PARAMETERS PARAMS;

        const T force = clip(get(action, 0, 0) * PARAMS::MAX_FORCE, -PARAMS::MAX_FORCE, PARAMS::MAX_FORCE);
        const T dt = PARAMS::DT;
        const T total_mass = PARAMS::CART_MASS + PARAMS::POLE_MASS;
        const T pole_half_length = PARAMS::POLE_HALF_LENGTH;
        const T g = PARAMS::G;
        const T sin_theta = rlt::math::sin(device.math, state.theta);
        const T cos_theta = rlt::math::cos(device.math, state.theta);

        const T theta_acc = (g * sin_theta + cos_theta * (-(force / total_mass) + PARAMS::POLE_MASS * pole_half_length * state.theta_dot * state.theta_dot * sin_theta))
            / (pole_half_length * (4.0 / 3.0 - PARAMS::POLE_MASS * cos_theta * cos_theta / total_mass));
        const T x_acc = (force - PARAMS::POLE_MASS * pole_half_length * (theta_acc * cos_theta + state.theta_dot * state.theta_dot * sin_theta)) / total_mass;

        next_state.x = clip(state.x + state.x_dot * dt + 0.5 * x_acc * dt * dt, -PARAMS::MAX_CART_POSITION, PARAMS::MAX_CART_POSITION);
        next_state.x_dot = clip(state.x_dot + x_acc * dt, -PARAMS::MAX_SPEED, PARAMS::MAX_SPEED);
        next_state.theta = state.theta + state.theta_dot * dt + 0.5 * theta_acc * dt * dt;
        next_state.theta_dot = clip(state.theta_dot + theta_acc * dt, -PARAMS::MAX_SPEED, PARAMS::MAX_SPEED);
        return PARAMS::DT;
    }
    template<typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    static typename SPEC::T reward(DEVICE& device, const CartPole<SPEC>& env, const typename CartPole<SPEC>::Parameters& parameters, const typename CartPole<SPEC>::State& state, const Matrix<ACTION_SPEC>& action, const typename CartPole<SPEC>::State& next_state, RNG& rng){
        typedef typename SPEC::T T;
        const T angle = angle_normalize(device.math, next_state.theta);
        const T x = next_state.x;
        const T action_value = get(action, 0, 0);
        const T reward = 1.0 - 0.5 * angle * angle - 0.1 * x * x - 0.01 * action_value * action_value;
        return reward;
    }

    template<typename DEVICE, typename SPEC, typename OBS_TYPE_SPEC, typename OBS_SPEC, typename RNG>
    static void observe(DEVICE& device, const CartPole<SPEC>& env, const typename CartPole<SPEC>::Parameters& parameters, const typename CartPole<SPEC>::State& state, const CartPoleFourierObservation<OBS_TYPE_SPEC, typename SPEC::T>&, Matrix<OBS_SPEC>& observation, RNG& rng){
        static_assert(OBS_SPEC::ROWS == 1);
        static_assert(OBS_SPEC::COLS == 4);
        typedef typename SPEC::T T;
        set(observation, 0, 0, state.x);
        set(observation, 0, 1, state.x_dot);
        set(observation, 0, 2, state.theta);
        set(observation, 0, 3, state.theta_dot);
    }
    template<typename DEVICE, typename SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static bool terminated(DEVICE& device, const CartPole<SPEC>& env, const typename CartPole<SPEC>::Parameters& parameters, const typename CartPole<SPEC>::State state, RNG& rng){
        using T = typename SPEC::T;
        const T angle = angle_normalize(device.math, state.theta);
        return (state.x < -SPEC::PARAMETERS::MAX_CART_POSITION) || (state.x > SPEC::PARAMETERS::MAX_CART_POSITION) || (angle < -0.9 * rlt::math::PI<T>) || (angle > 0.9 * rlt::math::PI<T>);
    }
}