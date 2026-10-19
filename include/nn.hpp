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
    constexpr static T MAX_CART_POSITION = 2.5;
    constexpr static T MAX_SPEED = 1;
    constexpr static T DT = 0.05;
    constexpr static T M = 1;
    constexpr static T L = 1;
    constexpr static T INITIAL_STATE_MIN_ANGLE = -rlt::math::PI<T>;
    constexpr static T INITIAL_STATE_MAX_ANGLE = rlt::math::PI<T>;
    constexpr static T INITIAL_STATE_MIN_SPEED = -1;
    constexpr static T INITIAL_STATE_MAX_SPEED = 1;
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