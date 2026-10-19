#include "nn.hpp"
#include "nn_environment.hpp"

using DEVICE = rlt::devices::DefaultCPU;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using DEVICE_BLAS = rlt::devices::DEVICE_FACTORY<rlt::devices::DefaultCPUSpecification>;
using T = double;
using TI = typename DEVICE::index_t;

struct NeuralNetwork::Impl {
    DEVICE device;
    DEVICE_BLAS device_blas;
    RNG rng;
    TI seed = 1;
    static constexpr TI SIZE = 500;
    rlt::Matrix<rlt::matrix::Specification<T, TI, SIZE, SIZE>> A, B, C, C_blas;
};

NeuralNetwork::NeuralNetwork() : impl(std::make_unique<Impl>()) {}

NeuralNetwork::~NeuralNetwork() = default;

void NeuralNetwork::init() {
    // Allocate matrices and fill with random numbers
    rlt::init(impl->device, impl->rng, impl->seed);
    rlt::malloc(impl->device_blas, impl->A);
    rlt::malloc(impl->device_blas, impl->B);
    rlt::malloc(impl->device_blas, impl->C);
    rlt::malloc(impl->device_blas, impl->C_blas);
    rlt::randn(impl->device, impl->A, impl->rng);
    rlt::randn(impl->device, impl->B, impl->rng);
    
}
