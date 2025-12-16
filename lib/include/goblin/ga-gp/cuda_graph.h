#pragma once
#ifndef __GOBLIN_GA_GP_CUDA_GRAPH_H

#include <cstdint>

#include "goblin/ga-gp/misc.h"

namespace goblin {

using CudaGraphHandle = std::uintptr_t;

CudaGraphHandle create_cuda_graph(
    float* h_type,
    float* h_value,
    float* h_result,
    float* d_X,
    float* d_Y,
    float* d_type,
    float* d_value,
    float* d_result,
    const LaunchConfig config
);

void launch_cuda_graph(CudaGraphHandle handle);

void destroy_cuda_graph(CudaGraphHandle handle);

}

#endif /* __GOBLIN_GA_GP_CUDA_GRAPH_H */