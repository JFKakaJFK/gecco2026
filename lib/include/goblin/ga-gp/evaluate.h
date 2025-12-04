#pragma once
#ifndef _GOBLIN_GA_GP_EVAL_KERNEL_H
#define _GOBLIN_GA_GP_EVAL_KERNEL_H

#include <vector>

#include "goblin/ga-gp/types.h"

namespace goblin {

#ifdef __CUDACC__
__global__
void evaluate_kernel(
    float* X, 
    float* Y, 
    float* v_type, 
    float* v_value, 
    int solution_length, 
    int num_datapoints,
    float* result
);

__device__
float compute_tree_output(
    float* X, 
    float* type,
    float* value,
    int solution_length,
    int num_datapoints,
    int datapoint_index
);

__global__
void compute_mse_kernel(
    float* se, 
    float* mse, 
    int num_solutions, 
    int num_datapoints
);

__global__
void compute_tree_output_wrapper(
    float* X, 
    float* type,
    float* value,
    int solution_length,
    int num_datapoints,
    int datapoint_index,
    float* result
);
#endif

void evaluate_kernel_wrapper(
    float* X, 
    float* Y, 
    float* type, 
    float* value, 
    int solution_length, 
    int num_solutions,
    int num_datapoints,
    float* se
);

void compute_mse_kernel_wrapper(
    float* se, 
    float* mse, 
    int num_solutions, 
    int num_datapoints
);

float test_compute_output_kernel(
    std::vector<float> h_X,
    std::vector<float> h_type,
    std::vector<float> h_value,
    int num_datapoints,
    int datapoint_index
);

std::vector<float> test_evaluate_kernel(
    std::vector<float> h_X, 
    std::vector<float> h_Y, 
    std::vector<float> h_type, 
    std::vector<float> h_value, 
    int num_solutions,
    int num_datapoints
);

std::vector<float> test_compute_mse_kernel(
    std::vector<float> se, 
    int num_solutions, 
    int num_datapoints
);

}

#endif /* _GOBLIN_GA_GP_EVAL_KERNEL_H */