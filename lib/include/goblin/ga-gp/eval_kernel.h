#pragma once
#ifndef _GOBLIN_GP_EVAL_KERNEL_H
#define _GOBLIN_GP_EVAL_KERNEL_H

namespace goblin {

void eval(
    const float* X, 
    const float* Y, 
    const int* v_type, 
    const float* v_value, 
    int solution_length, 
    int num_datapoints,
    float* out
);

float test_eval_kernel(
    std::vector<float> h_X,
    std::vector<float> h_Y,
    std::vector<int> h_type,
    std::vector<float> h_value,
    int num_solutions,
    int num_datapoints,
    std::vector<float>& result
);

}

#endif /* _GOBLIN_GP_EVAL_KERNEL_H */