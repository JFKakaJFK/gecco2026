#include <vector>

#include "cub/cub.cuh"

#include "goblin/ga-gp/evaluate.h"
#include "goblin/ga-gp/helper.h"
#include "goblin/ga-gp/misc.h"
#include "goblin/ga-gp/types.h"

#define __CHECK_CUDA_ERR__(err) check((err), #err, __FILE__, __LINE__)

#define BLOCK_SIZE_LIST \
    X(32)  X(64)  X(96)  X(128) X(160) X(192) X(224) X(256) \
    X(288) X(320) X(352) X(384) X(416) X(448) X(480) X(512) \
    X(544) X(576) X(608) X(640) X(672) X(704) X(736) X(768) \
    X(800) X(832) X(864) X(896) X(928) X(960) X(992) X(1024)

#define MAX_STACK_DEPTH 64
#define MAX_NUM_NODES 64

namespace goblin {

__global__
void evaluate_kernel_baseline(
    float* X, 
    float* Y, 
    float* v_type, 
    float* v_value, 
    float* result,
    int solution_length, 
    int num_datapoints
) {
    // Calculate datapoint index
    int datapoint_index = blockIdx.y * blockDim.x + threadIdx.x;
    int solution_index = blockIdx.x;

    if (datapoint_index < num_datapoints) {
        // Calculate offset for first element of solution
        int solution_offset = solution_index * solution_length;

        // Pointers to first element of solution
        float* type = v_type + solution_offset;
        float* value = v_value + solution_offset;

        // Compute output of solution
        float output = compute_tree_output_baseline(
            X, type, value, 
            solution_length, 
            num_datapoints, 
            datapoint_index
        );

        // Determine squared error
        float error = output - Y[datapoint_index];
        float se = error * error;

        // Store squared error in global memory
        result[solution_index * num_datapoints + datapoint_index] = se;
    }
};

__global__
void evaluate_kernel_restrict(
    const float* __restrict__ X, 
    const float* __restrict__ Y, 
    const float* __restrict__ v_type, 
    const float* __restrict__ v_value, 
    float* __restrict__ result,
    int solution_length, 
    int num_datapoints
) {
    // Calculate datapoint index
    int datapoint_index = blockIdx.y * blockDim.x + threadIdx.x;
    int solution_index = blockIdx.x;

    if (datapoint_index < num_datapoints) {
        // Calculate offset for first element of solution
        int solution_offset = solution_index * solution_length;

        // Pointers to first element of solution
        const float* type = v_type + solution_offset;
        const float* value = v_value + solution_offset;

        // Compute output of solution
        float output = compute_tree_output_restrict(
            X, type, value, 
            solution_length, 
            num_datapoints, 
            datapoint_index
        );

        // Determine squared error
        float error = output - Y[datapoint_index];
        float se = error * error;

        // Store squared error in global memory
        result[solution_index * num_datapoints + datapoint_index] = se;
    }
};


__global__
void evaluate_kernel_shared_memory(
    const float* __restrict__ X, 
    const float* __restrict__ Y, 
    const float* __restrict__ v_type, 
    const float* __restrict__ v_value, 
    float* __restrict__ result,
    int solution_length, 
    int num_datapoints
) {
    // Calculate datapoint index
    int datapoint_index = blockIdx.y * blockDim.x + threadIdx.x;
    int solution_index = blockIdx.x;

    // Calculate offset for solution
    int solution_offset = solution_index * solution_length;

    // Layout shared memory
    __shared__ float shmem[MAX_NUM_NODES * 2];
    float* sh_type = (float*)shmem;
    float* sh_value = (float*)(sh_type + solution_length);

    // Cooperative load of solution data into shared memory
    for (int i = threadIdx.x; i < solution_length; i+= blockDim.x) {
        sh_type[i] = v_type[solution_offset + i];
        sh_value[i] = v_value[solution_offset + i];
    }

    // Ensure all solution data has been copied into shared memory
    __syncthreads();

    if (datapoint_index < num_datapoints) {
        // Compute output of solution
        float output = compute_tree_output_restrict(
            X, sh_type, sh_value, 
            solution_length, 
            num_datapoints, 
            datapoint_index
        );

        // Determine squared error
        float error = output - Y[datapoint_index];
        float se = error * error;

        // Store squared error in global memory
        result[solution_index * num_datapoints + datapoint_index] = se;
    }
};

template <int BLOCK_SIZE>
__global__
void evaluate_kernel_block_reduce(
    const float* __restrict__ X, 
    const float* __restrict__ Y, 
    const float* __restrict__ v_type, 
    const float* __restrict__ v_value, 
    float* __restrict__ partial,
    int solution_length, 
    int num_datapoints
) {
    // Calculate datapoint index
    // blockIdx.y = block index of solution
    // blockDim.x = number of threads in a block
    int datapoint_index = blockIdx.y * blockDim.x + threadIdx.x;
    int solution_index = blockIdx.x;

    // Calculate offset for solution
    int solution_offset = solution_index * solution_length;

    // Layout shared memory
    __shared__ float shmem[MAX_NUM_NODES * 2];
    float* sh_type = (float*)shmem;
    float* sh_value = (float*)(sh_type + solution_length);

    // Cooperative load of solution data into shared memory
    for (int i = threadIdx.x; i < solution_length; i+= blockDim.x) {
        sh_type[i] = v_type[solution_offset + i];
        sh_value[i] = v_value[solution_offset + i];
    }

    __syncthreads();

    float se = 0.0f;

    // Early exit if thread does not correspond to datapoint
    if (datapoint_index < num_datapoints) {
        // Compute output of solution
        float output = compute_tree_output_restrict(
            X, sh_type, sh_value, 
            solution_length, 
            num_datapoints, 
            datapoint_index
        );

        // Determine squared error
        float error = output - Y[datapoint_index];
        se = error * error;
    }

    __syncthreads();

    using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE, cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    float block_sum = BlockReduce(temp_storage).Sum(se);

    __syncthreads();

    if (threadIdx.x == 0) {
        // gridDim.y = Number of partial results per solution
        // blockIdx.y = Partial result index of current solution
        partial[gridDim.y * solution_index + blockIdx.y] = block_sum;
    }
}

// https://nvidia.github.io/cccl/cub/index.html#flexible-data-arrangement
template <int BLOCK_THREADS>
__global__
void evaluate_mse_kernel(
    const float* __restrict__ X, 
    const float* __restrict__ Y, 
    const float* __restrict__ v_type, 
    const float* __restrict__ v_value, 
    float* __restrict__ result,
    const int solution_length, 
    const int num_datapoints,
    const int datapoints_per_thread
) {
    using BlockReduce = cub::BlockReduce<float, BLOCK_THREADS, cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int solution_index = blockIdx.x;

    // Calculate offset for solution
    int solution_offset = solution_index * solution_length;

    // Layout shared memory
    __shared__ float shmem[MAX_NUM_NODES * 2];
    float* sh_type = (float*)shmem;
    float* sh_value = (float*)(sh_type + solution_length);

    // Cooperative load of solution data into shared memory
    for (int i = threadIdx.x; i < solution_length; i+= blockDim.x) {
        sh_type[i] = v_type[solution_offset + i];
        sh_value[i] = v_value[solution_offset + i];
    }

    __syncthreads();

    float se = 0.0f;

    for (int i = 0; i < datapoints_per_thread; i++) {
        // Calculate datapoint index
        int datapoint_index = i * blockDim.x + threadIdx.x;

        // Check if datapoint_index corresponds to actual datapoint
        if (datapoint_index < num_datapoints) {
            // Compute output of solution
            float output = compute_tree_output_restrict(
                X, sh_type, sh_value, 
                solution_length, 
                num_datapoints, 
                datapoint_index
            );

            // Determine squared error
            float error = output - Y[datapoint_index];
            se += error * error;
        }
    }

    float sum = BlockReduce(temp_storage).Sum(se);

    if (threadIdx.x == 0) {
        result[solution_index] = sum / num_datapoints;
    }
}

template <int BLOCK_THREADS>
__global__
void evaluate_mse_kernel_fmaf(
    const float* __restrict__ X, 
    const float* __restrict__ Y, 
    const float* __restrict__ v_type, 
    const float* __restrict__ v_value, 
    float* __restrict__ result,
    const int solution_length, 
    const int num_datapoints,
    const int datapoints_per_thread
) {
    using BlockReduce = cub::BlockReduce<float, BLOCK_THREADS, cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int solution_index = blockIdx.x;

    // Calculate offset for solution
    int solution_offset = solution_index * solution_length;

    // Layout shared memory
    __shared__ float shmem[MAX_NUM_NODES * 2];
    float* sh_type = (float*)shmem;
    float* sh_value = (float*)(sh_type + solution_length);

    // Cooperative load of solution data into shared memory
    for (int i = threadIdx.x; i < solution_length; i+= blockDim.x) {
        sh_type[i] = v_type[solution_offset + i];
        sh_value[i] = v_value[solution_offset + i];
    }

    __syncthreads();

    float se = 0.0f;

    for (int i = 0; i < datapoints_per_thread; i++) {
        // Calculate datapoint index
        int datapoint_index = i * blockDim.x + threadIdx.x;

        // Check if datapoint_index corresponds to actual datapoint
        if (datapoint_index < num_datapoints) {
            // Compute output of solution
            float output = compute_tree_output_restrict(
                X, sh_type, sh_value, 
                solution_length, 
                num_datapoints, 
                datapoint_index
            );

            // Determine squared error
            float error = output - Y[datapoint_index];
            // se += error * error;
            se = __fmaf_rn(error, error, se); // TODO check SASS code
        }
    }

    float sum = BlockReduce(temp_storage).Sum(se);

    if (threadIdx.x == 0) {
        result[solution_index] = sum / num_datapoints;
    }
}

__device__
float compute_tree_output_baseline(
    float* X, 
    float* type,
    float* value,
    int solution_length,
    int num_datapoints,
    int datapoint_index
) {
    // Evaluation stack (per thread)
    float stack[MAX_STACK_DEPTH];
    int sp = 0;

    // Traverse through solution from left to right
    for (int index = 0; index < solution_length; index++) {
        // Get type of current element
        NodeType t = static_cast<NodeType>(type[index]);

        if (t == NodeType::Input) {
            int input_index = int(value[index]);
            // Push input variable onto stack and increase stack pointer
            stack[sp++] = X[datapoint_index + input_index * num_datapoints];
        } else if (t == NodeType::Constant) {
            // Push constant value onto stack and increase stack pointer
            stack[sp++] = value[index];
        } else if (t == NodeType::Operator) { // ValueKind::Operator
            // TODO improve
            Operator op_value = static_cast<Operator>(value[index]);
            
            // Get operands from stack depending on arity of operator
            int arity = 2; // Currently only arity of 2 is supported
            float args[2];
            for (int j = 0; j < arity; j++) {
                args[j] = stack[--sp];
            }

            float res = 0.0f;

            switch (op_value) {
                case Operator::Add: res = args[0] + args[1]; break;
                case Operator::Sub: res = args[0] - args[1]; break;
                case Operator::Mul: res = args[0] * args[1]; break;
                case Operator::Div: res = args[0] / args[1]; break;
            }

            stack[sp++] = res;   
        } else {
            break;
        }
    }

    // Final result for solution is on the top of the stack
    float result = stack[--sp];

    return result;
}

__device__
float compute_tree_output_restrict(
    const float* __restrict__ X, 
    const float* __restrict__ type,
    const float* __restrict__ value,
    int solution_length,
    int num_datapoints,
    int datapoint_index
) {
    // Evaluation stack (per thread)
    float stack[MAX_STACK_DEPTH];
    int sp = 0;

    // Traverse through solution from left to right
    for (int index = 0; index < solution_length; index++) {
        // Get type of current element
        NodeType t = static_cast<NodeType>(type[index]);

        if (t == NodeType::Input) {
            int input_index = int(value[index]);
            // Push input variable onto stack and increase stack pointer
            stack[sp++] = X[datapoint_index + input_index * num_datapoints];
        } else if (t == NodeType::Constant) {
            // Push constant value onto stack and increase stack pointer
            stack[sp++] = value[index];
        } else if (t == NodeType::Operator) { // ValueKind::Operator
            // TODO improve
            Operator op_value = static_cast<Operator>(value[index]);
            
            // Get operands from stack depending on arity of operator
            int arity = 2; // Currently only arity of 2 is supported
            float args[2];
            for (int j = 0; j < arity; j++) {
                args[j] = stack[--sp];
            }

            float res = 0.0f;

            switch (op_value) {
                case Operator::Add: res = args[0] + args[1]; break;
                case Operator::Sub: res = args[0] - args[1]; break;
                case Operator::Mul: res = args[0] * args[1]; break;
                case Operator::Div: res = args[0] / args[1]; break;
            }

            stack[sp++] = res;   
        } else {
            break;
        }
    }

    // Final result for solution is on the top of the stack
    float result = stack[--sp];

    return result;
}

__global__
void mse_kernel_baseline(float* partial, float* result, int num_solutions, int num_datapoints) {
    int solution_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (solution_index < num_solutions) {
        int start = solution_index * num_datapoints; 
        int end = start + num_datapoints - 1;

        float sum = 0.0f;

        // Loop through datapoints and sum squared errors
        for (int i = start; i <= end; i++) {
            sum += partial[i];
        }

        // Divide by num_datapoints to get the mean
        result[solution_index] = sum / num_datapoints;
    }
}

__global__
void mse_kernel_restrict(
    const float* __restrict__ partial, 
    float* __restrict__ result, 
    int num_solutions, 
    int num_datapoints
) {
    int solution_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (solution_index < num_solutions) {
        int start = solution_index * num_datapoints; 
        int end = start + num_datapoints - 1;

        float sum = 0.0f;

        // Loop through datapoints and sum squared errors
        for (int i = start; i <= end; i++) {
            sum += partial[i];
        }

        result[solution_index] = sum / num_datapoints;;
    }
}

// TODO add items_per_thread template
// https://nvidia.github.io/cccl/cub/index.html#flexible-data-arrangement
template <int BLOCK_THREADS, int ITEMS_PER_THREAD = 1>
__global__
void mse_kernel_block_reduce(
    const float* __restrict__ partial, 
    float* __restrict__ result,
    int num_partial,
    int num_datapoints
) {
    // Specialize BlockLoad and BlockReduce collective types
    using BlockLoad = cub::BlockLoad<float, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_DIRECT>;
    using BlockReduce = cub::BlockReduce<float, BLOCK_THREADS, cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>;

    // Allocate type-safe, repurposable shared memory for collectives
    __shared__ union {
        typename BlockLoad::TempStorage load;
        typename BlockReduce::TempStorage reduce;
    } temp_storage;

    int solution_index = blockIdx.x;
    float thread_data[ITEMS_PER_THREAD];
    int block_offset = solution_index * num_partial;

    // Use a block load to load the partial results
    BlockLoad(temp_storage.load).Load(partial + block_offset, thread_data, num_partial, 0);

    __syncthreads();

    float sum = BlockReduce(temp_storage.reduce).Sum(thread_data);

    __syncthreads();

    if (threadIdx.x == 0) {
        result[solution_index] = sum / num_datapoints;
    }
}

void evaluate_kernel_wrapper(
    float* X, 
    float* Y, 
    float* type, 
    float* value, 
    float* partial,
    const LaunchConfig config
) {
    dim3 block(config.eval.block.x, config.eval.block.y, config.eval.block.z);
    dim3 grid(config.eval.grid.x, config.eval.grid.y, config.eval.grid.z);

    // Launch evaluate kernel that calculates the squared error for every solution and datapoint combination
    switch (config.kernel_version) {
        case (KernelVersion::Baseline):
            evaluate_kernel_baseline<<<grid, block>>>(X, Y, type, value, partial, config.solution_length, config.num_datapoints);
            break;
        case (KernelVersion::Restrict):
            evaluate_kernel_restrict<<<grid, block>>>(X, Y, type, value, partial, config.solution_length, config.num_datapoints);
            break;
        case (KernelVersion::SharedMemory):
            evaluate_kernel_shared_memory<<<grid, block>>>(X, Y, type, value, partial, config.solution_length, config.num_datapoints);
            break;
        case (KernelVersion::BlockReduce):
            switch (block.x) {
                #define X(BS) \
                case BS: \
                    evaluate_kernel_block_reduce<BS><<<grid, block>>>( \
                        X, Y, type, value, partial, config.solution_length, config.num_datapoints); \
                    break;
                BLOCK_SIZE_LIST
                #undef X

                default: 
                    assert(false && "Unsupported block size");
            }
            break;
        default:
            break;
    }
    
    __CHECK_CUDA_ERR__(cudaGetLastError());

    // Wait until all blocks and threads are done
    __CHECK_CUDA_ERR__(cudaDeviceSynchronize());
}

void mse_kernel_wrapper(
    float* partial, 
    float* result, 
    const LaunchConfig config
) {
    dim3 block(config.mse.block.x, config.mse.block.y, config.mse.block.z);
    dim3 grid(config.mse.grid.x, config.mse.grid.y, config.mse.grid.z);

    int num_partials = config.eval.grid.y;
    
    switch(config.kernel_version) {
        case (KernelVersion::Baseline):
            mse_kernel_baseline<<<grid, block>>>(partial, result, config.num_solutions, config.num_datapoints);
            break;
        case (KernelVersion::Restrict):
        case (KernelVersion::SharedMemory):
            mse_kernel_restrict<<<grid, block>>>(partial, result, config.num_solutions, config.num_datapoints);  
            break;
        case (KernelVersion::BlockReduce):
            switch (config.mse.block.x) {
                #define X(BS) \
                case BS: \
                    mse_kernel_block_reduce<BS><<<grid, block>>>( \
                        partial, result, num_partials, config.num_datapoints); \
                    break;
                BLOCK_SIZE_LIST
                #undef X

                default: 
                    assert(false && "Unsupported block size");
            }   
            break;
        default:
            break;
    }

    __CHECK_CUDA_ERR__(cudaGetLastError());

    // Wait until all blocks and threads are done
    __CHECK_CUDA_ERR__(cudaDeviceSynchronize());
}

void evaluate_mse_kernel_wrapper(
    float* X, 
    float* Y, 
    float* type, 
    float* value, 
    float* result,
    const LaunchConfig config
) {
    dim3 block(config.eval.block.x, config.eval.block.y, config.eval.block.z);
    dim3 grid(config.eval.grid.x, config.eval.grid.y, config.eval.grid.z);

    switch(config.kernel_version) {
        case (KernelVersion::SingleKernel):
            switch (block.x) {
                #define X(BS) \
                case BS: \
                    evaluate_mse_kernel<BS><<<grid, block>>>( \
                        X, Y, type, value, result, config.solution_length, config.num_datapoints, config.items_per_thread); \
                    break;
                BLOCK_SIZE_LIST
                #undef X

                default: 
                    assert(false && "Unsupported block size");
            }
            break;
        case (KernelVersion::SingleKernelFMAF):
            switch (block.x) {
                #define X(BS) \
                case BS: \
                    evaluate_mse_kernel_fmaf<BS><<<grid, block>>>( \
                        X, Y, type, value, result, config.solution_length, config.num_datapoints, config.items_per_thread); \
                    break;
                BLOCK_SIZE_LIST
                #undef X

                default: 
                    assert(false && "Unsupported block size");
            }
            break;
        default:
            break;
    }

    __CHECK_CUDA_ERR__(cudaGetLastError());

    // Wait until all blocks and threads are done
    __CHECK_CUDA_ERR__(cudaDeviceSynchronize());
}

void kernel_wrapper(
    float* X, 
    float* Y, 
    float* type, 
    float* value, 
    float* partial,
    float* result,
    const LaunchConfig config
) {
    if (config.kernel_version == KernelVersion::SingleKernel || config.kernel_version == KernelVersion::SingleKernelFMAF) {
        evaluate_mse_kernel_wrapper(X, Y, type, value, result, config);
    } else {
        // Launch evaluate kernel that calculates the squared error for every solution and datapoint combination
        evaluate_kernel_wrapper(X, Y, type, value, partial, config);

        // Launch mse kernel that calculates the mse over all datapoints for each solution
        mse_kernel_wrapper(partial, result, config);
    }
}

/* 
 * The wrapper function below is only required to test the compute_tree_output kernel.
 * The compute_tree_output kernel is defined as a device-only function, so it cannot be 
 * called from functions running on the host. However, the wrapper can be called from the device,
 * and can call device-only functions.
*/
__global__
void compute_tree_output_wrapper(
    float* X, 
    float* type,
    float* value,
    float* result,
    int solution_length,
    int num_datapoints,
    int datapoint_index
) {
    result[0] = compute_tree_output_restrict(
        X, 
        type, 
        value, 
        solution_length, 
        num_datapoints, 
        datapoint_index
    );
}

float test_compute_output_kernel(
    std::vector<float> h_X,
    std::vector<float> h_type,
    std::vector<float> h_value,
    int num_datapoints,
    int datapoint_index
) {
    // Length of a single solution vector
    int solution_length = h_type.size();

    // Allocate memory and copy data
    float* d_X = allocate_and_copy(h_X.data(), h_X.size());
    float* d_type = allocate_and_copy(h_type.data(), solution_length);
    float* d_value = allocate_and_copy(h_value.data(), solution_length);

    // Allocate memory
    float* d_result = allocate_on_gpu<float>(1);

    dim3 block(1);
    dim3 grid(1, 1);

    compute_tree_output_wrapper<<<grid, block>>>(d_X, d_type, d_value, d_result, solution_length, num_datapoints, datapoint_index);

    __CHECK_CUDA_ERR__(cudaGetLastError());
    __CHECK_CUDA_ERR__(cudaDeviceSynchronize());

    float result;
    __CHECK_CUDA_ERR__(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost));

    free_on_gpu(d_type);
    free_on_gpu(d_value);
    free_on_gpu(d_result);
    
    return result;
}

std::vector<float> test_evaluate_kernel(
    std::vector<float> h_X, 
    std::vector<float> h_Y, 
    std::vector<float> h_type, 
    std::vector<float> h_value, 
    int num_solutions,
    int num_datapoints,
    KernelVersion version
) {
    int solution_length = h_type.size() / num_solutions;

    // Allocate memory and copy data
    float* d_X = allocate_and_copy(h_X.data(), h_X.size());
    float* d_Y = allocate_and_copy(h_Y.data(), h_Y.size());
    float* d_type = allocate_and_copy(h_type.data(), h_type.size());
    float* d_value = allocate_and_copy(h_value.data(), h_value.size());

    int partial_size;

    if (version == KernelVersion::BlockReduce) {
        partial_size = num_solutions;
    } else {
        partial_size = num_solutions * num_datapoints;
    }

    float* d_partial = allocate_on_gpu<float>(partial_size);    

    LaunchConfig config = LaunchConfig::determine(version, num_solutions, num_datapoints, solution_length);

    evaluate_kernel_wrapper(d_X, d_Y, d_type, d_value, d_partial, config);

    std::vector<float> result(partial_size);
    __CHECK_CUDA_ERR__(cudaMemcpy(result.data(), d_partial, partial_size * sizeof(float), cudaMemcpyDeviceToHost));

    free_on_gpu(d_type);
    free_on_gpu(d_value);
    free_on_gpu(d_partial);

    return result;
}

std::vector<float> test_compute_mse_kernel(
    std::vector<float> partial, 
    int num_solutions, 
    int num_datapoints, 
    KernelVersion version
) {
    // Allocate memory and copy data
    float* d_partial = allocate_and_copy(partial.data(), partial.size());

    // Allocate memory
    float* d_result = allocate_on_gpu<float>(num_solutions);

    LaunchConfig config = LaunchConfig::determine(version, num_solutions, num_datapoints, 0);

    mse_kernel_wrapper(d_partial, d_result, config);

    std::vector<float> result(num_solutions);
    __CHECK_CUDA_ERR__(cudaMemcpy(result.data(), d_result, num_solutions * sizeof(float), cudaMemcpyDeviceToHost));

    free_on_gpu(d_partial);
    free_on_gpu(d_result);

    return result;
}

std::vector<float> test_evaluate_mse_kernel(
    std::vector<float> h_X, 
    std::vector<float> h_Y, 
    std::vector<float> h_type, 
    std::vector<float> h_value, 
    int num_solutions,
    int num_datapoints
) {
    int solution_length = h_type.size() / num_solutions;

    // Allocate memory and copy data
    float* d_X = allocate_and_copy(h_X.data(), h_X.size());
    float* d_Y = allocate_and_copy(h_Y.data(), h_Y.size());
    float* d_type = allocate_and_copy(h_type.data(), h_type.size());
    float* d_value = allocate_and_copy(h_value.data(), h_value.size());

    float* d_result = allocate_on_gpu<float>(num_solutions);    

    LaunchConfig config = LaunchConfig::determine(KernelVersion::SingleKernel, num_solutions, num_datapoints, solution_length);

    evaluate_mse_kernel_wrapper(d_X, d_Y, d_type, d_value, d_result, config);

    std::vector<float> result(num_solutions);
    __CHECK_CUDA_ERR__(cudaMemcpy(result.data(), d_result, num_solutions * sizeof(float), cudaMemcpyDeviceToHost));

    free_on_gpu(d_type);
    free_on_gpu(d_value);
    free_on_gpu(d_result);

    return result;
}

};