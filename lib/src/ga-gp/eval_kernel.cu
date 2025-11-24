#include <vector>

#include "goblin/ga-gp/helper.h"
#include "goblin/ga-gp/types.h"

// Defines the number of elements in the stack that holds temporary values for the current thread
// Depends on the maximum tree depth (e.g. depth = 2 -> 7 temp values, depth = 3 -> 15 temp values)
#define MAX_STACK_DEPTH 64

namespace goblin {

/* The evaluation kernel handles the evaluation of the solutions over all the data.
 * The grid is 2 dimensional and the blocks are 1 dimensional. 
 * The grids and block dimensions are used as follows:
 * - The block dimension is based on the amount of datapoints.
 * - The x dimension of the grid is for the solutions (e.g. x = 0 is the first solution, x = 1 the second, etc..)
 * - The y dimension of the grid is for the possibility that the number of datapoints is more than the max amount
 *   of threads per block. If that is the case, multiple blocks are assigned to the same solution. (e.g. y = 0 is for 
 *   the first n datapoints, y = 1 for the second n datapoints, where n is the number of threads per block)
 * 
 * Parameters:
 * - X: pointer to the array containing all the datapoints stored in column-major order. 
 * - Y: pointer to the array containing the expected outputs for the data
 * - v_type: pointer to the array containing the types of the solution nodes
 * - v_value: pointer to the array containing the values of the solution nodes
 * - solution_length: max number of elements in a solution
 * - num_inputs: number of inputs (data columns)
 * - num_datapoints: number of datapoints (data rows)
 * 
 * Indexing:
 * - The solutions are stored contiguously in memory. To access a solution we need to know its index, and how much
 *   memory each solution occupies in memory. The first element of the nth solution, where each solution consists of
 *   m nodes is than located at offset n * m. 
 * - The datapoints are stored in column-major order. This means the nth input of the mth datapoint resides at m + n * rows
 * 
 * 
 * Evaluation
 * - The nodes of a solution are stored in a reverse preorder. So the right most child is the first element, and the root 
 *   is the last element. The expression tree should be evaluated bottom up, which can be done by traversing the elements
 *   in order (from left to right).
 * - Numerical values such as constants and inputs are pushed onto the stack when encountered. For operators, the arity
 *   is determined, and that amount of values are popped from the stack. The result of the operation is pushed onto 
 *   the stack again. This is process continues until there are no more nodes to evaluate.
*/

__global__
void eval(
    const float* X, 
    const float* Y, 
    const NodeType* v_type, 
    const float* v_value, 
    int solution_length, 
    int num_datapoints,
    float* out
) {
    // Calculate datapoint index
    int datapoint_index = blockIdx.y * blockDim.x + threadIdx.x;
    // Early exit if thread does not correspond to datapoint (redundant thread)
    if (datapoint_index >= num_datapoints) return;
    
    // Calculate offset for first solution element
    int solution_index = blockIdx.x;
    int solution_offset = solution_index * solution_length;

    // Pointers to first element of solution
    const NodeType* type = v_type + solution_offset;
    const float* value = v_value + solution_offset;

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
            Operator op_value = static_cast<Operator>(static_cast<int>(value[index]));
            
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
                case Operator::Div: res = (args[1] == 0.0f) ? 0.0f : args[0] / args[1]; break;
            }

            stack[sp++] = res;   
        }
    }

    // Final result for solution is on the top of the stack
    float result = stack[--sp];

    __syncthreads();

    out[solution_index * num_datapoints + datapoint_index] = result;
};

void test_eval_kernel(
    std::vector<float> h_X,
    std::vector<float> h_Y,
    std::vector<NodeType> h_type,
    std::vector<float> h_value,
    int num_solutions,
    int num_datapoints,
    std::vector<float>& result
) {
    int X_length = h_X.size();
    int Y_length = h_Y.size();
    // Length of entire population vector
    int population_length = h_type.size();
    // Length of a single solution vector
    int solution_length = h_type.size() / num_solutions;
    
    
    float* d_X;
    float* d_Y;
    NodeType* d_type;
    float* d_value;
    float* d_out;

    // Allocate memory on device
    __CHECK_CUDA_ERR__(cudaMalloc(&d_X, X_length * sizeof(float)));
    __CHECK_CUDA_ERR__(cudaMalloc(&d_Y, Y_length * sizeof(float)));
    __CHECK_CUDA_ERR__(cudaMalloc(&d_type, population_length* sizeof(int)));
    __CHECK_CUDA_ERR__(cudaMalloc(&d_value, population_length * sizeof(float)));
    __CHECK_CUDA_ERR__(cudaMalloc(&d_out, num_solutions * num_datapoints * sizeof(float)));

    // Copy data to the device
    __CHECK_CUDA_ERR__(cudaMemcpy(d_X, h_X.data(), X_length * sizeof(float), cudaMemcpyHostToDevice));
    __CHECK_CUDA_ERR__(cudaMemcpy(d_Y, h_Y.data(), Y_length * sizeof(float), cudaMemcpyHostToDevice));
    __CHECK_CUDA_ERR__(cudaMemcpy(d_type, h_type.data(), population_length * sizeof(int), cudaMemcpyHostToDevice));
    __CHECK_CUDA_ERR__(cudaMemcpy(d_value, h_value.data(), population_length * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(num_datapoints);
    dim3 grid(num_solutions, 1);

    eval<<<grid, block>>>(d_X, d_Y, d_type, d_value, solution_length, num_datapoints, d_out);

    __CHECK_CUDA_ERR__(cudaGetLastError());
    __CHECK_CUDA_ERR__(cudaDeviceSynchronize());

    __CHECK_CUDA_ERR__(cudaMemcpy(result.data(), d_out, num_solutions * num_datapoints * sizeof(float), cudaMemcpyDeviceToHost));

    __CHECK_CUDA_ERR__(cudaFree(d_type));
    __CHECK_CUDA_ERR__(cudaFree(d_value));
    __CHECK_CUDA_ERR__(cudaFree(d_out));
    
    return;
}

};