#include "goblin/ga-gp/cuda_graph.h"
#include "goblin/ga-gp/evaluate.h"
#include "goblin/ga-gp/misc.h"

namespace goblin {

struct CudaGraph {
    cudaGraph_t graph{};
    cudaGraphExec_t exec{};
    cudaStream_t stream{};
};

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
) {
    auto* g = new CudaGraph{};

    cudaStreamCreate(&g->stream);
    cudaGraphCreate(&g->graph, 0);

    std::vector<cudaGraphNode_t> deps;
    cudaGraphNode_t memcpyNode, kernelNode;

    cudaKernelNodeParams kernelParams = {0};
    cudaMemcpy3DParms memcpyParams = {0};

    const int input_size = config.num_solutions * config.solution_length;
    
    // ----------------------------------- //
    // Host-to-Device Type Node Parameters //
    // ----------------------------------- //
    // memcpyParams.srcArray = NULL;
    // memcpyParams.srcPos = make_cudaPos(0, 0, 0);
    memcpyParams.srcPtr = make_cudaPitchedPtr(
        h_type, input_size * sizeof(float), input_size, 1
    );
    // memcpyParams.dstArray = NULL;
    // memcpyParams.dstPos = make_cudaPos(0, 0, 0);
    memcpyParams.dstPtr = make_cudaPitchedPtr(
        d_type, input_size * sizeof(float), input_size, 1
    );
    memcpyParams.kind = cudaMemcpyHostToDevice;

    cudaGraphAddMemcpyNode(&memcpyNode, g->graph, NULL, 0, &memcpyParams);
    deps.push_back(memcpyNode);

    // ------------------------------------ //
    // Host-to-Device Value Node Parameters //
    // ------------------------------------ //
    // memcpyParams.srcArray = NULL;
    // memcpyParams.srcPos = make_cudaPos(0, 0, 0);
    memcpyParams.srcPtr = make_cudaPitchedPtr(
        h_value, input_size * sizeof(float), input_size, 1
    );
    // memcpyParams.dstArray = NULL;
    // memcpyParams.dstPos = make_cudaPos(0, 0, 0);
    memcpyParams.dstPtr = make_cudaPitchedPtr(
        d_value, input_size * sizeof(float), input_size, 1
    );
    memcpyParams.kind = cudaMemcpyHostToDevice;

    cudaGraphAddMemcpyNode(&memcpyNode, g->graph, NULL, 0, &memcpyParams);
    deps.push_back(memcpyNode);
    
    // ---------------------- //
    // Kernel Node Parameters //
    // ---------------------- //
    void* kernel_args[] = {
        &d_X,
        &d_Y,
        &d_type,
        &d_value,
        &d_result,
        (void *)&config
    };

    kernelParams.func = (void *)evaluate_mse_kernel_wrapper;
    kernelParams.gridDim = dim3(config.eval.grid.x, config.eval.grid.y, config.eval.grid.z);
    kernelParams.blockDim = dim3(config.eval.block.x, config.eval.block.y, config.eval.block.z);
    // kernel_node_params.sharedMemBytes = 0; // TODO check if required
    kernelParams.kernelParams = kernel_args;
    kernelParams.extra = NULL;

    cudaGraphAddKernelNode(&kernelNode, g->graph, deps.data(), deps.size(), &kernelParams);

    deps.clear();
    deps.push_back(kernelNode);

    // ------------------------------------ //
    // Device-to-Host Result Node Parameters //
    // ------------------------------------ //
    // memcpyParams.srcArray = NULL;
    // memcpyParams.srcPos = make_cudaPos(0, 0, 0);
    memcpyParams.srcPtr = make_cudaPitchedPtr(d_result, config.num_solutions * sizeof(float), config.num_solutions, 1);
    // memcpyParams.dstArray = NULL;
    // memcpyParams.dstPos = make_cudaPos(0, 0, 0);
    memcpyParams.dstPtr = make_cudaPitchedPtr(h_result, config.num_solutions * sizeof(float), config.num_solutions, 1);
    memcpyParams.kind = cudaMemcpyDeviceToHost;

    cudaGraphAddMemcpyNode(&memcpyNode, g->graph, deps.data(), deps.size(), &memcpyParams);

    cudaGraphInstantiate(&g->exec, g->graph, NULL, NULL, 0);

    return reinterpret_cast<CudaGraphHandle>(g);
};

void launch_cuda_graph(CudaGraphHandle h) {
    auto* g = reinterpret_cast<CudaGraph*>(h);
    cudaGraphLaunch(g->exec, g->stream);
    cudaStreamSynchronize(g->stream);
}

void destroy_cuda_graph(CudaGraphHandle h) {
    // if (!g) return;

    // cudaGraphExecDestroy(g->exec);
    // cudaGraphDestroy(g->graph);
    // cudaStreamDestroy(g->stream);
    // delete g;
    delete reinterpret_cast<CudaGraph*>(h);
}

}