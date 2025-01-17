/****************************************************************************
 * gemm_bf16_tf32.cu
 *   Compare GEMM performance among float32, tensorfloat32, and bfloat16
 *
 *   - float32       : normal FP32 GEMM
 *   - tensorfloat32 : set cublas TF32 math mode (still uses float* buffers)
 *   - bfloat16      : use __nv_bfloat16 buffers, cublasGemmEx with CUDA_R_16BF
 *
 * Build:
 *   nvcc -std=c++17 -o gemm_bf16_tf32 gemm_bf16_tf32.cu -lcublas
 ****************************************************************************/
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <iostream>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h> // __nv_bfloat16, float2bfloat16, etc.

//---------------------------------------------------------
// Error-check macros
//---------------------------------------------------------
static void checkCuda(cudaError_t err, const char* file, int line){
    if(err!=cudaSuccess){
        std::cerr << "CUDA Error: " << cudaGetErrorString(err)
                  << " at " << file << ":" << line << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
#define CHECK_CUDA(x) checkCuda(x, __FILE__, __LINE__)

static void checkCublas(cublasStatus_t stat, const char* file, int line){
    if(stat!=CUBLAS_STATUS_SUCCESS){
        std::cerr << "cuBLAS Error: code=" << (int)stat
                  << " at " << file << ":" << line << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
#define CHECK_CUBLAS(x) checkCublas(x, __FILE__, __LINE__)

//---------------------------------------------------------
// Device kernel: row->col (float)
//   in:  row-major [M*K]
//   out: col-major [M*K]
//   => out[col*M + row] = in[row*K + col]
//---------------------------------------------------------
__global__
void row2col_f32(const float* __restrict__ in,
                 float* __restrict__ out,
                 int M, int K)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < M*K){
        int row = tid / K;
        int col = tid % K;
        out[col*M + row] = in[tid];
    }
}

//---------------------------------------------------------
// Device kernel: col->row (float)
//   in:  col-major [M*K]
//   out: row-major [M*K]
//---------------------------------------------------------
__global__
void col2row_f32(const float* __restrict__ in,
                 float* __restrict__ out,
                 int M, int K)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < M*K){
        int row = tid / K;
        int col = tid % K;
        out[tid] = in[col*M + row];
    }
}

//---------------------------------------------------------
// Device kernel: row->col (bf16)
//---------------------------------------------------------
__global__
void row2col_bf16(const __nv_bfloat16* __restrict__ in,
                  __nv_bfloat16* __restrict__ out,
                  int M, int K)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < M*K){
        int row = tid / K;
        int col = tid % K;
        out[col*M + row] = in[tid];
    }
}

//---------------------------------------------------------
// Device kernel: col->row (bf16)
//---------------------------------------------------------
__global__
void col2row_bf16(const __nv_bfloat16* __restrict__ in,
                  __nv_bfloat16* __restrict__ out,
                  int M, int K)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < M*K){
        int row = tid / K;
        int col = tid % K;
        out[tid] = in[col*M + row];
    }
}

//---------------------------------------------------------
// Utility: compute GFLOPS = 2*M*N*K / (time_sec * 1e9)
//   ms: time in milliseconds
//---------------------------------------------------------
static float calcGflops(int M, int N, int K, float ms){
    double t_sec = ms * 1e-3;
    double flops = 2.0 * (double)M * (double)N * (double)K;
    double gflops = flops / (t_sec * 1e9);
    return (float)gflops;
}

//---------------------------------------------------------
// Run GEMM in float32
//   - A, B, C: row-major
//   - cublas expects col-major => device-side transpose
//---------------------------------------------------------
float gemm_fp32(cublasHandle_t handle,
                int M, int N, int K,
                const float* dA_row, 
                const float* dB_row,
                float* dC_row,
                int repeat=10)
{
    // (1) Allocate col-major buffers
    float *dA_col=nullptr, *dB_col=nullptr, *dC_col=nullptr;
    CHECK_CUDA( cudaMalloc(&dA_col, sizeof(float)*M*K) );
    CHECK_CUDA( cudaMalloc(&dB_col, sizeof(float)*K*N) );
    CHECK_CUDA( cudaMalloc(&dC_col, sizeof(float)*M*N) );

    // (2) row->col
    {
        dim3 block(256);
        dim3 gridA((M*K + block.x -1)/block.x);
        row2col_f32<<<gridA, block>>>(dA_row, dA_col, M, K);

        dim3 gridB((K*N + block.x -1)/block.x);
        row2col_f32<<<gridB, block>>>(dB_row, dB_col, K, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // (3) Warm-up
    float alpha=1.0f, beta=0.0f;
    CHECK_CUBLAS( cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,  // no transpose
        M, N, K,
        &alpha,
        dA_col, CUDA_R_32F, M,
        dB_col, CUDA_R_32F, K,
        &beta,
        dC_col, CUDA_R_32F, M,
        CUDA_R_32F,                // computeType
        CUBLAS_GEMM_DEFAULT        // algo
    ));
    CHECK_CUDA(cudaDeviceSynchronize());

    // (4) Measure
    cudaEvent_t startEv, stopEv;
    CHECK_CUDA(cudaEventCreate(&startEv));
    CHECK_CUDA(cudaEventCreate(&stopEv));
    CHECK_CUDA(cudaEventRecord(startEv));

    for(int i=0; i<repeat; i++){
        CHECK_CUBLAS( cublasGemmEx(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K,
            &alpha,
            dA_col, CUDA_R_32F, M,
            dB_col, CUDA_R_32F, K,
            &beta,
            dC_col, CUDA_R_32F, M,
            CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT
        ));
    }
    CHECK_CUDA(cudaEventRecord(stopEv));
    CHECK_CUDA(cudaEventSynchronize(stopEv));
    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, startEv, stopEv));
    ms /= repeat;

    // (5) col->row
    {
        dim3 block(256);
        dim3 gridC((M*N + block.x -1)/block.x);
        col2row_f32<<<gridC, block>>>(dC_col, dC_row, M, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // cleanup
    CHECK_CUDA(cudaFree(dA_col));
    CHECK_CUDA(cudaFree(dB_col));
    CHECK_CUDA(cudaFree(dC_col));
    CHECK_CUDA(cudaEventDestroy(startEv));
    CHECK_CUDA(cudaEventDestroy(stopEv));
    return ms;
}

//---------------------------------------------------------
// Run GEMM in tensorfloat32
//   - Almost same as gemm_fp32, but we enable TF32 math mode.
//   - Data buffers are still float*, but internally uses TF32
//---------------------------------------------------------
float gemm_tf32(cublasHandle_t handle,
                int M, int N, int K,
                const float* dA_row,
                const float* dB_row,
                float* dC_row,
                int repeat=10)
{
    // Set TF32 math mode
    CHECK_CUBLAS( cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH) );

    // (1) Allocate col-major buffers
    float *dA_col=nullptr, *dB_col=nullptr, *dC_col=nullptr;
    CHECK_CUDA( cudaMalloc(&dA_col, sizeof(float)*M*K) );
    CHECK_CUDA( cudaMalloc(&dB_col, sizeof(float)*K*N) );
    CHECK_CUDA( cudaMalloc(&dC_col, sizeof(float)*M*N) );

    // (2) row->col
    {
        dim3 block(256);
        dim3 gridA((M*K + block.x -1)/block.x);
        row2col_f32<<<gridA, block>>>(dA_row, dA_col, M, K);

        dim3 gridB((K*N + block.x -1)/block.x);
        row2col_f32<<<gridB, block>>>(dB_row, dB_col, K, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // (3) Warm-up
    float alpha=1.0f, beta=0.0f;
    CHECK_CUBLAS( cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        M, N, K,
        &alpha,
        dA_col, CUDA_R_32F, M,
        dB_col, CUDA_R_32F, K,
        &beta,
        dC_col, CUDA_R_32F, M,
        // computeType = CUDA_R_32F, but cublasSetMathMode(TF32) is active
        CUDA_R_32F,
        CUBLAS_GEMM_ALGO0_TENSOR_OP
    ));
    CHECK_CUDA(cudaDeviceSynchronize());

    // (4) Measure
    cudaEvent_t startEv, stopEv;
    CHECK_CUDA(cudaEventCreate(&startEv));
    CHECK_CUDA(cudaEventCreate(&stopEv));
    CHECK_CUDA(cudaEventRecord(startEv));

    for(int i=0; i<repeat; i++){
        CHECK_CUBLAS( cublasGemmEx(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K,
            &alpha,
            dA_col, CUDA_R_32F, M,
            dB_col, CUDA_R_32F, K,
            &beta,
            dC_col, CUDA_R_32F, M,
            CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ));
    }
    CHECK_CUDA(cudaEventRecord(stopEv));
    CHECK_CUDA(cudaEventSynchronize(stopEv));
    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, startEv, stopEv));
    ms /= repeat;

    // (5) col->row
    {
        dim3 block(256);
        dim3 gridC((M*N + block.x -1)/block.x);
        col2row_f32<<<gridC, block>>>(dC_col, dC_row, M, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // restore math mode if needed
    CHECK_CUBLAS( cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH) );

    // cleanup
    CHECK_CUDA(cudaFree(dA_col));
    CHECK_CUDA(cudaFree(dB_col));
    CHECK_CUDA(cudaFree(dC_col));
    CHECK_CUDA(cudaEventDestroy(startEv));
    CHECK_CUDA(cudaEventDestroy(stopEv));

    return ms;
}

//---------------------------------------------------------
// Run GEMM in BF16
//   - A, B, C are stored in __nv_bfloat16
//   - compute type is FP32
//---------------------------------------------------------
float gemm_bf16(cublasHandle_t handle,
                int M, int N, int K,
                const __nv_bfloat16* dA_row,
                const __nv_bfloat16* dB_row,
                __nv_bfloat16* dC_row,
                int repeat=10)
{
    // (1) Allocate col-major buffers
    __nv_bfloat16 *dA_col=nullptr, *dB_col=nullptr, *dC_col=nullptr;
    CHECK_CUDA( cudaMalloc(&dA_col, sizeof(__nv_bfloat16)*M*K) );
    CHECK_CUDA( cudaMalloc(&dB_col, sizeof(__nv_bfloat16)*K*N) );
    CHECK_CUDA( cudaMalloc(&dC_col, sizeof(__nv_bfloat16)*M*N) );

    // (2) row->col
    {
        dim3 block(256);
        dim3 gridA((M*K + block.x -1)/block.x);
        row2col_bf16<<<gridA, block>>>(dA_row, dA_col, M, K);

        dim3 gridB((K*N + block.x -1)/block.x);
        row2col_bf16<<<gridB, block>>>(dB_row, dB_col, K, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // (3) Warm-up
    float alpha=1.0f, beta=0.0f;
    CHECK_CUBLAS( cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        M, N, K,
        &alpha,
        dA_col, CUDA_R_16BF, M,  // dataType = BF16
        dB_col, CUDA_R_16BF, K,
        &beta,
        dC_col, CUDA_R_16BF, M,
        // computeType = CUBLAS_COMPUTE_32F (== CUDA_R_32F),
        //   but API expects "cudaDataType_t" => use CUDA_R_32F
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP

    ));
    CHECK_CUDA(cudaDeviceSynchronize());

    // (4) Measure
    cudaEvent_t startEv, stopEv;
    CHECK_CUDA(cudaEventCreate(&startEv));
    CHECK_CUDA(cudaEventCreate(&stopEv));
    CHECK_CUDA(cudaEventRecord(startEv));

    for(int i=0; i<repeat; i++){
        CHECK_CUBLAS( cublasGemmEx(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K,
            &alpha,
            dA_col, CUDA_R_16BF, M,
            dB_col, CUDA_R_16BF, K,
            &beta,
            dC_col, CUDA_R_16BF, M,
            CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ));
    }
    CHECK_CUDA(cudaEventRecord(stopEv));
    CHECK_CUDA(cudaEventSynchronize(stopEv));
    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, startEv, stopEv));
    ms /= repeat;

    // (5) col->row
    {
        dim3 block(256);
        dim3 gridC((M*N + block.x -1)/block.x);
        col2row_bf16<<<gridC, block>>>(dC_col, dC_row, M, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // cleanup
    CHECK_CUDA(cudaFree(dA_col));
    CHECK_CUDA(cudaFree(dB_col));
    CHECK_CUDA(cudaFree(dC_col));
    CHECK_CUDA(cudaEventDestroy(startEv));
    CHECK_CUDA(cudaEventDestroy(stopEv));
    return ms;
}

//---------------------------------------------------------
// main
//---------------------------------------------------------
int main()
{
    // create cublas handle
    cublasHandle_t handle;
    CHECK_CUBLAS( cublasCreate(&handle) );

    // Generate random M, N, K
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(128, 512);
    int M = dist(gen);
    int K = dist(gen);
    int N = dist(gen);

    // For demonstration, we pick M,K,N in multiples of 8 to align well
    // but it's optional
    M = M * 64;
    K = K * 64;
    N = N * 64;
    if(M<64) M=64;
    if(K<64) K=64;
    if(N<64) N=64;

    std::cout << "GEMM size: M=" << M << ", K=" << K << ", N=" << N << "\n";

    // allocate host memory
    std::vector<float> hA_fp32(M*K), hB_fp32(K*N), hC_fp32(M*N, 0.f);

    // random initialization
    {
        std::mt19937 rg(1234);
        std::uniform_real_distribution<float> distf(0.f, 1.f);
        for(int i=0; i<M*K; i++){
            hA_fp32[i] = distf(rg);
        }
        for(int i=0; i<K*N; i++){
            hB_fp32[i] = distf(rg);
        }
    }

    // device memory for float32
    float *dA_fp32=nullptr, *dB_fp32=nullptr, *dC_fp32=nullptr;
    CHECK_CUDA( cudaMalloc(&dA_fp32, sizeof(float)*M*K) );
    CHECK_CUDA( cudaMalloc(&dB_fp32, sizeof(float)*K*N) );
    CHECK_CUDA( cudaMalloc(&dC_fp32, sizeof(float)*M*N) );

    CHECK_CUDA( cudaMemcpy(dA_fp32, hA_fp32.data(), sizeof(float)*M*K, cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dB_fp32, hB_fp32.data(), sizeof(float)*K*N, cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemset(dC_fp32, 0, sizeof(float)*M*N) );

    // 1) Run float32 GEMM
    float ms_fp32 = gemm_fp32(handle, M, N, K, dA_fp32, dB_fp32, dC_fp32, 50);
    float gfl_fp32 = calcGflops(M,N,K, ms_fp32);
    std::cout << "[FP32]  " << ms_fp32 << " ms, " << gfl_fp32 << " GFLOPS\n";

    // readback to host (if needed)
    CHECK_CUDA( cudaMemcpy(hC_fp32.data(), dC_fp32, sizeof(float)*M*N, cudaMemcpyDeviceToHost) );

    // 2) Run TF32 GEMM
    //    - same input arrays, but we set TF32 math mode
    CHECK_CUDA( cudaMemset(dC_fp32, 0, sizeof(float)*M*N) );
    float ms_tf32 = gemm_tf32(handle, M, N, K, dA_fp32, dB_fp32, dC_fp32, 50);
    float gfl_tf32 = calcGflops(M,N,K, ms_tf32);
    std::cout << "[TF32]  " << ms_tf32 << " ms, " << gfl_tf32 << " GFLOPS\n";

    // 3) Run BF16 GEMM
    //    - we need __nv_bfloat16 buffers
    // create bf16 host buffers
    std::vector<__nv_bfloat16> hA_bf16(M*K), hB_bf16(K*N), hC_bf16(M*N);
    for(int i=0; i<M*K; i++){
        // float -> bfloat16
        hA_bf16[i] = __float2bfloat16(hA_fp32[i]);
    }
    for(int i=0; i<K*N; i++){
        hB_bf16[i] = __float2bfloat16(hB_fp32[i]);
    }
    for(int i=0; i<M*N; i++){
        hC_bf16[i] = __float2bfloat16(0.f);
    }

    // allocate device bf16
    __nv_bfloat16 *dA_bf16=nullptr, *dB_bf16=nullptr, *dC_bf16=nullptr;
    CHECK_CUDA( cudaMalloc(&dA_bf16, sizeof(__nv_bfloat16)*M*K) );
    CHECK_CUDA( cudaMalloc(&dB_bf16, sizeof(__nv_bfloat16)*K*N) );
    CHECK_CUDA( cudaMalloc(&dC_bf16, sizeof(__nv_bfloat16)*M*N) );

    CHECK_CUDA( cudaMemcpy(dA_bf16, hA_bf16.data(), sizeof(__nv_bfloat16)*M*K, cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dB_bf16, hB_bf16.data(), sizeof(__nv_bfloat16)*K*N, cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dC_bf16, hC_bf16.data(), sizeof(__nv_bfloat16)*M*N, cudaMemcpyHostToDevice) );

    float ms_bf16 = gemm_bf16(handle, M, N, K, dA_bf16, dB_bf16, dC_bf16, 50);
    float gfl_bf16 = calcGflops(M,N,K, ms_bf16);
    std::cout << "[BF16]  " << ms_bf16 << " ms, " << gfl_bf16 << " GFLOPS\n";

    // finalize
    CHECK_CUDA( cudaFree(dA_fp32) );
    CHECK_CUDA( cudaFree(dB_fp32) );
    CHECK_CUDA( cudaFree(dC_fp32) );
    CHECK_CUDA( cudaFree(dA_bf16) );
    CHECK_CUDA( cudaFree(dB_bf16) );
    CHECK_CUDA( cudaFree(dC_bf16) );

    cublasDestroy(handle);
    return 0;
}

