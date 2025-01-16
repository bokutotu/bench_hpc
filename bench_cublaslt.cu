// nvcc -std=c++17 -o bench_cublaslt_vs_cublas bench_cublaslt_vs_cublas.cu \
//     -lcublasLt -lcublas

#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cstdlib>

// ===== エラーチェックマクロ =========================================
static void checkCuda(cudaError_t status, const char* file, int line) {
    if (status != cudaSuccess) {
        std::cerr << "[CUDA Error] " << cudaGetErrorString(status)
                  << " at " << file << ":" << line << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
#define CHECK_CUDA(stat) checkCuda(stat, __FILE__, __LINE__)

static void checkCublasLt(cublasStatus_t status, const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "[cuBLASLt Error] code=" << (int)status
                  << " at " << file << ":" << line << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
#define CHECK_CUBLASLT(stat) checkCublasLt(stat, __FILE__, __LINE__)

static void checkCublas(cublasStatus_t status, const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "[cuBLAS Error] code=" << (int)status
                  << " at " << file << ":" << line << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
#define CHECK_CUBLAS(stat) checkCublas(stat, __FILE__, __LINE__)

// ===== デバイス関数: row-major -> col-major のコピー ================
//  - dA_row: (M*K) row-major 配列
//  - dA_col: (M*K) col-major 配列 (leading dimension = M)
//  - M行, K列 (row-major)
__global__
void copyRowMajorToColMajorKernel(const float* __restrict__ dA_row,
                                  float* __restrict__ dA_col,
                                  int M, int K)
{
    // (row, col) in row-major => idx_row = row*K + col
    // col-major => idx_col = col*M + row
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < M*K) {
        int row = tid / K;   // integer div
        int col = tid % K;
        int idx_row = row*K + col;
        int idx_col = col*M + row;
        dA_col[idx_col] = dA_row[idx_row];
    }
}

// col-major -> row-major
__global__
void copyColMajorToRowMajorKernel(const float* __restrict__ dA_col,
                                  float* __restrict__ dA_row,
                                  int M, int K)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < M*K) {
        int row = tid / K;
        int col = tid % K;
        int idx_row = row*K + col;
        int idx_col = col*M + row;
        dA_row[idx_row] = dA_col[idx_col];
    }
}

// ======== ベンチマーク: cuBLASLt (row-major) ==========================
// - row-majorの A(M*K), B(K*N), C(M*N) をそのまま使う
// - workspace を maxWsBytes で確保し、 cublasLtMatmul() 実行時間を計測
// - 失敗(アルゴリズム無し)なら -1
float benchGemmCublasLt(
    cublasLtHandle_t ltHandle,
    int M, int N, int K,
    const float* dA, int lda,
    const float* dB, int ldb,
    float* dC, int ldc,
    size_t maxWsBytes,
    int repeat=5
) {
    // 1) MatmulDesc
    cublasLtMatmulDesc_t opDesc = nullptr;
    CHECK_CUBLASLT( cublasLtMatmulDescCreate(&opDesc,
                                             CUBLAS_COMPUTE_32F,
                                             CUDA_R_32F) );

    // 転置なし
    cublasOperation_t opA = CUBLAS_OP_N;
    cublasOperation_t opB = CUBLAS_OP_N;
    CHECK_CUBLASLT( cublasLtMatmulDescSetAttribute(
        opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
    CHECK_CUBLASLT( cublasLtMatmulDescSetAttribute(
        opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));

    // 2) MatrixLayout (row-major)
    cublasLtMatrixLayout_t layoutA = nullptr, layoutB = nullptr, layoutC = nullptr;
    CHECK_CUBLASLT( cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_32F, M, K, lda) );
    CHECK_CUBLASLT( cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_32F, K, N, ldb) );
    CHECK_CUBLASLT( cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_32F, M, N, ldc) );

    cublasLtOrder_t rowOrder = CUBLASLT_ORDER_ROW;
    CHECK_CUBLASLT( cublasLtMatrixLayoutSetAttribute(layoutA,
        CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder)) );
    CHECK_CUBLASLT( cublasLtMatrixLayoutSetAttribute(layoutB,
        CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder)) );
    CHECK_CUBLASLT( cublasLtMatrixLayoutSetAttribute(layoutC,
        CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder)) );

    // 3) Preference
    cublasLtMatmulPreference_t preference = nullptr;
    CHECK_CUBLASLT( cublasLtMatmulPreferenceCreate(&preference) );
    CHECK_CUBLASLT( cublasLtMatmulPreferenceSetAttribute(
        preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &maxWsBytes,
        sizeof(maxWsBytes)) );

    // 4) ヒューリスティクス
    const int requestCount = 10;
    cublasLtMatmulHeuristicResult_t heuristics[requestCount];
    int returnedAlgoCount = 0;
    auto st = cublasLtMatmulAlgoGetHeuristic(
        ltHandle,
        opDesc,
        layoutA, layoutB,
        layoutC, layoutC,
        preference,
        requestCount,
        heuristics,
        &returnedAlgoCount
    );
    if (st != CUBLAS_STATUS_SUCCESS || returnedAlgoCount == 0) {
        cublasLtMatmulPreferenceDestroy(preference);
        cublasLtMatrixLayoutDestroy(layoutA);
        cublasLtMatrixLayoutDestroy(layoutB);
        cublasLtMatrixLayoutDestroy(layoutC);
        cublasLtMatmulDescDestroy(opDesc);
        return -1.0f; // algo not found
    }
    auto bestAlgo = heuristics[0].algo;

    // 5) ワークスペース確保
    void* dWorkspace = nullptr;
    if (maxWsBytes > 0) {
        CHECK_CUDA( cudaMalloc(&dWorkspace, maxWsBytes) );
    }

    // 6) ウォームアップ
    float alpha=1.0f, beta=0.0f;
    CHECK_CUBLASLT( cublasLtMatmul(
        ltHandle,
        opDesc,
        &alpha,
        dA, layoutA,
        dB, layoutB,
        &beta,
        dC, layoutC,
        dC, layoutC,
        &bestAlgo,
        dWorkspace,
        maxWsBytes,
        0
    ));
    CHECK_CUDA( cudaDeviceSynchronize() );

    // 7) GPU イベント計測
    cudaEvent_t startEv, stopEv;
    CHECK_CUDA( cudaEventCreate(&startEv) );
    CHECK_CUDA( cudaEventCreate(&stopEv) );

    CHECK_CUDA( cudaEventRecord(startEv, 0) );
    for (int i=0; i<repeat; i++){
        CHECK_CUBLASLT( cublasLtMatmul(
            ltHandle,
            opDesc,
            &alpha,
            dA, layoutA,
            dB, layoutB,
            &beta,
            dC, layoutC,
            dC, layoutC,
            &bestAlgo,
            dWorkspace,
            maxWsBytes,
            0
        ));
    }
    CHECK_CUDA( cudaEventRecord(stopEv, 0) );
    CHECK_CUDA( cudaEventSynchronize(stopEv) );

    float ms = 0.0f;
    CHECK_CUDA( cudaEventElapsedTime(&ms, startEv, stopEv) );
    ms /= repeat;

    // 後片付け
    CHECK_CUDA( cudaEventDestroy(startEv) );
    CHECK_CUDA( cudaEventDestroy(stopEv) );
    if (dWorkspace) {
        CHECK_CUDA( cudaFree(dWorkspace) );
    }
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(layoutA);
    cublasLtMatrixLayoutDestroy(layoutB);
    cublasLtMatrixLayoutDestroy(layoutC);
    cublasLtMatmulDescDestroy(opDesc);

    return ms;
}

// ======== ベンチマーク: cuBLAS (v2)  ================================
// - 従来の cublasSgemm は ColumnMajor 前提
// - RowMajor (M*K, K*N, M*N) をそのまま渡すと正しく計算できない
// => ソフトウェア的に RowMajor->ColMajor に転置コピーしてから gemm
// => 結果 C を colMajor->rowMajor に戻す
// => これらのコピーは計測対象に含めない
// => matmulのカーネル実行時間だけを測定
float benchGemmCublasV2(
    cublasHandle_t handle,
    int M, int N, int K,
    float* dA_row, // input row-major
    float* dB_row, // input row-major
    float* dC_row, // output row-major
    int repeat=5
) {
    // 1) col-major用にメモリ確保
    //    A_col: shape=(M,K) col-major => leadingDimension=M
    //    B_col: shape=(K,N) col-major => leadingDimension=K
    //    C_col: shape=(M,N) col-major => leadingDimension=M
    float *dA_col=nullptr, *dB_col=nullptr, *dC_col=nullptr;
    CHECK_CUDA( cudaMalloc(&dA_col, sizeof(float)*M*K) );
    CHECK_CUDA( cudaMalloc(&dB_col, sizeof(float)*K*N) );
    CHECK_CUDA( cudaMalloc(&dC_col, sizeof(float)*M*N) );

    // row->col copy for A, B
    {
        int blockSize = 256;
        int gridA = (M*K + blockSize - 1)/blockSize;
        copyRowMajorToColMajorKernel<<<gridA, blockSize>>>(dA_row, dA_col, M, K);

        int gridB = (K*N + blockSize - 1)/blockSize;
        copyRowMajorToColMajorKernel<<<gridB, blockSize>>>(dB_row, dB_col, K, N);

        CHECK_CUDA( cudaDeviceSynchronize() );
    }

    // 2) warm-up
    float alpha=1.0f, beta=0.0f;

    //   cublasSgemm( handle, opA, opB, M, N, K, alpha, dA_col, lda, dB_col, ldb, beta, dC_col, ldc)
    //   col-major => "A is MxK, B is KxN"
    //   => lda=M, ldb=K, ldc=M
    CHECK_CUBLAS( cublasSgemm(
        handle,
        CUBLAS_OP_N, // no-trans
        CUBLAS_OP_N,
        M, // rows of C
        N, // cols of C
        K,
        &alpha,
        dA_col, /*lda=*/M,
        dB_col, /*ldb=*/K,
        &beta,
        dC_col, /*ldc=*/M
    ));
    CHECK_CUDA( cudaDeviceSynchronize() );

    // 3) measure
    cudaEvent_t startEv, stopEv;
    CHECK_CUDA( cudaEventCreate(&startEv) );
    CHECK_CUDA( cudaEventCreate(&stopEv) );

    CHECK_CUDA( cudaEventRecord(startEv, 0) );
    for (int i=0; i<repeat; i++){
        CHECK_CUBLAS( cublasSgemm(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_T,
            M,
            N,
            K,
            &alpha,
            dA_col, M,
            dB_col, K,
            &beta,
            dC_col, M
        ));
    }
    CHECK_CUDA( cudaEventRecord(stopEv, 0) );
    CHECK_CUDA( cudaEventSynchronize(stopEv) );

    float ms=0.0f;
    CHECK_CUDA( cudaEventElapsedTime(&ms, startEv, stopEv) );
    ms /= repeat;

    // 4) col->row copy for C
    {
        int blockSize = 256;
        int gridC = (M*N + blockSize - 1)/blockSize;
        copyColMajorToRowMajorKernel<<<gridC, blockSize>>>(dC_col, dC_row, M, N);
        CHECK_CUDA( cudaDeviceSynchronize() );
    }

    CHECK_CUDA( cudaEventDestroy(startEv) );
    CHECK_CUDA( cudaEventDestroy(stopEv) );

    CHECK_CUDA( cudaFree(dA_col) );
    CHECK_CUDA( cudaFree(dB_col) );
    CHECK_CUDA( cudaFree(dC_col) );

    return ms;
}


int main()
{
    // 1) cuBLASLt & cublasHandle
    cublasLtHandle_t ltHandle;
    CHECK_CUBLASLT( cublasLtCreate(&ltHandle) );

    cublasHandle_t cublasHandle;
    CHECK_CUBLAS( cublasCreate_v2(&cublasHandle) );

    // 2) 行列サイズを複数パターン試す
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(32, 512);
    const int numCases = 5;

    // 3) ワークスペースの候補
    std::vector<size_t> wsCandidates = {0, 8ULL<<20, 64ULL<<20, 256ULL<<20, 512ULL<<20, 1024ULL<<20, 2048ULL<<20};

    int factor = 32;

    // 4) ベンチマーク
    for (int c = 0; c < numCases; ++c) {
        int M = dist(gen)*factor;  // 8の倍数
        // int K = dist(gen)*factor;
        // int N = dist(gen)*factor;
        int K = M;
        int N = M;

        // row-majorバッファをGPUに確保 (A,M*K; B,K*N; C,M*N)
        float *dA=nullptr, *dB=nullptr, *dC=nullptr;
        CHECK_CUDA( cudaMalloc(&dA, sizeof(float)*M*K) );
        CHECK_CUDA( cudaMalloc(&dB, sizeof(float)*K*N) );
        CHECK_CUDA( cudaMalloc(&dC, sizeof(float)*M*N) );

        // 適当に初期化
        {
            std::vector<float> hA(M*K), hB(K*N), hC(M*N, 0.f);
            for(int i=0;i<M*K;i++) { hA[i] = float((i+1) % 13); }
            for(int i=0;i<K*N;i++) { hB[i] = float((i*3) % 7); }
            // hC=0
            CHECK_CUDA( cudaMemcpy(dA, hA.data(), sizeof(float)*M*K, cudaMemcpyHostToDevice) );
            CHECK_CUDA( cudaMemcpy(dB, hB.data(), sizeof(float)*K*N, cudaMemcpyHostToDevice) );
            CHECK_CUDA( cudaMemcpy(dC, hC.data(), sizeof(float)*M*N, cudaMemcpyHostToDevice) );
        }

        std::cout << "=== Case " << c << ": M="<<M<<", K="<<K<<", N="<<N << " ===\n";

        // ---- cuBLAS V2 (sgemm) ----
        {
            // cublasSgemm用に row-major dC を初期化(再度0に戻す)
            CHECK_CUDA( cudaMemset(dC, 0, sizeof(float)*M*N) );

            // ここで warm-up は benchGemmCublasV2内部で実行
            float msCublas = benchGemmCublasV2(cublasHandle, M,N,K, dA,dB,dC, 50);
            if (msCublas < 0.0f) {
                std::cout << "  cublasSgemm => error\n";
            } else {
                std::cout << "  cublasSgemm => " << msCublas << " ms\n";
            }
        }

        // ---- cuBLASLt (いろいろなwsサイズ) ----
        for (auto ws : wsCandidates) {
            // row-major dC 初期化
            CHECK_CUDA( cudaMemset(dC, 0, sizeof(float)*M*N) );

            float msLt = benchGemmCublasLt(
                ltHandle,
                M, N, K,
                dA, K,  // row-major => lda=K
                dB, N,  // row-major => ldb=N
                dC, N,  // row-major => ldc=N
                ws,
                50 // repeat
            );
            if (msLt < 0.f) {
                std::cout << "  cuBLASLt(ws=" << (ws>>20) << "MB) => no algo found\n";
            } else {
                std::cout << "  cuBLASLt(ws=" << (ws>>20) << "MB) => " << msLt << " ms\n";
            }
        }
        std::cout << std::endl;

        CHECK_CUDA( cudaFree(dA) );
        CHECK_CUDA( cudaFree(dB) );
        CHECK_CUDA( cudaFree(dC) );
    }

    // 終了
    CHECK_CUBLAS( cublasDestroy_v2(cublasHandle) );
    CHECK_CUBLASLT( cublasLtDestroy(ltHandle) );
    return 0;
}

