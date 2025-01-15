// nvcc -std=c++17 -o bench_cublaslt bench_cublaslt.cu -lcublasLt -lcublas
#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cstdlib>

// ----------------- エラーチェック用 --------------------
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

// -------------------------------------------------------
// ベンチマーク用: M, N, K の単精度 RowMajor GEMM を実行し、
// cublasLtMatmul() の GPU実行時間を計測 (ms) して返す
//   - workspace は maxWsBytes の範囲で確保
//   - cublasLtMatmulAlgoGetHeuristic() で見つかった先頭のalgoを使う
//   - 失敗した場合(アルゴリズムなし)は -1を返す
//   - repeat 回だけ繰り返して平均実行時間を返す
// -------------------------------------------------------
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

    // 転置なし (RowMajorで lda=K, ldb=N, なので実質CUBLAS_OP_N)
    cublasOperation_t opA = CUBLAS_OP_N;
    cublasOperation_t opB = CUBLAS_OP_N;
    CHECK_CUBLASLT( cublasLtMatmulDescSetAttribute(
        opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
    CHECK_CUBLASLT( cublasLtMatmulDescSetAttribute(
        opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));

    // 2) MatrixLayout
    cublasLtMatrixLayout_t layoutA = nullptr, layoutB = nullptr, layoutC = nullptr;
    CHECK_CUBLASLT( cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_32F, M, K, lda) );
    CHECK_CUBLASLT( cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_32F, K, N, ldb) );
    CHECK_CUBLASLT( cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_32F, M, N, ldc) );

    // RowMajorを指定
    cublasLtOrder_t rowOrder = CUBLASLT_ORDER_ROW;
    CHECK_CUBLASLT( cublasLtMatrixLayoutSetAttribute(
        layoutA, CUBLASLT_MATRIX_LAYOUT_ORDER,
        &rowOrder, sizeof(rowOrder)) );
    CHECK_CUBLASLT( cublasLtMatrixLayoutSetAttribute(
        layoutB, CUBLASLT_MATRIX_LAYOUT_ORDER,
        &rowOrder, sizeof(rowOrder)) );
    CHECK_CUBLASLT( cublasLtMatrixLayoutSetAttribute(
        layoutC, CUBLASLT_MATRIX_LAYOUT_ORDER,
        &rowOrder, sizeof(rowOrder)) );

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
    auto stat = cublasLtMatmulAlgoGetHeuristic(
        ltHandle,
        opDesc,
        layoutA, layoutB,
        layoutC, layoutC,
        preference,
        requestCount,
        heuristics,
        &returnedAlgoCount
    );
    if (stat != CUBLAS_STATUS_SUCCESS || returnedAlgoCount == 0) {
        // cleanup
        cublasLtMatmulPreferenceDestroy(preference);
        cublasLtMatrixLayoutDestroy(layoutA);
        cublasLtMatrixLayoutDestroy(layoutB);
        cublasLtMatrixLayoutDestroy(layoutC);
        cublasLtMatmulDescDestroy(opDesc);
        return -1.0f; // algo not found
    }

    // 先頭のアルゴリズムを使用 (本当は複数試して最速を選ぶのが理想)
    auto bestAlgo = heuristics[0].algo;

    // 5) ワークスペースを確保 (malloc/free は計測時間に含めない)
    void* dWorkspace = nullptr;
    if (maxWsBytes > 0) {
        CHECK_CUDA( cudaMalloc(&dWorkspace, maxWsBytes) );
    }

    // 6) ウォームアップ
    float alpha=1.0f, beta=0.0f;
    CHECK_CUBLASLT(
        cublasLtMatmul(
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
            0 // stream=0
        )
    );
    CHECK_CUDA( cudaDeviceSynchronize() );

    // 7) CUDAイベントで時間測定
    cudaEvent_t startEv, stopEv;
    CHECK_CUDA( cudaEventCreate(&startEv) );
    CHECK_CUDA( cudaEventCreate(&stopEv) );

    CHECK_CUDA( cudaEventRecord(startEv, 0) );
    for (int i = 0; i < repeat; ++i) {
        CHECK_CUBLASLT(
            cublasLtMatmul(
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
            )
        );
    }
    CHECK_CUDA( cudaEventRecord(stopEv, 0) );
    CHECK_CUDA( cudaEventSynchronize(stopEv) );

    float elapsedMs = 0.0f;
    CHECK_CUDA( cudaEventElapsedTime(&elapsedMs, startEv, stopEv) );
    elapsedMs /= repeat; // 1回あたり平均

    CHECK_CUDA( cudaEventDestroy(startEv) );
    CHECK_CUDA( cudaEventDestroy(stopEv) );
    if (dWorkspace) {
        CHECK_CUDA( cudaFree(dWorkspace) );
    }

    // cleanup
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(layoutA);
    cublasLtMatrixLayoutDestroy(layoutB);
    cublasLtMatrixLayoutDestroy(layoutC);
    cublasLtMatmulDescDestroy(opDesc);

    return elapsedMs; // [ms]
}

int main()
{
    // 1) cuBLASLtハンドルを作成
    cublasLtHandle_t ltHandle;
    CHECK_CUBLASLT( cublasLtCreate(&ltHandle) );

    // 2) 行列サイズを複数パターン試す
    //    (ランダム or 固定; ここではランダム)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(256, 2048); // [256..2048]
    const int numCases = 5; // 5パターン計測

    // 3) ワークスペースの候補
    std::vector<size_t> wsCandidates = {0, 8ULL<<20, 64ULL<<20, 256ULL<<20, 1024ULL<<20}; 
      // 0MB, 8MB, 64MB

    // 4) ベンチマーク
    for (int c = 0; c < numCases; ++c) {
        int M = dist(gen)*8;
        int K = dist(gen)*8;
        int N = dist(gen)*8;

        // GPUメモリに A(M*K), B(K*N), C(M*N) 確保
        float *dA=nullptr, *dB=nullptr, *dC=nullptr;
        CHECK_CUDA( cudaMalloc(&dA, sizeof(float)*M*K) );
        CHECK_CUDA( cudaMalloc(&dB, sizeof(float)*K*N) );
        CHECK_CUDA( cudaMalloc(&dC, sizeof(float)*M*N) );

        // 初期化 (簡易にホストで埋めてコピー)
        {
            std::vector<float> hA(M*K), hB(K*N), hC(M*N);
            for(int i=0;i<M*K;i++){ hA[i] = float((i+1) % 17); }
            for(int i=0;i<K*N;i++){ hB[i] = float((i*7) % 13); }
            for(int i=0;i<M*N;i++){ hC[i] = 0.f; }
            CHECK_CUDA( cudaMemcpy(dA, hA.data(), sizeof(float)*M*K, cudaMemcpyHostToDevice) );
            CHECK_CUDA( cudaMemcpy(dB, hB.data(), sizeof(float)*K*N, cudaMemcpyHostToDevice) );
            CHECK_CUDA( cudaMemcpy(dC, hC.data(), sizeof(float)*M*N, cudaMemcpyHostToDevice) );
        }

        std::cout << "=== Case " << c << ": M="<<M<<", K="<<K<<", N="<<N << " ===\n";
        for (auto ws : wsCandidates) {
            // bench
            float ms = benchGemmCublasLt(
                ltHandle,
                M,N,K,
                dA, K,  // lda=K
                dB, N,  // ldb=N
                dC, N,  // ldc=N
                ws,
                100 // repeat
            );
            if (ms < 0.f) {
                // アルゴリズムが見つからない場合
                std::cout << "  ws=" << (ws>>20) << "MB => no algo found\n";
            } else {
                std::cout << "  ws=" << (ws>>20) << "MB => " << ms << " ms\n";
            }
        }
        std::cout << std::endl;

        CHECK_CUDA( cudaFree(dA) );
        CHECK_CUDA( cudaFree(dB) );
        CHECK_CUDA( cudaFree(dC) );
    }

    // 終了
    CHECK_CUBLASLT( cublasLtDestroy(ltHandle) );
    return 0;
}

