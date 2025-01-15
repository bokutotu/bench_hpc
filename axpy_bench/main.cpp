#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>    // std::abs

#ifdef _OPENMP
#include <omp.h>
#endif

// BLAS (OpenBLAS) の CBLAS インターフェースを使うためのヘッダ
// インクルードパスが通っていれば <cblas.h> でOK
// 通っていなければ #include "path/to/cblas.h" が必要
extern "C" {
#include <cblas.h>
}

//--------------------------------------------
// 自前実装 AXPY: y[i] += alpha * x[i]
// （OpenMP SIMDを使うかどうかはコンパイル時に pragma をコメント/アンコメント）
//--------------------------------------------
double axpy_benchmark_omp(size_t N, double alpha, const std::vector<double>& x, std::vector<double>& y)
{
    auto start = std::chrono::high_resolution_clock::now();

    // OpenMP の SIMD 指示をつける場合
    #pragma omp parallel for simd
    for (size_t i = 0; i < N; i++) {
        y[i] += alpha * x[i];
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    return diff.count();
}

//--------------------------------------------
// OpenBLAS (cblas_daxpy) を使った AXPY
//--------------------------------------------
double axpy_benchmark_openblas(size_t N, double alpha, const std::vector<double>& x, std::vector<double>& y)
{
    auto start = std::chrono::high_resolution_clock::now();

    // cblas_daxpy(
    //     N,             // 配列サイズ
    //     alpha,         // スカラ
    //     x.data(), 1,   // x配列, ストライド1
    //     y.data(), 1    // y配列, ストライド1
    // );
    cblas_daxpy((int)N, alpha, x.data(), 1, y.data(), 1);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    return diff.count();
}

//--------------------------------------------
// ベンチマークを行う補助関数
//--------------------------------------------
double run_benchmark(
    size_t N,
    double alpha,
    const std::vector<double>& x,
    const std::vector<double>& y_original,
    int num_trials,
    bool use_openblas // true: OpenBLAS, false: 自前実装(OMP SIMD)
)
{
    // ウォームアップ
    {
        std::vector<double> y_tmp = y_original;
        if(use_openblas) {
            axpy_benchmark_openblas(N, alpha, x, y_tmp);
        } else {
            axpy_benchmark_omp(N, alpha, x, y_tmp);
        }
    }

    // ベンチマーク計測
    double total_time = 0.0;
    for(int i = 0; i < num_trials; i++){
        std::vector<double> y_trial = y_original; // 毎回同じ初期状態
        double t = 0.0;
        if(use_openblas) {
            t = axpy_benchmark_openblas(N, alpha, x, y_trial);
        } else {
            t = axpy_benchmark_omp(N, alpha, x, y_trial);
        }
        total_time += t;
    }
    return total_time / num_trials;
}

//--------------------------------------------
// メインルーチン
//--------------------------------------------
int main()
{
    // ベクトルサイズ
    const size_t N = 100'000'000; // 1e8
    const double alpha = 2.0;

    // ベクトルを初期化
    //   x[i] = 1.0
    //   y[i] = 1.0
    std::vector<double> x(N, 1.0);
    std::vector<double> y(N, 1.0);

    // 試行回数
    const int num_trials = 5;

    //--- (1) 自前実装 (OpenMP SIMD) の平均実行時間 ---
    double avg_time_omp = run_benchmark(N, alpha, x, y, num_trials, /*use_openblas=*/false);

    //--- (2) OpenBLAS (cblas_daxpy) の平均実行時間 ---
    double avg_time_blas = run_benchmark(N, alpha, x, y, num_trials, /*use_openblas=*/true);

    //=== 正しさ確認用 ===
    // 最後に1回ずつ実行して結果を比較（誤差チェック）

    // (1) 自前実装
    std::vector<double> y_check_omp = y;
    axpy_benchmark_omp(N, alpha, x, y_check_omp);

    // (2) OpenBLAS
    std::vector<double> y_check_blas = y;
    axpy_benchmark_openblas(N, alpha, x, y_check_blas);

    // 正解値 ( 1 + alpha*1 = 1 + 2*1 = 3 ) との乖離をざっくり確認
    auto check_error = [&](const std::vector<double>& v){
        double error_sum = 0.0;
        for(auto val : v){
            error_sum += std::abs(val - 3.0);
        }
        return error_sum;
    };

    double err_omp = check_error(y_check_omp);
    double err_blas = check_error(y_check_blas);

    std::cout << "[Check] Self-AXPY (OMP) error sum = " << err_omp << "\n";
    std::cout << "[Check] OpenBLAS daxpy error sum = " << err_blas << "\n";

    //=== 性能指標(GFLOPS)算出 (乗算+加算 = 2N FLOP) ===
    double gflops_omp  = (2.0 * (double)N) / (avg_time_omp  * 1e9);
    double gflops_blas = (2.0 * (double)N) / (avg_time_blas * 1e9);

    //=== 結果表示 ===
    std::cout << "\n=== Benchmark Results (N = " << N << ", trials = " << num_trials << ") ===\n";
    std::cout << "OpenMP(AXPY)   : " << avg_time_omp  << " [sec], " << gflops_omp  << " GFLOPS\n";
    std::cout << "OpenBLAS(daxpy): " << avg_time_blas << " [sec], " << gflops_blas << " GFLOPS\n";

    return 0;
}

