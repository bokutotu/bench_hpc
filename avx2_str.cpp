// g++ -O3 -mavx2 -fopenmp avx2_str.cpp -o sum_avx2_vs_omp
#include <cstdlib>      // for aligned_alloc (C++17) or posix_memalign
#include <immintrin.h>  // AVX2 / Gather 命令用
#include <omp.h>
#include <chrono>
#include <iostream>
#include <random>
#include <cmath>

//==============================================================
// 1. OpenMPのみ (stride付き, 64バイトアライン配列に対して)
//==============================================================
double sum_omp_stride_aligned(const float* data, size_t N, size_t stride)
{
    double sum = 0.0;

#pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < N; i += stride) {
        sum += data[i];
    }
    return sum;
}

//==============================================================
// 2. AVX2 + OpenMP (stride付き, 64バイトアライン配列に対して)
//    - _mm256_i32gather_ps を用いて非連続アクセスをまとめてロード
//    - 8要素ずつ処理し、末尾が合わない部分は通常ループで処理して continue
//==============================================================
double sum_avx_omp_stride_aligned(const float* data, size_t N, size_t stride)
{
    double result = 0.0;

#pragma omp parallel
    {
        double local_sum = 0.0;

#pragma omp for
        for (size_t i = 0; i < N; i += 8 * stride) {
            // もし i + 7*stride >= N なら、ここから先の 8要素分ロードはできない。
            // 端数を通常ループで処理して continueする。
            if (i + 7 * stride >= N) {
                for (size_t j = i; j < N; j += stride) {
                    local_sum += data[j];
                }
                // breakではなくcontinueに変更
                continue;
            }

            // 8つのインデックスを __m256i に格納
            __m256i index_v = _mm256_setr_epi32(
                static_cast<int>(i + 0 * stride),
                static_cast<int>(i + 1 * stride),
                static_cast<int>(i + 2 * stride),
                static_cast<int>(i + 3 * stride),
                static_cast<int>(i + 4 * stride),
                static_cast<int>(i + 5 * stride),
                static_cast<int>(i + 6 * stride),
                static_cast<int>(i + 7 * stride)
            );

            // Gather命令で非連続アクセスをまとめてロード (scale=4: floatサイズ)
            __m256 v = _mm256_i32gather_ps(data, index_v, 4);

            // 水平方向に加算 ( _mm256_hadd_ps を2回 )
            __m256 hsum  = _mm256_hadd_ps(v, v);
            __m256 hsum2 = _mm256_hadd_ps(hsum, hsum);

            float tmp[8];
            _mm256_store_ps(tmp, hsum2);
            // tmp[0], tmp[4] に同じ結果
            local_sum += tmp[0];
        }

#pragma omp atomic
        result += local_sum;
    }

    return result;
}

//===================================================================
// メイン関数
//   - 64バイトアラインでメモリ確保
//   - データ初期化 (乱数)
//   - 上記2種類の関数を各100回ずつ実行して平均時間を測定
//===================================================================
int main()
{
    // 配列サイズ・ストライド
    const size_t N = 100000000UL;  // 1億要素
    const size_t stride = 10;

    // 実験回数
    const int NUM_TRIALS = 100;

    //===========================================================
    // 64バイトアラインでメモリを確保 (C++17 aligned_alloc)
    //===========================================================
    // 注意: N*sizeof(float) が64の倍数でないと失敗する可能性がある
    float* data_aligned = static_cast<float*>(std::aligned_alloc(64, N * sizeof(float)));
    if (!data_aligned) {
        std::cerr << "aligned_alloc failed!" << std::endl;
        return 1;
    }

    // 配列初期化 (乱数)
    {
        std::mt19937_64 mt(12345);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (size_t i = 0; i < N; i++) {
            data_aligned[i] = dist(mt);
        }
    }

    //----------------------------------------------------------------------------------
    // 1) sum_omp_stride_aligned を100回実行し、平均時間を測る
    //----------------------------------------------------------------------------------
    double total_time_omp = 0.0;
    double sum_check_omp  = 0.0; // 最後にsum値を記録(複数回の結果はほぼ同じになる想定)

    for (int t = 0; t < NUM_TRIALS; t++) {
        auto start = std::chrono::high_resolution_clock::now();
        double s = sum_omp_stride_aligned(data_aligned, N, stride);
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-6;

        total_time_omp += elapsed;
        // 最終回の値を記録 or 平均を取ってもOK
        if (t == NUM_TRIALS - 1) {
            sum_check_omp = s;
        }
    }
    double avg_time_omp = total_time_omp / NUM_TRIALS;

    //----------------------------------------------------------------------------------
    // 2) sum_avx_omp_stride_aligned を100回実行し、平均時間を測る
    //----------------------------------------------------------------------------------
    double total_time_avx = 0.0;
    double sum_check_avx  = 0.0; // 最後にsum値を記録

    for (int t = 0; t < NUM_TRIALS; t++) {
        auto start = std::chrono::high_resolution_clock::now();
        double s = sum_avx_omp_stride_aligned(data_aligned, N, stride);
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-6;

        total_time_avx += elapsed;
        if (t == NUM_TRIALS - 1) {
            sum_check_avx = s;
        }
    }
    double avg_time_avx = total_time_avx / NUM_TRIALS;

    //----------------------------------------------------------------------------------
    // 結果表示
    //----------------------------------------------------------------------------------
    std::cout << "Array size    : " << N << "\n";
    std::cout << "Stride        : " << stride << "\n";
    std::cout << "Aligned alloc : 64-byte\n";
    std::cout << "Trials        : " << NUM_TRIALS << "\n\n";

    std::cout << "[OMP stride]       -> sum = " << sum_check_omp
              << ", avg_time = " << avg_time_omp << " [sec] (over " << NUM_TRIALS << " runs)\n";
    std::cout << "[AVX2+OMP stride]  -> sum = " << sum_check_avx
              << ", avg_time = " << avg_time_avx << " [sec] (over " << NUM_TRIALS << " runs)\n";

    // 差分チェック
    double diff = std::fabs(sum_check_omp - sum_check_avx);
    std::cout << "\nDifference in sums = " << diff << "\n";

    // メモリ解放
    std::free(data_aligned);

    return 0;
}

