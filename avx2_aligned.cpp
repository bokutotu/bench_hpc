// g++ -O3 -mavx2 -fopenmp main.cpp -o sum_avx2_vs_omp
#include <cstdlib>      // for aligned_alloc (C++17~) または posix_memalign
#include <immintrin.h>  // AVX2 intrinsic
#include <omp.h>
#include <chrono>
#include <iostream>
#include <random>
#include <cmath>
#include <cstring>      // memsetなど使う場合は

//--------------------------------------------------------------------------------------
// 1. OpenMPのみ (64バイトアライン配列への単純足し算)
//--------------------------------------------------------------------------------------
double sum_omp_aligned(const float* data, size_t N)
{
    double sum = 0.0;

#pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < N; i++) {
        sum += data[i];
    }
    return sum;
}

//--------------------------------------------------------------------------------------
// 2. AVX2 + OpenMP (64バイトアライン配列に対して8要素ずつSIMDロード)
//   - stride=1なので連続アクセス -> _mm256_load_ps が利用可能
//   - N が 8 の倍数でない場合は末尾で処理
//--------------------------------------------------------------------------------------
double sum_avx_omp_aligned(const float* data, size_t N)
{
    double result = 0.0;

#pragma omp parallel
    {
        double local_sum = 0.0;

#pragma omp for
        for (size_t i = 0; i < N; i += 8) {
            // 末尾が8の倍数に満たない場合の処理
            if (i + 7 >= N) {
                for (size_t j = i; j < N; j++) {
                    local_sum += data[j];
                }
                continue;
            }
            // 8つのfloatをロード(アラインされているので _mm256_load_ps が使える)
            __m256 v = _mm256_load_ps(&data[i]);

            // 水平加算 ( _mm256_hadd_ps を2回使う方法 )
            __m256 hsum  = _mm256_hadd_ps(v, v);
            __m256 hsum2 = _mm256_hadd_ps(hsum, hsum);

            // レジスタを一時配列に書き出して合計を取り出す
            float tmp[8];
            _mm256_store_ps(tmp, hsum2);
            // tmp[0] と tmp[4] に同じ合計が入っている
            local_sum += tmp[0];
        }

#pragma omp atomic
        result += local_sum;
    }

    return result;
}

//--------------------------------------------------------------------------------------
// メイン関数
//   - 64バイトアラインされた領域を確保
//   - データを初期化
//   - それぞれの関数を呼び出して実行時間を測定
//--------------------------------------------------------------------------------------
int main()
{
    // 配列サイズ
    const size_t N = 100000000UL; // 1億要素

    //========================
    // 64バイトアラインで確保
    //========================
    // C++17 以降: std::aligned_alloc(align, size)
    //   align は 2 の冪乗である必要がある (64は 2^6)
    //   size は align の倍数である必要がある (N * sizeof(float) が64の倍数になるよう注意)
    //
    // もし strict な要件でサイズが64の倍数でない場合、
    // 多少オーバー分を確保してポインタ操作を行うなど工夫が必要ですが、
    // 今回は簡便のために N*sizeof(float) が 64の倍数になるかはあまり気にしないことにします。
    //
    // POSIX 環境なら:
    //   posix_memalign((void**)&data_aligned, 64, N*sizeof(float));
    // でもOK。
    //========================
    float* data_aligned = static_cast<float*>(std::aligned_alloc(64, N * sizeof(float)));
    if (!data_aligned) {
        std::cerr << "aligned_alloc failed!" << std::endl;
        return 1;
    }

    // 万が一、N*sizeof(float) が 64の倍数でない場合は std::aligned_alloc が失敗する可能性あり

    // 配列を初期化 (乱数で)
    {
        std::mt19937_64 mt(12345);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        for (size_t i = 0; i < N; i++) {
            data_aligned[i] = dist(mt);
        }
    }

    // 時間計測用の構造体
    struct ResultInfo {
        double sum_value;
        double time_sec;
        std::string label;
    };

    std::vector<ResultInfo> results;

    //----------------------------------------------------------------------------------
    // 1) OpenMPのみ
    //----------------------------------------------------------------------------------
    double elapsed_time = 0.0;
    for (int i = 0; i < 300; i++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        double sum_val = sum_omp_aligned(data_aligned, N);
        auto end   = std::chrono::high_resolution_clock::now();
        elapsed_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-6;

    }

    //----------------------------------------------------------------------------------
    // 2) AVX2 + OpenMP
    //----------------------------------------------------------------------------------
    double elapsed_time_avx = 0.0;
    for (int i = 0; i < 300; i++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        double sum_val = sum_avx_omp_aligned(data_aligned, N);
        auto end   = std::chrono::high_resolution_clock::now();
        /*double elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-6;*/
        elapsed_time_avx += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-6;

    }

    // 結果表示
    std::cout << "OpenMP only: " << elapsed_time / 300 << " [sec]" << std::endl;
    std::cout << "AVX2 + OpenMP: " << elapsed_time_avx / 300 << " [sec]" << std::endl;
    /*std::cout << "Array size: " << N << std::endl;*/
    /*std::cout << "Allocated 64-byte aligned memory\n\n";*/
    /*for (auto &r : results) {*/
    /*    std::cout << r.label << " -> sum = " << r.sum_value*/
    /*              << ", time = " << r.time_sec << " [sec]" << std::endl;*/
    /*}*/
    /**/
    /*// 差分チェック (実際にはほぼ同じになるはず)*/
    /*double diff = std::fabs(results[0].sum_value - results[1].sum_value);*/
    /*std::cout << "\nDifference(OMP only vs AVX2+OMP) = " << diff << std::endl;*/

    // 後始末（メモリ解放）
    std::free(data_aligned);

    return 0;
}

