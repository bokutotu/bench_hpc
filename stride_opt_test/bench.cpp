#include <benchmark/benchmark.h>
#include <vector>

//-------------------------------------------------------
// i += stride のループを実際に回す関数
// __attribute__((noinline)) によりインライン化を抑制
//-------------------------------------------------------
#ifdef _MSC_VER
// MSVC の場合は __declspec(noinline)
__declspec(noinline)
#else
__attribute__((noinline))
#endif
int sum_by_stride(const int* data, int n, int stride)
{
    int sum = 0;
    for (int i = 0; i < n; i += stride) {
        sum += data[i];
    }
    return sum;
}

//-------------------------------------------------------
// i++ のループを実際に回す関数
//-------------------------------------------------------
#ifdef _MSC_VER
__declspec(noinline)
#else
__attribute__((noinline))
#endif
int sum_by_increment(const int* data, int n)
{
    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    return sum;
}

//-------------------------------------------------------
// ベンチマーク: stride で回す (stride を実行時に指定)
//-------------------------------------------------------
static void BM_StrideLoop(benchmark::State& state)
{
    const int n = static_cast<int>(state.range(0));
    const int stride = static_cast<int>(state.range(1));

    // 毎回同じデータを準備
    std::vector<int> data(n);
    for (int i = 0; i < n; i++) {
        data[i] = i;
    }

    for (auto _ : state) {
        // 実際の計算を実行し、最適化で消されないようにする
        int result = sum_by_stride(data.data(), n, stride);
        benchmark::DoNotOptimize(result);
    }
}

//-------------------------------------------------------
// ベンチマーク: i++ で回す
//-------------------------------------------------------
static void BM_IncrementLoop(benchmark::State& state)
{
    const int n = static_cast<int>(state.range(0));

    std::vector<int> data(n);
    for (int i = 0; i < n; i++) {
        data[i] = i;
    }

    for (auto _ : state) {
        int result = sum_by_increment(data.data(), n);
        benchmark::DoNotOptimize(result);
    }
}

//-------------------------------------------------------
// ベンチマークの登録
//   - (n=10000000, stride=1) と (n=10000000, stride=4) を用意
//   - i++ のパターン (n=10000000)
//-------------------------------------------------------
BENCHMARK(BM_StrideLoop)->Args({10'000'000, 1});
BENCHMARK(BM_StrideLoop)->Args({10'000'000, 4});
BENCHMARK(BM_IncrementLoop)->Arg(10'000'000);

// Google Benchmark のエントリーポイント
BENCHMARK_MAIN();

