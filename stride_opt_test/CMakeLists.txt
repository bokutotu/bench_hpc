# cmake_minimum_required(VERSION 3.14)
# project(StrideVsIncrementBench LANGUAGES CXX)
#
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
#
# # ---- Google Benchmark を FetchContent で自動取得する設定 ----
# include(FetchContent)
#
# FetchContent_Declare(
#   googlebenchmark
#   GIT_REPOSITORY https://github.com/google/benchmark.git
#   GIT_TAG v1.8.0
# )
# FetchContent_MakeAvailable(googlebenchmark)
#
# # ---- ベンチマーク実行ファイルを生成 ----
# add_executable(benchmark_main benchmark.cpp)
# target_link_libraries(benchmark_main PRIVATE benchmark::benchmark benchmark::benchmark_main)
#
# # リリースビルドでコンパイルしたい場合
# #   cmake -DCMAKE_BUILD_TYPE=Release ..
# # などと指定してください。-O3 相当の最適化が有効になります。
#
cmake_minimum_required(VERSION 3.14)
project(MyStrideBench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
setCMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# (重要) Google Benchmark 内部のテスト機能をオフにするオプション
option(BENCHMARK_ENABLE_TESTING "Enable testing of the benchmark library." OFF)
option(BENCHMARK_ENABLE_GTEST_TESTS "Enable building and running the benchmark library's GTest tests." OFF)

# Google Benchmark の自動ダウンロード & ビルド
FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.8.0
)
FetchContent_MakeAvailable(googlebenchmark)

# ベンチマーク用の実行ファイル
add_executable(my_benchmark benchmark_main.cpp)
target_link_libraries(my_benchmark PRIVATE benchmark::benchmark benchmark::benchmark_main)
(
