cmake_minimum_required(VERSION 3.10)
project(axpy_benchmark CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Release ビルド時に -O3
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

add_executable(axpy_benchmark main.cpp)

#
# OpenBLAS を探す (config.cmake がない場合の代替策)
#
# 1) まず 'openblas' ライブラリを探す
find_library(OPENBLAS_LIB
    NAMES openblas
    PATHS /usr/lib /usr/local/lib
          /lib /opt/OpenBLAS/lib
    # etc... 必要に応じて追加
)

if(NOT OPENBLAS_LIB)
    message(FATAL_ERROR "Could NOT find libopenblas. Please install OpenBLAS or set OPENBLAS_LIB.")
else()
    message(STATUS "Found OpenBLAS lib: ${OPENBLAS_LIB}")
endif()

# 2) インクルードパスを設定（/usr/include が通っていないなら明示する）
#    ただし通常の Linux なら /usr/include はデフォルトで通っているはず
include_directories("/usr/include")

# 3) リンクする
target_link_libraries(axpy_benchmark PUBLIC ${OPENBLAS_LIB})

