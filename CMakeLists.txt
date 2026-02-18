cmake_minimum_required(VERSION 3.20)
project(LazyTensor VERSION 1.0 LANGUAGES CXX)

# Enforce C++23
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler Warnings
add_compile_options(-Wall -Wextra -Wpedantic -Werror)

# Define the Header-Only Library
add_library(tensor_lib INTERFACE)
target_include_directories(tensor_lib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Define the Main Demo Executable
add_executable(tensor_demo src/main.cpp)
target_link_libraries(tensor_demo PRIVATE tensor_lib)

# ==========================================
# GoogleTest Integration via FetchContent
# ==========================================
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Define the Unit Tests
enable_testing()
add_executable(tensor_tests tests/test_tensor.cpp)
target_link_libraries(tensor_tests PRIVATE tensor_lib gtest_main)

include(GoogleTest)
gtest_discover_tests(tensor_tests)
