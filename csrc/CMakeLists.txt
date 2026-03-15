cmake_minimum_required(VERSION 3.24)
project(foveated_ext LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MLX (installed via pip, MLX_DIR set by mlx.extension.CMakeBuild)
find_package(MLX CONFIG REQUIRED)

# Find Python and nanobind
find_package(Python 3.11 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_DIR)
find_package(nanobind CONFIG REQUIRED)

# Build the extension module
# NB_DOMAIN mlx: share type registry with MLX's core module
# NB_STATIC: statically link nanobind (matches MLX's build)
nanobind_add_module(foveated_ext
    NB_STATIC STABLE_ABI LTO NOMINSIZE
    NB_DOMAIN mlx
    bindings.cpp
    foveated_attn.cpp
    foveated_primitive.cpp
)

target_include_directories(foveated_ext PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(foveated_ext PRIVATE mlx)

if(APPLE)
    target_link_libraries(foveated_ext PRIVATE "-framework Metal" "-framework Foundation")
endif()
