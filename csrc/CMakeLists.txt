cmake_minimum_required(VERSION 3.24)
project(foveated_ext LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MLX (installed via pip, MLX_DIR set by mlx.extension.CMakeBuild)
find_package(MLX CONFIG REQUIRED)

# Include MLX's extension.cmake for mlx_build_metallib macro
include(${MLX_CMAKE_DIR}/extension.cmake)

# Compile Metal kernel to precompiled metallib
mlx_build_metallib(
    TARGET foveated_metallib
    TITLE foveated_attn
    OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/foveated_attn.metal
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/foveated_compress.metal
    INCLUDE_DIRS ${MLX_INCLUDE_DIRS}
)

# Find Python and nanobind
find_package(Threads REQUIRED)
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
    foveated_compress.cpp
    promotion_pipeline.cpp
)

target_include_directories(foveated_ext PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(foveated_ext PRIVATE mlx Threads::Threads)

# Ensure metallib is built before the extension
add_dependencies(foveated_ext foveated_metallib)

if(APPLE)
    target_link_libraries(foveated_ext PRIVATE "-framework Metal" "-framework Foundation")
endif()

# Install metallib alongside the extension so it can be found at runtime
install(FILES ${CMAKE_BINARY_DIR}/foveated_attn.metallib
        DESTINATION ${CMAKE_INSTALL_PREFIX})

# Copy built artifacts to project root for easy import
add_custom_command(TARGET foveated_ext POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:foveated_ext>
        ${CMAKE_SOURCE_DIR}/../$<TARGET_FILE_NAME:foveated_ext>
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_BINARY_DIR}/foveated_attn.metallib
        ${CMAKE_SOURCE_DIR}/../foveated_attn.metallib
    COMMENT "Copying extension + metallib to project root"
)
