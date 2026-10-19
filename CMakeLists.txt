
cmake_minimum_required(VERSION 3.20)

project(rltools_mujoco_cpp_test LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Keep the initial build focused on the libraries we need.
set(MUJOCO_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(MUJOCO_BUILD_SIMULATE OFF CACHE BOOL "" FORCE)
set(MUJOCO_BUILD_TESTS OFF CACHE BOOL "" FORCE)

# Build MuJoCo from the submodule.
add_subdirectory(external/mujoco EXCLUDE_FROM_ALL)

add_executable(test
    src/main.cpp
    src/mujoco_env.cpp
    src/nn.cpp
)

# RL-Tools is primarily header-based.
target_include_directories(test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/rl-tools/include
)

target_link_libraries(test PRIVATE mujoco)

if(WIN32)
    add_custom_command(TARGET test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:mujoco>"
            "$<TARGET_FILE_DIR:test>"
        VERBATIM
    )
endif()
