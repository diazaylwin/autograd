cmake_minimum_required(VERSION 3.20)

project(miniad
    VERSION 0.1.0
    LANGUAGES CXX
)

# Global C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Common warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Library target (so you can scale later)
add_library(miniad_lib
    src/runtime.cpp
    src/autodiff.cpp
    src/utils.cpp
    src/gradcheck.cpp
    src/optimization.cpp
    examples/rk4.cpp
)

target_include_directories(miniad_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/examples
)

# Example executable
add_executable(miniad
    main.cpp
)

target_link_libraries(miniad PRIVATE miniad_lib)
