cmake_minimum_required(VERSION 3.18)

project(Splendor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Use an optimized default for single-config generators unless the caller
# explicitly chooses a build type.
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)

# Ensure optimized builds use -O3 explicitly for Release and RelWithDebInfo
if(NOT CMAKE_C_FLAGS_RELEASE MATCHES "-O3")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3")
endif()
if(NOT CMAKE_CXX_FLAGS_RELEASE MATCHES "-O3")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
endif()
if(NOT CMAKE_C_FLAGS_RELWITHDEBINFO MATCHES "-O3")
    set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} -O3")
endif()
if(NOT CMAKE_CXX_FLAGS_RELWITHDEBINFO MATCHES "-O3")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O3")
endif()
option(SPLENDOR_BUILD_TEST_BINS "Build local C++ test/helper binaries" ON)
option(SPLENDOR_BUILD_PYBIND "Build pybind11 Python extension (requires py_splendor.cpp + pybind11)" ON)

add_library(splendor_core STATIC
    game_logic.cpp
    state_encoder.cpp
)

target_include_directories(splendor_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Makes the core library reusable from a shared Python extension target.
set_target_properties(splendor_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(SPLENDOR_BUILD_TEST_BINS)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_game_logic_core.cpp")
        add_executable(test_game_logic
            tests/test_game_logic_core.cpp
        )
        target_link_libraries(test_game_logic PRIVATE splendor_core)
    endif()

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_applymove_invalid_cases.cpp")
        add_executable(test_applymove_invalid_cases_bin
            tests/test_applymove_invalid_cases.cpp
        )
        target_link_libraries(test_applymove_invalid_cases_bin PRIVATE splendor_core)
    endif()

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/manual_game_loop.cpp")
        add_executable(manual_game_loop
            tests/manual_game_loop.cpp
        )
        target_link_libraries(manual_game_loop PRIVATE splendor_core)
    endif()
endif()

if(SPLENDOR_BUILD_PYBIND)
    set(SPLENDOR_PYBIND_SRC "${CMAKE_CURRENT_SOURCE_DIR}/py_splendor.cpp")
    if(NOT EXISTS "${SPLENDOR_PYBIND_SRC}")
        message(FATAL_ERROR "SPLENDOR_BUILD_PYBIND=ON but ${SPLENDOR_PYBIND_SRC} does not exist.")
    endif()

    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(pybind11 CONFIG QUIET)
    if(NOT pybind11_FOUND)
        execute_process(
            COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
            OUTPUT_VARIABLE _pybind11_cmakedir
            RESULT_VARIABLE _pybind11_cmakedir_result
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(_pybind11_cmakedir_result EQUAL 0 AND EXISTS "${_pybind11_cmakedir}")
            set(pybind11_DIR "${_pybind11_cmakedir}" CACHE PATH "pybind11 CMake config directory" FORCE)
            find_package(pybind11 CONFIG REQUIRED)
        else()
            message(FATAL_ERROR
                "pybind11 CMake package not found. Install pybind11 into the active Python environment "
                "(e.g. `python -m pip install pybind11`) and re-run CMake, or set pybind11_DIR manually."
            )
        endif()
    endif()

    find_package(Threads REQUIRED)

    pybind11_add_module(splendor_native MODULE
        py_splendor.cpp
        native_mcts.cpp
        native_endgame.cpp
    )

    target_link_libraries(splendor_native PRIVATE splendor_core Threads::Threads)
    target_include_directories(splendor_native PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(splendor_native PRIVATE
        SPLENDOR_BUILD_TYPE="$<CONFIG>"
        SPLENDOR_BUILD_OPTIMIZED=$<IF:$<CONFIG:Debug>,0,1>
    )
endif()
