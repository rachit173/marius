# Build instructions
# 1. mkdir build && cd build
# 2. Something like cmake -DCUDNN_INCLUDE_PATH=/usr/local/cuda/include -DCUDNN_LIBRARY_PATH=/usr/local/cuda/lib64 -DCMAKE_CXX_COMPILER=/usr/bin/g++-8  -DMARIUS_TORCH_DIR=<path to libtorch> ..
# 3. make -j4
cmake_minimum_required(VERSION 3.12.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
cmake_policy(SET CMP0048 NEW)
project(marius VERSION 0.1 LANGUAGES CXX)
include(FindPackageHandleStandardArgs)

add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11.0)
        message(FATAL_ERROR "Clang version must be at least 11!")
    endif()
    set(CLANG TRUE)
elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.5)
        message(FATAL_ERROR "GCC version must be at least 7.5!")
    endif()
    set(GCC TRUE)
else ()
    message(FATAL_ERROR "Unknown compiler")
endif ()

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CMAKE_MACOSX_RPATH 1)
endif ()

if(NOT IS_ABSOLUTE ${MARIUS_TORCH_DIR})
    set(MARIUS_TORCH_DIR "${CMAKE_CURRENT_BINARY_DIR}/${MARIUS_TORCH_DIR}")
endif()

if(NOT IS_ABSOLUTE ${MARIUS_GLOO_DIR})
    set(MARIUS_GLOO_DIR "${CMAKE_CURRENT_BINARY_DIR}/${MARIUS_GLOO_DIR}")
endif()

set(Torch_DIR ${MARIUS_TORCH_DIR}/share/cmake/Torch)
set(Gloo_DIR ${MARUIS_GLOO_DIR}/share/cmake/Gloo)
set(CMAKE_CXX_FLAGS "-g ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} ${GLOO_CXX_FLAGS}")

if(${USE_CUDA})
    add_definitions(-DMARIUS_CUDA=${USE_CUDA})
endif()


# Find torch location
execute_process(
        COMMAND python3 -c "import torch; import os; print(os.path.dirname(torch.__file__), end='')"
        OUTPUT_VARIABLE TorchPath
)
list(APPEND CMAKE_PREFIX_PATH ${TorchPath})

execute_process(
        COMMAND python3 -c "import torch; print(torch.__version__, end='')"
        OUTPUT_VARIABLE TorchVersion
)
message(STATUS "Torch Version: ${TorchVersion}")

# Add the cmake folder so the FindSphinx module is found

set(MARIUS_CPP_SOURCE ${CMAKE_CURRENT_LIST_DIR}/src/cpp)
set(CMAKE_MODULE_PATH "${MARIUS_CPP_SOURCE}/cmake" ${CMAKE_MODULE_PATH})
set(project_INCLUDE_DIR ${MARIUS_CPP_SOURCE}/include)
set(project_SOURCE_DIR ${MARIUS_CPP_SOURCE}/src)
set(project_TARGET_DIR ${MARIUS_CPP_SOURCE}/targets)
set(project_TEST_DIR ${CMAKE_CURRENT_LIST_DIR}/test)
set(project_DOCS_DIR ${CMAKE_CURRENT_LIST_DIR}/docs)
set(project_BINDINGS_DIR ${MARIUS_CPP_SOURCE}/python_bindings)
set(project_THIRD_PARTY_DIR ${MARIUS_CPP_SOURCE}/third_party)


set(project_WORKING_DIR ${CMAKE_CURRENT_BINARY_DIR})
add_definitions(-DMARIUS_BASE_DIRECTORY="${CMAKE_CURRENT_LIST_DIR}")
add_definitions(-DMARIUS_TEST_DIRECTORY="${project_TEST_DIR}")

if (EXISTS ${project_INCLUDE_DIR})
    file(GLOB_RECURSE project_HEADERS ${project_HEADERS} ${project_INCLUDE_DIR}/*.h)
endif ()
if (EXISTS ${project_SOURCE_DIR})
    file(GLOB_RECURSE project_SOURCES ${project_SOURCES} ${project_SOURCE_DIR}/*.cpp)
endif ()
if (EXISTS ${project_BINDINGS_DIR})
    file(GLOB_RECURSE project_BINDINGS ${project_BINDINGS} ${project_BINDINGS_DIR}/*.cpp)
endif ()

add_subdirectory(${project_THIRD_PARTY_DIR})
set_property(TARGET spdlog PROPERTY POSITION_INDEPENDENT_CODE ON)
include_directories(${project_INCLUDE_DIR})
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${GLOO_INCLUDE_DIRS})
include_directories(${project_THIRD_PARTY_DIR}/inih)
add_library(${PROJECT_NAME} SHARED ${project_SOURCES} ${project_HEADERS})

find_package(Torch REQUIRED)
# find_package(Gloo REQUIRED)
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${GLOO_LIBRARIES})
target_link_libraries(${PROJECT_NAME} spdlog)
target_link_libraries(${PROJECT_NAME} cxxopts pthread stdc++fs)

set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${project_HEADERS}")

if (EXISTS ${project_INCLUDE_DIR})
    target_include_directories(${PROJECT_NAME} PUBLIC ${project_INCLUDE_DIR})
endif ()
if (EXISTS ${project_SOURCE_DIR})
    target_include_directories(${PROJECT_NAME} PRIVATE ${project_SOURCE_DIR})
endif ()

target_compile_options(${PROJECT_NAME}
        PRIVATE -fomit-frame-pointer
        PRIVATE -O1
        )

IF(CMAKE_BUILD_TYPE MATCHES Debug AND MARIUS_USE_ASAN)
    message("Using ASAN")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address  -fsanitize=leak")
    set(CMAKE_MODULE_LINKER_FLAGS "{$CMAKE_MODULE_LINKER_FLAGS}  -fsanitize=address -fsanitize=leak")
ENDIF(CMAKE_BUILD_TYPE MATCHES Debug AND MARIUS_USE_ASAN)


IF(BUILD_DOCS)
    add_subdirectory(${project_DOCS_DIR})
ENDIF()

if (EXISTS ${project_TEST_DIR})
    enable_testing()
    add_subdirectory(${project_TEST_DIR})
endif ()

add_executable(marius_train ${project_TARGET_DIR}/marius_main.cpp)
add_executable(marius_eval ${project_TARGET_DIR}/marius_main.cpp)
add_executable(coordinator ${project_TARGET_DIR}/coordinator_main.cpp)
add_executable(worker ${project_TARGET_DIR}/worker_main.cpp)
target_link_libraries(marius_train ${PROJECT_NAME})
target_link_libraries(marius_eval ${PROJECT_NAME})
target_link_libraries(coordinator ${PROJECT_NAME})
target_link_libraries(worker ${PROJECT_NAME})

pybind11_add_module(_pymarius ${project_SOURCES} ${project_BINDINGS})

find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
message(STATUS "TORCH_PYTHON_LIBRARY: ${TORCH_PYTHON_LIBRARY}")
target_link_libraries(_pymarius PRIVATE ${PROJECT_NAME} ${TORCH_PYTHON_LIBRARY})
target_include_directories(_pymarius PRIVATE ${project_INCLUDE_DIR})
