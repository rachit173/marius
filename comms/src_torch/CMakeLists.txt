cmake_minimum_required(VERSION 3.13)
# set(Torch_DIR "${CMAKE_LIST_DIR}/../libtorch")
set(Torch_DIR /home/rtibrewal_wisc_edu/graph_learning/marius/comms/libtorch/share/cmake/Torch)
set(Gloo_DIR /home/rtibrewal_wisc_edu/graph_learning/marius/comms/libtorch/share/cmake/Gloo)
find_package(Torch REQUIRED)
# find_package(Gloo REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} ${GLOO_CXX_FLAGS}")

if(NOT MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")
endif()
add_executable(coordinator coordinator.cc)
target_link_libraries(coordinator pthread "${TORCH_LIBRARIES}" ${GLOO_LIBRARIES})
set_property(TARGET coordinator PROPERTY CXX_STANDARD 17)


add_executable(worker worker.cc)
target_link_libraries(worker pthread "${TORCH_LIBRARIES}" ${GLOO_LIBRARIES})
set_property(TARGET worker PROPERTY CXX_STANDARD 17)