cmake_minimum_required(VERSION 2.8)
project(kalman_filters)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Armadillo REQUIRED)
include_directories(/usr/include/python2.7)

add_executable(simulate src/simulate.cpp)
target_link_libraries(simulate ${ARMADILLO_LIBRARIES} libpython2.7.so)
