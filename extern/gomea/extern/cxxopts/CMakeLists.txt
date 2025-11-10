cmake_minimum_required(VERSION 3.15...4.0)

project(cxxopts VERSION "3.1.1")

# for IDE support
file(GLOB_RECURSE HEADER_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/*.hpp")

add_library(${PROJECT_NAME} INTERFACE ${HEADER_LIST})

target_include_directories(${PROJECT_NAME} INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
