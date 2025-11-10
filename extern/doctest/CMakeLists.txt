cmake_minimum_required(VERSION 3.15...4.0)

project(doctest VERSION "2.4.12")

# for IDE support
file(GLOB_RECURSE HEADER_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/*.h")

add_library(${PROJECT_NAME} INTERFACE ${HEADER_LIST})
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME} SYSTEM INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
