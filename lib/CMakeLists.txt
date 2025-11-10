cmake_minimum_required(VERSION 3.15...4.0)

project(lib VERSION "0.1.0")

add_subdirectory(src)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR GOBLIN_BUILD_DOCS)
    find_package(Doxygen)
    if(Doxygen_FOUND)
        add_subdirectory(docs)
    else()
        message(STATUS "Doxygen not found, not building docs")
    endif()
endif()

if((CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR GOBLIN_BUILD_TESTING) AND BUILD_TESTING)
    add_subdirectory(tests)
endif()
