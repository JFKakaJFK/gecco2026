# prevent infinite recursion
if (DEFINED _FindEigen3_shim)
    return()
endif()
set(_FindEigen3_shim TRUE CACHE INTERNAL "")

# prevent adding the target multiple times
if (TARGET Eigen3::Eigen)
    set(Eigen3_FOUND TRUE CACHE INTERNAL "")
    return()
endif()

find_package(Eigen3 QUIET BYPASS_PROVIDER)
if(Eigen3_FOUND)
    return()
endif()

set(_extern_path "${CMAKE_CURRENT_LIST_DIR}/../extern/Eigen")
if (EXISTS "${_extern_path}/CMakeLists.txt")
    message(STATUS "Using local Eigen3 from ${_extern_path}")
    add_subdirectory("${_extern_path}" EXCLUDE_FROM_ALL)
    set(Eigen3_FOUND TRUE CACHE INTERNAL "")
endif()
