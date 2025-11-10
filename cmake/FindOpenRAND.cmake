# prevent infinite recursion
if (DEFINED _FindOpenRAND_shim)
    return()
endif()
set(_FindOpenRAND_shim TRUE CACHE INTERNAL "")

# prevent adding the target multiple times
if (TARGET OpenRAND::OpenRAND)
    set(OpenRAND_FOUND TRUE CACHE INTERNAL "")
    return()
endif()

find_package(OpenRAND QUIET BYPASS_PROVIDER)
if(OpenRAND_FOUND)
    return()
endif()

set(_extern_path "${CMAKE_CURRENT_LIST_DIR}/../extern/OpenRAND")
if (EXISTS "${_extern_path}/CMakeLists.txt")
    message(STATUS "Using local OpenRAND from ${_extern_path}")
    add_subdirectory("${_extern_path}" EXCLUDE_FROM_ALL)
    set(OpenRAND_FOUND TRUE CACHE INTERNAL "")
endif()
