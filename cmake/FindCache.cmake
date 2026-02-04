# prevent infinite recursion
if (DEFINED _FindCache_shim)
    return()
endif()
set(_FindCache_shim TRUE CACHE INTERNAL "")

# prevent adding the target multiple times
if (TARGET Cache::Cache)
    set(Cache_FOUND TRUE CACHE INTERNAL "")
    return()
endif()

find_package(Cache QUIET BYPASS_PROVIDER)
if(Cache_FOUND)
    return()
endif()

set(_extern_path "${CMAKE_CURRENT_LIST_DIR}/../extern/Cache")
if (EXISTS "${_extern_path}/CMakeLists.txt")
    message(STATUS "Using local Cache from ${_extern_path}")
    add_subdirectory("${_extern_path}" EXCLUDE_FROM_ALL)
    set(Cache_FOUND TRUE CACHE INTERNAL "")
endif()
