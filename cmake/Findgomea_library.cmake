# prevent infinite recursion
if (DEFINED _Findgomea_library_shim)
    return()
endif()
set(_Findgomea_library_shim TRUE CACHE INTERNAL "")

# prevent adding the target multiple times
if (TARGET gomea_library)
    set(gomea_library_FOUND TRUE CACHE INTERNAL "")
    return()
endif()

find_package(gomea_library QUIET BYPASS_PROVIDER)
if(gomea_library_FOUND)
    return()
endif()

set(_extern_path "${CMAKE_CURRENT_LIST_DIR}/../extern/gomea_library")
if (EXISTS "${_extern_path}/CMakeLists.txt")
    message(STATUS "Using local gomea_library from ${_extern_path}")
    add_subdirectory("${_extern_path}" EXCLUDE_FROM_ALL)
    set(gomea_library_FOUND TRUE CACHE INTERNAL "")
endif()
