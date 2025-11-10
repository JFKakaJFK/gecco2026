# prevent infinite recursion
if (DEFINED _Findgoblin::lib_shim)
    return()
endif()
set(_Findgoblin::lib_shim TRUE CACHE INTERNAL "")

# prevent adding the target multiple times
if (TARGET goblin::lib)
    set(goblin::lib_FOUND TRUE CACHE INTERNAL "")
    return()
endif()

find_package(goblin::lib QUIET BYPASS_PROVIDER)
if(goblin::lib_FOUND)
    return()
endif()

set(_extern_path "${CMAKE_CURRENT_LIST_DIR}/../lib")
if (EXISTS "${_extern_path}/CMakeLists.txt")
    message(STATUS "Using local goblin::lib from ${_extern_path}")
    add_subdirectory("${_extern_path}" EXCLUDE_FROM_ALL)
    set(goblin::lib_FOUND TRUE CACHE INTERNAL "")
endif()
