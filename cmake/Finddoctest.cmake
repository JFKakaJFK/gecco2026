# prevent infinite recursion
if (DEFINED _Finddoctest_shim)
    return()
endif()
set(_Finddoctest_shim TRUE CACHE INTERNAL "")

# prevent adding the target multiple times
if (TARGET doctest::doctest)
    set(doctest_FOUND TRUE CACHE INTERNAL "")
    return()
endif()

find_package(doctest QUIET BYPASS_PROVIDER)
if(doctest_FOUND)
    return()
endif()

set(_extern_path "${CMAKE_CURRENT_LIST_DIR}/../extern/doctest")
if (EXISTS "${_extern_path}/CMakeLists.txt")
    message(STATUS "Using local doctest from ${_extern_path}")
    add_subdirectory("${_extern_path}" EXCLUDE_FROM_ALL)
    set(doctest_FOUND TRUE CACHE INTERNAL "")
endif()
