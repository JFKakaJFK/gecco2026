find_package(doctest REQUIRED)

enable_testing()

set(deps
    goblin::lib
    doctest::doctest
    Eigen3::Eigen
    OpenRAND::OpenRAND
    gomea_library
)

set(test_target ${PROJECT_NAME}_test_main)
add_library(${test_target} test_main.cpp)
target_link_libraries(${test_target} PRIVATE ${deps})

set(
    LIB_TEST_SOURCES
    archive
    fitness
    solution
    default_gradients
    problem
    amalgam
    so_gomea
    mo_binary
    tracked
    ims
    entropy
    mixed
    mo_mixed
    mixed_rv
    mixed_mixed
    gp_operators
    sr
    sr_gradient_step
    archive_objectives
    eval_kernel
    ctx_subfunction
    # ctx_introns
)

foreach(testcase IN LISTS LIB_TEST_SOURCES)
    set(_target lib_test_${testcase})
    add_executable(${_target} ${testcase}.cpp $<TARGET_OBJECTS:${test_target}>)
    target_link_libraries(${_target} PRIVATE ${deps})
    add_test(NAME ${_target} COMMAND ${_target})
endforeach()

target_compile_options(lib_test_mixed PRIVATE -O2 -g)
target_compile_options(lib_test_sr PRIVATE -O2 -g)
target_compile_options(lib_test_mo_mixed PRIVATE -O2 -g)
target_compile_options(lib_test_mixed_mixed PRIVATE -O2 -g)
target_compile_options(lib_test_archive_objectives PRIVATE -O2 -g)
target_compile_options(lib_test_ctx_introns PRIVATE -O2 -g)
