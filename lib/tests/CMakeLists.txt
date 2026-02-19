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
    amalgam
    archive_objectives
    archive
    ctx_gpu_repr
    ctx_introns
    ctx_subfunction
    default_gradients
    entropy
    evaluate
    fitness
    ga_sr
    gp_operators
    ims
    misc
    mixed_ia
    mixed_mixed
    mixed_rv
    mixed
    mo_binary
    mo_mixed
    problem
    so_gomea
    solution
    sr_gradient_step
    sr
    tracked
)

foreach(testcase IN LISTS LIB_TEST_SOURCES)
    set(_target lib_test_${testcase})
    add_executable(${_target} ${testcase}.cpp $<TARGET_OBJECTS:${test_target}>)
    target_link_libraries(${_target} PRIVATE ${deps})
    add_test(NAME ${_target} COMMAND ${_target})
endforeach()

target_compile_options(lib_test_mixed PRIVATE -O2 -g)
target_compile_options(lib_test_sr PRIVATE -O2 -g)
target_compile_options(lib_test_ga_sr PRIVATE -O2 -g)
target_compile_options(lib_test_mo_mixed PRIVATE -O2 -g)
target_compile_options(lib_test_mixed_mixed PRIVATE -O2 -g)
target_compile_options(lib_test_archive_objectives PRIVATE -O2 -g)
target_compile_options(lib_test_ctx_introns PRIVATE -O2 -g)
target_compile_options(lib_test_mixed_ia PRIVATE -O2 -g)
