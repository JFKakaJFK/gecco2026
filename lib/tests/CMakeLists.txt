find_package(doctest REQUIRED)

enable_testing()

set(deps
    goblin::lib
    doctest::doctest
    Eigen3::Eigen
    OpenRAND::OpenRAND
    gomea_library
    Cache
)

set(test_target ${PROJECT_NAME}_test_main)
add_library(${test_target} OBJECT test_main.cpp)
target_link_libraries(${test_target} PRIVATE ${deps})

set(
    LIB_TEST_SOURCES
    archive
    fitness
    solution
    solution_extension
    default_gradients
    problem
    amalgam
    so_gomea
    mo_binary
    tracked
    ims
    entropy
    mixed
    mixed_giga
    mo_mixed
    mixed_rv
    mixed_ia
    mixed_mixed
    mixed_mo_rv
    mo_rv_gomea
    gp_operators
    sr
    sr_stats
    sr_precision
    sr_gradient_step
    archive_objectives
    ctx_subfunction
    ctx_introns
    quality
)

set(LIB_OPTIMIZED_TESTS
    # mixed
    sr
    sr_stats
    mo_mixed
    mixed_mixed
    archive_objectives
    ctx_introns
    mixed_ia
)

foreach(testcase IN LISTS LIB_TEST_SOURCES)
  set(_target lib_test_${testcase})
  add_executable(${_target} ${testcase}.cpp $<TARGET_OBJECTS:${test_target}>)
  target_link_libraries(${_target} PRIVATE ${deps})
  add_test(NAME ${_target} COMMAND ${_target})

  # if(testcase IN_LIST LIB_OPTIMIZED_TESTS)
  #     target_compile_options(${_target} PRIVATE -O2 -g1)
  # endif()
endforeach()
