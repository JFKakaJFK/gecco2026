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
    amalgam
    archive
    archive_objectives
    ctx_gpu_repr
    ctx_introns
    ctx_subfunction
    default_gradients
    entropy
    fitness
    gp_operators
    ims
    launch_config
    mixed_ia
    mixed_mixed
    mixed_rv
    mixed
    mo_binary
    mo_mixed
    problem
    quality
    solution
    solution_extension
    so_gomea
    mixed_giga
    mixed_mo_rv
    mo_rv_gomea
    sr
    sr_stats
    sr_precision
    sr_gradient_step
    tracked
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

if(GOBLIN_HAS_CUDA)
  list(APPEND LIB_TEST_SOURCES evaluate)
endif()

foreach(testcase IN LISTS LIB_TEST_SOURCES)
  set(_target lib_test_${testcase})
  add_executable(${_target} ${testcase}.cpp $<TARGET_OBJECTS:${test_target}>)
  target_link_libraries(${_target} PRIVATE ${deps})
  add_test(NAME ${_target} COMMAND ${_target})

  # if(testcase IN_LIST LIB_OPTIMIZED_TESTS)
  #     target_compile_options(${_target} PRIVATE -O2 -g1)
  # endif()
endforeach()
