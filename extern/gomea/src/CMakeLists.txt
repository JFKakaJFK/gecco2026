find_package(Eigen3 REQUIRED)   # Eigen3::Eigen

file(GLOB_RECURSE HEADER_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/*.hpp")

add_library(${PROJECT_NAME} OBJECT
    ${HEADER_LIST}
    common/linkage_config.cpp
    common/linkage_model.cpp
    common/output_statistics.cpp
    common/partial_solution.cpp
    common/solution.cpp
    discrete/Config.cpp
    discrete/gomea.cpp
    discrete/gomeaIMS.cpp
    discrete/Population.cpp
    discrete/utils.cpp
    fitness/benchmarks-discrete/deceptiveTrap_t.cpp
    fitness/benchmarks-discrete/asymmetricDeceptiveTrapBBO_t.cpp
    fitness/benchmarks-discrete/bimodalDeceptiveTrapBBO_t.cpp
    fitness/benchmarks-discrete/asymmetricDeceptiveTrap_t.cpp
    fitness/benchmarks-discrete/hierarchicalIfAndOnlyIfBBO_t.cpp
    fitness/benchmarks-discrete/bimodalDeceptiveTrap_t.cpp
    fitness/benchmarks-discrete/deceptiveTrapBBO_t.cpp
    fitness/benchmarks-discrete/hierarchicalDeceptiveTrapBBO_t.cpp
    fitness/benchmarks-discrete/maxCut_t.cpp
    fitness/benchmarks-discrete/oneMaxBBO_t.cpp
    fitness/benchmarks-discrete/NKlandscapesBBO_t.cpp
    fitness/benchmarks-discrete/oneMax_t.cpp
    fitness/benchmarks-discrete/asymmetricHierarchicalDeceptiveTrapBBO_t.cpp
    fitness/benchmarks-discrete/NKlandscapes_t.cpp
    fitness/benchmarks-discrete/maxCutBBO_t.cpp
    fitness/benchmarks-rv/sphereFunctionBBO_t.cpp
    fitness/benchmarks-rv/rosenbrockFunction_t.cpp
    fitness/benchmarks-rv/rosenbrockFunctionBBO_t.cpp
    fitness/benchmarks-rv/SOREBChainStrong_t.cpp
    fitness/benchmarks-rv/circlesInASquareBBO_t.cpp
    fitness/benchmarks-rv/SOREBChainStrongBBO_t.cpp
    fitness/benchmarks-rv/sphereFunction_t.cpp
    fitness/bbo_fitness.cpp
    fitness/fitness.cpp
    fitness/gbo_fitness.cpp
    real_valued/Config.cpp
    real_valued/distribution.cpp
    real_valued/linkage_model.cpp
    real_valued/partial_solutionRV.cpp
    real_valued/population.cpp
    real_valued/rv-gomea.cpp
    real_valued/solutionRV.cpp
    real_valued/tools.cpp
    utils/time.cpp
    utils/tools.cpp
)
# disable any warnings in the code & disable sanitizers

# fixes the assertion that fitness is somewhat descending after sorting in ascending order...
# Assertion failed: (population[selection_indices[0]]->getObjectiveValue() >= population[selection_indices[num_solutions-1]]->getObjectiveValue()), function estimateParametersForSingleBinaryMarginal, file linkage_model.cpp, line 949.
target_compile_options(${PROJECT_NAME} PRIVATE
    -w -O3 -DNDEBUG -fno-sanitize=address,undefined
)
target_link_options(${PROJECT_NAME} PRIVATE
    -fno-sanitize=address,undefined
)

target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        Eigen3::Eigen
    PUBLIC
        cxxopts
)

# for IDE support
source_group(
    TREE "${PROJECT_SOURCE_DIR}/include"
    PREFIX "Header Files"
    FILES ${HEADER_LIST}
)
