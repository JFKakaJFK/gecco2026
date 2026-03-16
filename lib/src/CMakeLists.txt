find_package(Eigen3 REQUIRED)   # Eigen3::Eigen
find_package(OpenRAND REQUIRED) # OpenRAND::OpenRAND
find_package(gomea_library REQUIRED) # gomea_library

file(GLOB_RECURSE HEADER_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/*.h")

set(LIB_SOURCES
    ${HEADER_LIST}
    gp/sr.cpp
    lib/archive.cpp
    lib/archive_adaptive.cpp
    lib/archive_unbounded.cpp
    lib/entropy.cpp
    lib/fitness.cpp
    lib/instance.cpp
    lib/solution.cpp
    methods/amalgam.cpp
    methods/mo_gomea.cpp
)

if(GOBLIN_HAS_CUDA)
    find_package(CUDAToolkit REQUIRED)
    list(APPEND LIB_SOURCES
        gp/gpu_evaluation/evaluate.cu
        gp/gpu_evaluation/memory.cu
    )
endif()

add_library(${PROJECT_NAME} ${LIB_SOURCES})

# endif()
# disable any warnings in the amalgam code
# also there is a memory bug in initializePopulationsAndFitnessValues() amalgam.cpp:1915...
set_source_files_properties(
    methods/amalgam.cpp
    PROPERTIES
        COMPILE_FLAGS "-w -fno-sanitize=address,undefined"
)
# disable any warnings in the mo gomea code
set_source_files_properties(
    methods/mo_gomea.cpp
    PROPERTIES
        COMPILE_FLAGS "-w -fno-sanitize=address,undefined"
)
add_library(goblin::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        Eigen3::Eigen
        OpenRAND::OpenRAND
        gomea_library
        Cache
)

if(GOBLIN_HAS_CUDA)
    target_compile_definitions(${PROJECT_NAME} PUBLIC GOBLIN_HAS_CUDA)
    target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart)
endif()

# users will need at least C++23
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_23)

# for IDE support
source_group(
    TREE "${PROJECT_SOURCE_DIR}/include"
    PREFIX "Header Files"
    FILES ${HEADER_LIST}
)
