find_package(Eigen3 REQUIRED)   # Eigen3::Eigen
find_package(OpenRAND REQUIRED) # OpenRAND::OpenRAND
find_package(gomea_library REQUIRED) # gomea_library

file(GLOB_RECURSE HEADER_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/*.h")


# if(SKBUILD OR GOBLIN_BUILD_PYLIB)
#     add_library(${PROJECT_NAME}
#         STATIC
#         ${HEADER_LIST}
#         lib/instance.cpp
#         gp/sr.cpp
#         methods/amalgam.cpp
#         methods/mo_gomea.cpp
#     )
#     set_target_properties(lib PROPERTIES COMPILE_FLAGS "-O3")
#     set_target_properties(lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
# else()
add_library(${PROJECT_NAME}
        ${HEADER_LIST}
        lib/archive.cpp
        lib/archive_unbounded.cpp
        lib/instance.cpp
        gp/sr.cpp
        methods/amalgam.cpp
        methods/mo_gomea.cpp
    )
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
)

# users will need at least C++23
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_23)

# for IDE support
source_group(
    TREE "${PROJECT_SOURCE_DIR}/include"
    PREFIX "Header Files"
    FILES ${HEADER_LIST}
)
