cmake_minimum_required(VERSION 3.15...4.0)

project(gomea_library VERSION "1.1.0")
# https://github.com/CWI-EvolutionaryIntelligence/GOMEA/tree/769fa6e309c78565415eb65a8deb4c5da8f920af
#
# Modifications
#
# - lib -> extern with a custom CMakeLists for cxxopts (looked easier than removing the dependency)
# - separate include folders mirroring the source layout
# - some header fixes until the compiler was happy
#   - <Eigen> -> <Eigen/Dense>
#   - "src/..." -> "gomea/src/..."
# - the main files and sources for the benchmark problems are not included

add_subdirectory(extern/cxxopts)

add_subdirectory(src)
