cmake_minimum_required(VERSION 3.15...3.27)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

project(_sitsfeats LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------
find_package(Python 3.11
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

# ---------------------------------------------------------------------------
# nanobind (installed as a Python package; locate its CMake config)
# ---------------------------------------------------------------------------
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)

find_package(nanobind CONFIG REQUIRED)

# ---------------------------------------------------------------------------
# Eigen (header-only). Use a system install when available, otherwise fetch a
# pinned release so wheels stay self-contained.
# ---------------------------------------------------------------------------
find_package(Eigen3 3.4 QUIET NO_MODULE)

if(NOT Eigen3_FOUND)
  message(STATUS "Eigen not found locally; fetching a pinned release...")
  include(FetchContent)

  set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
  set(BUILD_TESTING OFF CACHE BOOL "" FORCE)

  FetchContent_Declare(Eigen3
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz)
  FetchContent_MakeAvailable(Eigen3)
endif()

# ---------------------------------------------------------------------------
# Extension module
# ---------------------------------------------------------------------------
file(GLOB SOURCE_FILES "src/*.cpp")

nanobind_add_module(_sitsfeats NB_STATIC ${SOURCE_FILES})
target_link_libraries(_sitsfeats PRIVATE Eigen3::Eigen)
target_compile_definitions(_sitsfeats PRIVATE VERSION_INFO=${PROJECT_VERSION})

install(TARGETS _sitsfeats LIBRARY DESTINATION .)
