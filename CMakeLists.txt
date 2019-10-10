#module load cmake/3.9.5
#module load intel/2019
#module load cuda/9.2
#module load openmpi/4.0.1/icc

cmake_minimum_required(VERSION 3.9)
project(puc LANGUAGES C CXX CUDA)

find_package(MPI REQUIRED)

add_executable(vadd vadd.cu)
target_compile_features(vadd PRIVATE cxx_std_11)
target_link_libraries(vadd PRIVATE MPI::MPI_C)

