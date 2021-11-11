cmake_minimum_required(VERSION 3.2)
project(as3_bench)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-fsanitize=thread")
find_package(MPI REQUIRED)
find_package(Threads REQUIRED)
set(CMAKE_CXX_STANDARD 20)

include_directories(
        include
        ${MPI_CXX_INCLUDE_DIRS})

add_library(core STATIC src/pthread_barrier.c)


add_executable(sequential src/sequential.cpp )
add_executable(mpi src/mpi.cpp )
add_executable(pthread src/pthread.cpp src/pthread_barrier.c )
add_executable(openmp src/openmp.cpp )
add_executable(openmp_mpi src/openmp_mpi.cpp )


target_link_libraries(core PUBLIC
        Threads::Threads ${MPI_CXX_LIBRARIES})

target_link_libraries(sequential core)
target_link_libraries(mpi core)
target_link_libraries(pthread core)
target_link_libraries(openmp core)
target_link_libraries(openmp  ${OpenMP_CXX_LIBRARIES})
target_link_libraries(openmp_mpi core)
target_link_libraries(openmp_mpi  ${OpenMP_CXX_LIBRARIES})

target_compile_options(sequential PRIVATE -Wextra -Wpedantic)
target_compile_options(mpi PRIVATE -Wextra -Wpedantic)
target_compile_options(pthread PRIVATE -Wextra -Wpedantic)
target_compile_options(openmp PRIVATE -Wextra -Wpedantic)
target_compile_options(openmp_mpi PRIVATE -Wextra -Wpedantic)
target_compile_definitions(core PUBLIC -DImDrawIdx=unsigned)
endif()