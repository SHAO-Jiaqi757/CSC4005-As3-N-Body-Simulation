cmake_minimum_required(VERSION 3.2)
project(as3_bench)

set(CMAKE_CXX_STANDARD 17)
find_package(SDL2 REQUIRED)
find_package(Freetype REQUIRED)
find_package(MPI REQUIRED)
find_package(Threads REQUIRED)
set(OpenGL_GL_PREFERENCE "GLVND")
find_package(OpenGL REQUIRED)

find_package(OpenMP REQUIRED)
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
include_directories(
        include
        ${SDL2_INCLUDE_DIRS}
        ${FREETYPE_INCLUDE_DIRS}
        ${MPI_CXX_INCLUDE_DIRS})


add_library(core STATIC src/pthread_barrier.c)


add_executable(sequential src/sequential.cpp)
add_executable(mpi src/mpi.cpp )
add_executable(pthread src/pthread.cpp src/pthread_barrier.c )
add_executable(openmp src/openmp.cpp )
add_executable(openmp_mpi src/openmp_mpi.cpp )


target_link_libraries(core PUBLIC
        Freetype::Freetype SDL2::SDL2 OpenGL::GL ${CMAKE_DL_LIBS} Threads::Threads ${MPI_CXX_LIBRARIES})

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

if (WIN32)
    target_link_options(sequential BEFORE PRIVATE -lmingw32 -lSDL2main -lSDL2 -mwindows)
        target_link_options(mpi BEFORE PRIVATE -lmingw32 -lSDL2main -lSDL2 -mwindows)
        target_link_options(pthread BEFORE PRIVATE -lmingw32 -lSDL2main -lSDL2 -mwindows)
                target_link_options(openmp BEFORE PRIVATE -lmingw32 -lSDL2main -lSDL2 -mwindows)
                target_link_options(openmp_mpi BEFORE PRIVATE -lmingw32 -lSDL2main -lSDL2 -mwindows)


endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message("release mode include cuda")
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin=/opt/rh/devtoolset-10/root/usr/bin/gcc -gencode=arch=compute_75,code=[sm_75,compute_75]")
    add_executable(cuda src/cuda.cu)
    target_link_libraries(cuda core)
    set_target_properties(cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(
       cuda 
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES OFF
        )
endif()