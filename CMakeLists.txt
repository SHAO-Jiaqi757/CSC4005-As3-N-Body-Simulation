cmake_minimum_required(VERSION 3.2)
project(as3_bench)

set(CMAKE_CXX_STANDARD 17)
find_package(SDL2 REQUIRED)
find_package(Freetype REQUIRED)
find_package(MPI REQUIRED)
find_package(Threads REQUIRED)
set(OpenGL_GL_PREFERENCE "GLVND")
find_package(OpenGL REQUIRED)

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