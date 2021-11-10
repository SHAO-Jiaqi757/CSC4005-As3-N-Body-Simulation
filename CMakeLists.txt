cmake_minimum_required(VERSION 3.2)
project(csc4005_imgui)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-fsanitize=thread")
find_package(SDL2 REQUIRED)
find_package(Freetype REQUIRED)
find_package(MPI REQUIRED)
find_package(Threads REQUIRED)
set(CMAKE_CXX_STANDARD 20)
set(OpenGL_GL_PREFERENCE "GLVND")
find_package(OpenGL REQUIRED)

include_directories(
        include
        imgui
        imgui/backends
        ${SDL2_INCLUDE_DIRS}
        ${FREETYPE_INCLUDE_DIRS}
        ${MPI_CXX_INCLUDE_DIRS})

file(GLOB IMGUI_SRC
        imgui/*.cpp
        imgui/backends/imgui_impl_sdl.cpp
        imgui/backends/imgui_impl_opengl2.cpp
        imgui/misc/freetype/imgui_freetype.cpp
        imgui/misc/cpp/imgui_stdlib.cpp
        )
add_library(core STATIC ${IMGUI_SRC})

add_executable(sequential_gui src/sequential_gui.cpp src/graphic.cpp )
add_executable(mpi_gui src/mpi_gui.cpp src/graphic.cpp )
add_executable(pthread_gui src/pthread_gui.cpp src/graphic.cpp )
get_filename_component(FONT_PATH imgui/misc/fonts/DroidSans.ttf ABSOLUTE)
target_link_libraries(core PUBLIC
        Freetype::Freetype SDL2::SDL2 OpenGL::GL ${CMAKE_DL_LIBS} Threads::Threads ${MPI_CXX_LIBRARIES})

target_link_libraries(sequential_gui core)
target_link_libraries(mpi_gui core)
target_link_libraries(pthread_gui core)

target_compile_options(sequential_gui PRIVATE -Wextra -Wpedantic)
target_compile_options(mpi_gui PRIVATE -Wextra -Wpedantic)
target_compile_options(pthread_gui PRIVATE -Wextra -Wpedantic)

target_compile_definitions(core PUBLIC -DImDrawIdx=unsigned)

target_compile_definitions(sequential_gui PRIVATE -DFONT_PATH=\"${FONT_PATH}\")
target_compile_definitions(mpi_gui PRIVATE -DFONT_PATH=\"${FONT_PATH}\")
if (WIN32)
    target_link_options(sequential_gui BEFORE PRIVATE -lmingw32 -lSDL2main -lSDL2 -mwindows)
        target_link_options(mpi_gui BEFORE PRIVATE -lmingw32 -lSDL2main -lSDL2 -mwindows)
        target_link_options(pthread_gui BEFORE PRIVATE -lmingw32 -lSDL2main -lSDL2 -mwindows)

endif()