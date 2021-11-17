#include <graphic/graphic.hpp>
#include <imgui_impl_sdl.h>
#include <cstring>
#include <nbody/body.cuh>
#include <cuda_runtime.h>
// #define Get_Thread_Number
template <typename... Args>
void UNUSED(Args &&...args [[maybe_unused]]);
__device__ __managed__ float gravity = 100;
__device__ __managed__ float space = 800;
__device__ __managed__ float radius = 5;
__device__ __managed__ int bodies = 200;
__device__ __managed__ float elapse = 0.1;
__device__ __managed__ float max_mass = 50;
__device__ __managed__ BodyPool *pool;
__device__ __managed__ int thread_number;
static ImVec4 color = ImVec4(1.0f, 1.0f, 0.4f, 1.0f);

__global__ void check_and_update()
{
    int thread_id = threadIdx.x;
    size_t body_indx = thread_id;

#ifdef DEBUG
    printf("threadIdx: %d, start: %d, end: %d\n", thread_id, start_body, end_body);
#endif
    pool->get_body(body_indx).clear_collision();
    for (size_t j = 0; j < pool->size; ++j)
    {
        if (body_indx != j)
            pool->check_and_update_thread(pool->get_body(body_indx), pool->get_body(j), radius, gravity);
    }
    __syncthreads();

    // for (size_t i = start_body; i < end_body; i++)
    pool->get_body(body_indx).update_for_tick_thread(elapse, space, radius);
}
int main(int argc, char **argv)
{
    // UNUSED(argc, argv);

    static float current_space = space;
    static float current_max_mass = max_mass;
    static int current_bodies = bodies;
    graphic::GraphicContext context{"Assignment 3"};
    context.run([&](graphic::GraphicContext *context [[maybe_unused]], SDL_Window *)
                {
        auto io = ImGui::GetIO();
        ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
        ImGui::SetNextWindowSize(io.DisplaySize);
        ImGui::Begin("Assignment 3", nullptr,
                     ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize);
        ImDrawList *draw_list = ImGui::GetWindowDrawList();
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate,
                    ImGui::GetIO().Framerate);
        ImGui::DragFloat("Space", &current_space, 10, 200, 1600, "%f");
        ImGui::DragFloat("Gravity", &gravity, 0.5, 0, 1000, "%f");
        ImGui::DragFloat("Radius", &radius, 0.5, 2, 20, "%f");
        ImGui::DragInt("Bodies", &current_bodies, 1, 2, 100, "%d");
        ImGui::DragFloat("Elapse", &elapse, 0.001, 0.001, 10, "%f");
        ImGui::DragFloat("Max Mass", &current_max_mass, 0.5, 5, 100, "%f");
        ImGui::ColorEdit4("Color", &color.x);
        if (current_space != space || current_bodies != bodies || current_max_mass != max_mass)
        {
            space = current_space;
            bodies = current_bodies;
            max_mass = current_max_mass;
            pool = new BodyPool{static_cast<size_t>(bodies), space, max_mass};
           
        }
        {
            const ImVec2 p = ImGui::GetCursorScreenPos();
            pool->clear_acceleration();
            check_and_update<<<1, bodies>>>();
            cudaDeviceSynchronize();
            // drawing...
            for (size_t i = 0; i < pool->size; ++i)
            {
                auto body = pool->get_body(i);
                auto x = p.x + static_cast<float>(body.get_x());
                auto y = p.y + static_cast<float>(body.get_y());
                draw_list->AddCircleFilled(ImVec2(x, y), radius, ImColor{color});
            }
        }
        ImGui::End(); });
    delete pool;
    cudaDeviceReset();
}
