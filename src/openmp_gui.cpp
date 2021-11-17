#include <graphic/graphic.hpp>
#include <imgui_impl_sdl.h>
#include <cstring>
#include <nbody/body.hpp>
#include <pthread.h>
#include <vector>
// #define Get_Thread_Number
template <typename... Args>
void UNUSED(Args &&...args [[maybe_unused]]) {}
void check_and_update();

static float gravity = 100;
static float space = 800;
static float radius = 5;
static int bodies = 200;
std::vector<double> collide_posX(bodies);
std::vector<double> collide_posY(bodies);
std::vector<double> collide_vx(bodies);
std::vector<double> collide_vy(bodies);
static float elapse = 0.001;
static ImVec4 color = ImVec4(1.0f, 1.0f, 0.4f, 1.0f);
static float max_mass = 50;
BodyPool pool(static_cast<size_t>(bodies), space, max_mass);
int main(int argc, char **argv)
{
    UNUSED(argc, argv);
#ifdef Get_Thread_Number
    if (argc < 2)
    {
        printf("Error: No <thread_number> found! \nUseage: pthread_gui <thread_number> \n");
        return 0;
    }
    thread_number = atoi(argv[1]);
    printf("thread_number: %d \n", thread_number);
#endif // Get_Thread_Number
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
            pool = BodyPool{static_cast<size_t>(bodies), space, max_mass};
            collide_posX.resize(bodies);
            collide_posY.resize(bodies);
            collide_vx.resize(bodies);
            collide_vy.resize(bodies);
        }
        {
            const ImVec2 p = ImGui::GetCursorScreenPos();
            check_and_update();
            // drawing...
            for (size_t i = 0; i < pool.size(); ++i)
            {
                auto body = pool.get_body(i);
                auto x = p.x + static_cast<float>(body.get_x());
                auto y = p.y + static_cast<float>(body.get_y());
                draw_list->AddCircleFilled(ImVec2(x, y), radius, ImColor{color});
            }
        }
        ImGui::End(); });
}

void check_and_update()
{
    pool.clear_acceleration();
#pragma omp parallel for shared(pool, collide_vx, collide_vy, collide_posX, collide_posY)
    for (int i = 0; i < bodies; i++)
    {
        collide_posX[i] = 0;
        collide_posY[i] = 0;
        collide_vx[i] = 0;
        collide_vy[i] = 0;
        for (int j = 0; j < bodies; j++)
        {
            if (i != j)
                pool.check_and_update_thread(pool.get_body(i), pool.get_body(j), radius, gravity, collide_posX, collide_posY, collide_vx, collide_vy);
        }
    }
#pragma omp barrier
#pragma omp parallel for shared(pool, collide_vx, collide_vy, collide_posX, collide_posY)
    for (int i = 0; i < bodies; i++)
    {
        pool.get_body(i).update_for_tick_thread(elapse, space, radius, collide_posX[i], collide_posY[i], collide_vx[i], collide_vy[i]);
    }
}
