#include <graphic/graphic.hpp>
#include <imgui_impl_sdl.h>
#include <cstring>
#include <nbody/body.hpp>
#include <pthread.h>
#include <vector>
template <typename... Args>
void UNUSED(Args &&...args [[maybe_unused]]) {}
struct thread_args
{
    int thread_id;
};
void *check_and_update_thread(void *args);
void *update_for_tick_thread(void *args);
int thread_number = 0;
static float gravity = 100;
static float space = 800;
static float radius = 5;
static int bodies = 20;
static float elapse = 0.001;
static ImVec4 color = ImVec4(1.0f, 1.0f, 0.4f, 1.0f);
static float max_mass = 50;
BodyPool pool(static_cast<size_t>(bodies), space, max_mass);
int main(int argc, char **argv)
{
    // UNUSED(argc, argv);

    if (argc < 2)
    {
        printf("Error: No <thread_number> found! \nUseage: pthread_gui <thread_number> \n");
        return 0;
    }
    thread_number = atoi(argv[1]);
    printf("thread_number: %d \n", thread_number);

    static float current_space = space;
    static float current_max_mass = max_mass;
    static int current_bodies = bodies;
    graphic::GraphicContext context{"Assignment 2"};
    context.run([&](graphic::GraphicContext *context [[maybe_unused]], SDL_Window *)
                {
        auto io = ImGui::GetIO();
        ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
        ImGui::SetNextWindowSize(io.DisplaySize);
        ImGui::Begin("Assignment 2", nullptr,
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
        }
        {
            const ImVec2 p = ImGui::GetCursorScreenPos();
            // pool.update_for_tick(elapse, gravity, space, radius);
            std::vector<pthread_t> threads(thread_number);
            std::vector<int> thread_ids(thread_number);
            pool.clear_acceleration();
            // step 1;
            for (int i = 0; i < thread_number; i++)
            {
                thread_ids[i] = i;
                pthread_create(&threads[i], nullptr, check_and_update_thread, (void *)&thread_ids[i]);
            }
            for (int i = 0; i < thread_number; i++)
            {
                pthread_join(threads[i], NULL);
            }
            // step 2;
            for (int i = 0; i < thread_number; i++)
            {
                thread_ids[i] = i;
                pthread_create(&threads[i], nullptr, update_for_tick_thread, (void *)&thread_ids[i]);
            }
            for (int i = 0; i < thread_number; i++)
            {
                pthread_join(threads[i], NULL);
            }

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

void *check_and_update_thread(void *args)
{
    int thread_id = *((int *)args);

    int n = pool.size();
    int m = n / thread_number, rem = n % thread_number;
    size_t start_body = m * thread_id;

    size_t end_body = (thread_id == thread_number - 1) ? start_body + m + rem : start_body + m;
    if (start_body >= end_body)
        return NULL;
    // printf("I am thread %d, start body: %d, end body: %d \n (pool.size: %d)", thread_id, start_body, end_body, n);
    for (size_t i = start_body; i < end_body; i++)
    {

        // pool.get_body(i).get_ax() = 0;
        // pool.get_body(i).get_ay() = 0;
        for (size_t j = i + 1; j < (size_t)n; j++)
        {
            // if (i != j)
            pool.check_and_update(pool.get_body(i), pool.get_body(j), radius, gravity);
        }
    }

    return nullptr;
}
void *update_for_tick_thread(void *args)
{
    int thread_id = *((int *)args);
    int m = pool.size() / thread_number;
    int rem = pool.size() % thread_number;
    size_t start_body = m * thread_id;
    size_t end_body = (thread_id == thread_number - 1) ? start_body + m + rem : start_body + m;
    if (start_body >= end_body)
        return NULL;
    // printf("thread %d, update position... \n", thread_id);
    for (size_t i = start_body; i < end_body; i++)
    {
        pool.get_body(i).update_for_tick(elapse, space, radius);
    }
    return nullptr;
}