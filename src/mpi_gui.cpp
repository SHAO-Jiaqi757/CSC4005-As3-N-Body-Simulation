#include <graphic/graphic.hpp>
#include <imgui_impl_sdl.h>
#include <cstring>
#include <nbody/body.hpp>
#include <mpi.h>

template <typename... Args>
void UNUSED(Args &&...args [[maybe_unused]]) {}
void master(BodyPool &pool, float elapse, float gravity, float space, float radius);
void slave();

struct Info
{
    float space;
    float bodies;
    float max_mass;
    float gravity;
    float elapse;
    float radius;
};
int main(int argc, char **argv)
{
    UNUSED(argc, argv);
    static float gravity = 100;
    static float space = 800;
    static float radius = 5;
    static int bodies = 20;
    static float elapse = 0.001;
    static ImVec4 color = ImVec4(1.0f, 1.0f, 0.4f, 1.0f);
    static float max_mass = 50;
    static float current_space = space;
    static float current_max_mass = max_mass;
    static int current_bodies = bodies;

    int rank, comm_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    MPI_Datatype MPI_BodyPool;
    MPI_Datatype MPI_Info;

    MPI_Type_contiguous(6, MPI_DOUBLE, &MPI_Info);
    MPI_Type_commit(&MPI_Info);
    MPI_Type_contiguous(7, MPI_DOUBLE, &MPI_BodyPool);
    MPI_Type_commit(&MPI_BodyPool);

    if (rank == 0)
    {

        Info globalInfo = {.space = space, .bodies = (float)bodies, .gravity = gravity, .max_mass = max_mass, .elapse = elapse, .radius = radius};
        MPI_Bcast(&globalInfo, 1, MPI_Info, 0, comm);
        BodyPool pool(static_cast<size_t>(bodies), space, max_mass);
        // MPI_Bcast(&pool, 1, MPI_BodyPool, 0, comm);
        graphic::GraphicContext context{"Assignment 2"};
        context.run([&](graphic::GraphicContext *context [[maybe_unused]], SDL_Window *)
                    {
        auto io = ImGui::GetIO();
        ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
        ImGui::SetNextWindowSize(io.DisplaySize);
        ImGui::Begin("Assignment 2", nullptr,
                     ImGuiWindowFlags_NoMove
                     | ImGuiWindowFlags_NoCollapse
                     | ImGuiWindowFlags_NoTitleBar
                     | ImGuiWindowFlags_NoResize);
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
        if (current_space != space || current_bodies != bodies || current_max_mass != max_mass) {
            space = current_space;
            bodies = current_bodies;
            max_mass = current_max_mass;
            pool = BodyPool{static_cast<size_t>(bodies), space, max_mass};
            globalInfo = {.bodies = (float)bodies, .max_mass = max_mass, .elapse= elapse, .gravity = gravity, .space = space, .radius=radius};
            MPI_Bcast(&globalInfo, 1, MPI_Info, 0, comm);
            // MPI_Bcast(&pool, 1, MPI_BodyPool, 0, comm);
        }
        {
            const ImVec2 p = ImGui::GetCursorScreenPos();
            
            pool.check_and_update_mpi(gravity, radius, rank, comm_size);

            int step1_single;
            MPI_Status status;
            MPI_Recv(&step1_single, 1, MPI_INT, MPI_ANY_SOURCE, 0, comm, &status);

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
    else
    {
        while (1)
        {
            Info globalInfo;
            MPI_Bcast(&globalInfo, 1, MPI_Info, 0, comm);
            // MPI_Bcast(&pool, 1, MPI_BodyPool, 0, comm);

            BodyPool pool(static_cast<size_t>(globalInfo.bodies), globalInfo.space, globalInfo.max_mass);
            pool.check_and_update_mpi(globalInfo.gravity, globalInfo.radius, rank, comm_size);

            int step1_single = 1;
            MPI_Send(&step1_single, 1, MPI_INT, 0, 0, comm);
        }
    }
    MPI_Finalize();
}
