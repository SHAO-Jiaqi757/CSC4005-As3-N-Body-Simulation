#include <graphic/graphic.hpp>
#include <imgui_impl_sdl.h>
#include <cstring>
#include <nbody/body.hpp>
#include <mpi.h>

#define DEBUG

template <typename... Args>
void UNUSED(Args &&...args [[maybe_unused]]) {}
void master(BodyPool &pool, float max_mass, int bodies, float elapse, float gravity, float space, float radius);
void slaves();
void sendrecv_results(BodyPool &pool, int comm_size, int rank, int bodies);
void scatter_pool(BodyPool &pool, int bodies);
void get_slice(int &start_body, int &end_body, int nbody, int rank, int total_rank); // get subtask, rank r get job from start_body to end_body;
void check_and_update_mpi(int rank, int nbody, int comm_size, BodyPool &pool, double radius, double gravity, double elapse, double space);
struct Info
{
    float space;
    float bodies;
    float max_mass;
    float gravity;
    float elapse;
    float radius;
} __attribute__((packed));

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    UNUSED(argc, argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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

    if (rank == 0)
    {

        BodyPool pool(static_cast<size_t>(bodies), space, max_mass);
        // MPI_Bcast(&pool, 1, MPI_BodyPool, 0, comm);
        graphic::GraphicContext context{"Assignment 3"};
        context.run([&](graphic::GraphicContext *context [[maybe_unused]], SDL_Window *)
                    {
        auto io = ImGui::GetIO();
        ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
        ImGui::SetNextWindowSize(io.DisplaySize);
        ImGui::Begin("Assignment 3", nullptr,
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
        }
        {
            const ImVec2 p = ImGui::GetCursorScreenPos();
            master(pool, max_mass, bodies, elapse, gravity, space, radius);

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
            slaves();
        }
    }
    MPI_Finalize();
}
void check_and_update_mpi(int rank, int nbody, int comm_size, BodyPool &pool, double radius, double gravity, double elapse, double space)
{
    int start_body, end_body;
    get_slice(start_body, end_body, nbody, rank, comm_size);

    if (start_body >= end_body)
        return;
    for (int i = start_body; i < end_body; i++)
    {
        for (int j = 0; j < nbody; j++)
        {
            pool.check_and_update(pool.get_body(i), pool.get_body(j), radius, gravity);
        }
        pool.get_body(i).update_for_tick(elapse, space, radius);
    }
    return;
}

void master(BodyPool &pool, float max_mass, int bodies, float elapse, float gravity, float space, float radius)
{
    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    MPI_Datatype MPI_BodyPool;
    MPI_Datatype MPI_Info;

    MPI_Type_contiguous(6, MPI_FLOAT, &MPI_Info);
    MPI_Type_commit(&MPI_Info);
    Info globalInfo = {space, (float)bodies, max_mass, gravity, elapse, radius};

    MPI_Bcast(&globalInfo, 1, MPI_Info, 0, MPI_COMM_WORLD);

    pool.clear_acceleration();
    scatter_pool(pool, bodies);
#ifdef DEBUG
    check_and_update_mpi(0, bodies, comm_size, pool, radius, gravity, elapse, space);
    sendrecv_results(pool, comm_size, 0, bodies);
#endif // DEBUG
}
void slaves()
{
    int rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Datatype MPI_Info;
    MPI_Type_contiguous(6, MPI_FLOAT, &MPI_Info);
    MPI_Type_commit(&MPI_Info);

    Info globalInfo;
    MPI_Bcast(&globalInfo, 1, MPI_Info, 0, MPI_COMM_WORLD);
    int bodies = globalInfo.bodies;
    // printf("Receive %d bodies \n", bodies);
    BodyPool pool(static_cast<size_t>(bodies), globalInfo.space, globalInfo.max_mass);

    scatter_pool(pool, bodies); // receive pool, data is ready for calculation

#ifdef DEBUG
    // update pool.ax, pool.ay and record collision in collide_vx, collide_vy, collide_x, collide_y;
    check_and_update_mpi(rank, bodies, comm_size, pool, globalInfo.radius, globalInfo.gravity, globalInfo.elapse, globalInfo.space);
    // send result to rank 0;
    sendrecv_results(pool, comm_size, rank, bodies);
#endif // DEBUG
    return;
}

void get_slice(int &start_body, int &end_body, int nbody, int rank, int total_rank)
{
    int m = nbody / total_rank, rem = nbody % total_rank;
    start_body = (rank < rem) ? (m + 1) * rank : rem + m * rank;
    end_body = (rank < rem) ? start_body + m + 1 : start_body + m;
}
void scatter_pool(BodyPool &pool, int bodies)
{

    MPI_Bcast(pool.x.data(), bodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(pool.y.data(), bodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(pool.vx.data(), bodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(pool.vy.data(), bodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(pool.m.data(), bodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(pool.ay.data(), bodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(pool.ax.data(), bodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}
void sendrecv_results(BodyPool &pool, int comm_size, int rank, int bodies)
{
    if (rank == 0)
    {
        for (int source = 1; source < comm_size; source++)
        {
            int start_body, end_body;
            get_slice(start_body, end_body, bodies, source, comm_size);
            int cnts = end_body - start_body;
            MPI_Recv(pool.ay.data() + start_body, cnts, MPI_DOUBLE, source, 6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(pool.ax.data() + start_body, cnts, MPI_DOUBLE, source, 7, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(pool.vx.data() + start_body, cnts, MPI_DOUBLE, source, 8, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(pool.vy.data() + start_body, cnts, MPI_DOUBLE, source, 9, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(pool.x.data() + start_body, cnts, MPI_DOUBLE, source, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(pool.y.data() + start_body, cnts, MPI_DOUBLE, source, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    else
    {
        int start_body, end_body;
        get_slice(start_body, end_body, bodies, rank, comm_size);
        int cnts = end_body - start_body;
        // MPI_Send( const void* buf , int count , MPI_Datatype datatype , int dest , int tag , MPI_Comm comm);
        MPI_Send(pool.ay.data() + start_body, cnts, MPI_DOUBLE, 0, 6, MPI_COMM_WORLD);
        MPI_Send(pool.ax.data() + start_body, cnts, MPI_DOUBLE, 0, 7, MPI_COMM_WORLD);
        MPI_Send(pool.vx.data() + start_body, cnts, MPI_DOUBLE, 0, 8, MPI_COMM_WORLD);
        MPI_Send(pool.vy.data() + start_body, cnts, MPI_DOUBLE, 0, 9, MPI_COMM_WORLD);
        MPI_Send(pool.x.data() + start_body, cnts, MPI_DOUBLE, 0, 10, MPI_COMM_WORLD);
        MPI_Send(pool.y.data() + start_body, cnts, MPI_DOUBLE, 0, 11, MPI_COMM_WORLD);
    }
}