#include <graphic/graphic.hpp>
#include <imgui_impl_sdl.h>
#include <cstring>
#include <nbody/body.hpp>
#include <mpi.h>

// #define DEBUG

template <typename... Args>
void UNUSED(Args &&...args [[maybe_unused]]) {}
void master(BodyPool &pool, float elapse, float gravity, float space, float radius);
void slave();

void master(BodyPool &pool, float max_mass, int bodies, float elapse, float gravity, float space, float radius);
void slaves();
void get_slice(int &start_body, int &end_body, int nbody, int rank, int total_rank); // get subtask, rank r get job from start_body to end_body;
void check_and_update_mpi(int rank, int nbody, int comm_size, BodyPool &pool, double radius, double gravity);
void update_for_tick_mpi(int rank, int nbody, int comm_size, BodyPool &pool, double elapse, double space, double radius);
void update_master_pool(BodyPool &master_pool, BodyPool &slave_pool, int start_body, int end_body); // master should update the pool after receiving from slaves;
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
void check_and_update_mpi(int rank, int nbody, int comm_size, BodyPool &pool, double radius, double gravity)
{
    int start_body, end_body;
    get_slice(start_body, end_body, nbody, rank, comm_size);

    if (start_body >= end_body)
        return;
    printf("I am rank %d, start body: %d, end body: %d \n (pool.size: %d)", rank, start_body, end_body, nbody);
    for (int i = start_body; i < end_body; i++)
    {

        for (int j = i + 1; j < nbody; j++)
        {
            pool.check_and_update(pool.get_body(i), pool.get_body(j), radius, gravity);
        }
    }
    return;
}
void update_for_tick_mpi(int rank, int nbody, int comm_size, BodyPool &pool, double elapse, double space, double radius)
{
    int start_body, end_body;
    get_slice(start_body, end_body, nbody, rank, comm_size);

    // if (start_body >= end_body)
    //     return;
    // printf("rank %d, update position... \n",rank);
    // for (int i = start_body; i < end_body; i++)
    for (int i = 0; i < nbody; i++)
    {
        pool.get_body(i).update_for_tick(elapse, space, radius);
    }
    return;
}

void master(Pool &pool, float max_mass, int bodies, float elapse, float gravity, float space, float radius)
{
    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    MPI_Datatype MPI_BodyPool;
    MPI_Datatype MPI_Info;

    MPI_Type_contiguous(6, MPI_FLOAT, &MPI_Info);
    MPI_Type_commit(&MPI_Info);
    MPI_Type_contiguous(7, MPI_DOUBLE, &MPI_BodyPool);
    MPI_Type_commit(&MPI_BodyPool);
    printf("in pool: %f \n", pool.vx[10]);
    Info globalInfo = {space, (float)bodies, max_mass, gravity, elapse, radius};

    MPI_Bcast(&globalInfo, 1, MPI_Info, 0, MPI_COMM_WORLD);
    pool.clear_acceleration();

    MPI_Bcast(&pool, bodies, MPI_BodyPool, 0, MPI_COMM_WORLD);
    // step 1;
#ifdef DEBUG
    check_and_update_mpi(0, bodies, comm_size, pool, radius, gravity);

    for (int i = 1; i < comm_size; i++)
    {
        MPI_Status status;
        BodyPool slave_pool(static_cast<size_t>(bodies), space, max_mass);
        MPI_Recv(&slave_pool, bodies + 1, MPI_BodyPool, i, 1, MPI_COMM_WORLD, &status);
        int source = status.MPI_SOURCE;
        int start_body, end_body;
        get_slice(start_body, end_body, bodies, 0, comm_size);

        update_master_pool(pool, slave_pool, start_body, end_body);
    }
    // step2;
    update_for_tick_mpi(0, bodies, comm_size, pool, elapse, space, radius);
#endif // DEBUG
}
void slaves()
{
    int rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    MPI_Datatype MPI_BodyPool;
    MPI_Datatype MPI_Info;

    MPI_Type_contiguous(6, MPI_FLOAT, &MPI_Info);
    MPI_Type_commit(&MPI_Info);
    MPI_Type_contiguous(7, MPI_DOUBLE, &MPI_BodyPool);
    MPI_Type_commit(&MPI_BodyPool);

    Info globalInfo;
    MPI_Bcast(&globalInfo, 1, MPI_Info, 0, MPI_COMM_WORLD);
    int bodies = globalInfo.bodies;
    // printf("Receive %d bodies \n", bodies);
    BodyPool pool(static_cast<size_t>(bodies), globalInfo.space, globalInfo.max_mass);

    MPI_Bcast(&pool, bodies, MPI_BodyPool, 0, MPI_COMM_WORLD);
    printf("Receive %f pool \n", pool.get_body(1).get_m());
#ifdef DEBUG
    check_and_update_mpi(rank, bodies, comm_size, pool, globalInfo.radius, globalInfo.gravity);

    MPI_Send(&pool, bodies, MPI_BodyPool, 0, 1, MPI_COMM_WORLD);
#endif // DEBUG
    return;
}

void get_slice(int &start_body, int &end_body, int nbody, int rank, int total_rank)
{
    int m = nbody / total_rank, rem = nbody % total_rank;
    start_body = (rank < rem) ? (m + 1) * rank : rem + m * rank;
    end_body = (rank < rem) ? start_body + m + 1 : start_body + m;
}
void update_master_pool(BodyPool &master_pool, BodyPool &slave_pool, int start_body, int end_body)
{
    for (int i = start_body; i < end_body; i++)
    {
        master_pool.get_body(i).get_ax() = slave_pool.get_body(i).get_ax();
        master_pool.get_body(i).get_ay() = slave_pool.get_body(i).get_ay();
        master_pool.get_body(i).get_x() = slave_pool.get_body(i).get_x();
        master_pool.get_body(i).get_y() = slave_pool.get_body(i).get_y();
        master_pool.get_body(i).get_vy() = slave_pool.get_body(i).get_vy();
        master_pool.get_body(i).get_vx() = slave_pool.get_body(i).get_vx();
    }
}
