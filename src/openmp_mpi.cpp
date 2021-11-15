#include <cstring>
#include <nbody/body.hpp>
#include <mpi.h>
#include <chrono>
#include <omp.h>
#define DEBUG

template <typename... Args>
void UNUSED(Args &&...args [[maybe_unused]]) {}
void master(BodyPool &pool, float max_mass, int bodies, float elapse, float gravity, float space, float radius, int &iter, int comm_size);
int slaves(int comm_size);
void sendrecv_results(BodyPool &pool, int comm_size, int rank, int bodies);
void scatter_pool(BodyPool &pool, int bodies);
void get_slice(int &start_body, int &end_body, int nbody, int rank, int total_rank); // get subtask, rank r get job from start_body to end_body;
void check_and_update_mpi(int rank, int nbody, int comm_size, BodyPool &pool, double radius, double gravity, double elapse, double space);
int thread_number;
struct Info
{
    float space;
    float bodies;
    float max_mass;
    float gravity;
    float elapse;
    float radius;
    float iter;
} __attribute__((packed));

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    // UNUSED(argc, argv);
    int rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    float gravity = 100;
    float space = 500;
    float radius = 5;
    float elapse = 0.040;
    float max_mass = 50;
    size_t duration = 0;
    if (rank == 0)
    {
        if (argc < 3)
        {
            printf("Error: Invalid input!\nUseage: openmp_mpi <bodies> <iteration> \n");
            return 0;
        }
        int bodies = atoi(argv[1]); // change by input
        int iter = atoi(argv[2]);
        int current_iter = iter;
        auto begin = std::chrono::high_resolution_clock::now();
        while (current_iter > 0)
        {
            BodyPool pool(static_cast<size_t>(bodies), space, max_mass);
            current_iter--;
            master(pool, max_mass, bodies, elapse, gravity, space, radius, current_iter, comm_size);
        }
        auto end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();

        printf("cores(omp+mpi): %d \n", comm_size);
        printf("openmp_thread: %d \n", thread_number);
        printf("body: %d \n", bodies);
        printf("iterations: %d \n", iter);
        printf("duration(ns/iter): %lu \n", duration / iter);
    }
    else
    {
        while (1)
        {
            if (!slaves(comm_size))
                break;
        }
    }
    MPI_Finalize();
}

void check_and_update_mpi(int rank, int nbody, int comm_size, BodyPool &pool, double radius, double gravity, double elapse, double space)
{
    int start_body, end_body;
    get_slice(start_body, end_body, nbody, rank, comm_size);
    std::vector<double> collide_x(nbody), collide_y(nbody), collide_vy(nbody), collide_vx(nbody);
    if (start_body >= end_body)
        return;
#pragma omp parallel for shared(pool, collide_vx, collide_vy, collide_x, collide_y)
    for (int i = start_body; i < end_body; i++)
    {
        for (int j = 0; j < nbody; j++)
        {
            if (i == j)
                continue;
            pool.check_and_update_thread(pool.get_body(i), pool.get_body(j), radius, gravity, collide_x, collide_y, collide_vx, collide_vy);
        }
    }
#pragma omp parallel for shared(pool, collide_vx, collide_vy, collide_x, collide_y)
    for (int i = start_body; i < end_body; i++)
    {
        thread_number = omp_get_num_threads();
        pool.get_body(i).update_for_tick_thread(elapse, space, radius, collide_x[i], collide_y[i], collide_vx[i], collide_vy[i]);
    }
    return;
}

void master(BodyPool &pool, float max_mass, int bodies, float elapse, float gravity, float space, float radius, int &iter, int comm_size)
{

    MPI_Datatype MPI_Info;

    MPI_Type_contiguous(7, MPI_FLOAT, &MPI_Info);
    MPI_Type_commit(&MPI_Info);
    Info globalInfo = {space, (float)bodies, max_mass, gravity, elapse, radius, (float)iter};

    MPI_Bcast(&globalInfo, 1, MPI_Info, 0, MPI_COMM_WORLD);
    if (iter <= 0)
        return;
    pool.clear_acceleration();
    scatter_pool(pool, bodies);
// step 1;
#ifdef DEBUG
    check_and_update_mpi(0, bodies, comm_size, pool, radius, gravity, elapse, space);

    sendrecv_results(pool, comm_size, 0, bodies);
    // step2;
#endif // DEBUG
}
int slaves(int comm_size)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Datatype MPI_Info;
    MPI_Type_contiguous(7, MPI_FLOAT, &MPI_Info);
    MPI_Type_commit(&MPI_Info);

    Info globalInfo;

    MPI_Bcast(&globalInfo, 1, MPI_Info, 0, MPI_COMM_WORLD);
    int iter = globalInfo.iter;
    if (iter <= 0)
        return 0;
    int bodies = globalInfo.bodies;
    // printf("Receive %d bodies \n", bodies);
    BodyPool pool(static_cast<size_t>(bodies), globalInfo.space, globalInfo.max_mass);
    scatter_pool(pool, bodies); // send pool, data is ready for calculation

#ifdef DEBUG
    check_and_update_mpi(rank, bodies, comm_size, pool, globalInfo.radius, globalInfo.gravity, globalInfo.elapse, globalInfo.space);
    // send result to rank 0;
    sendrecv_results(pool, comm_size, rank, bodies);
#endif // DEBUG
    return 1;
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