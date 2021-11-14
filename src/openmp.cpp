#include <cstring>
#include <nbody/body.hpp>
#include <vector>
#include <chrono>
#include <omp.h>
#define Get_Thread_Number
template <typename... Args>
void UNUSED(Args &&...args [[maybe_unused]]) {}
void check_and_update();

float gravity = 100;
float space = 500;
float radius = 5;
int bodies = 20;
std::vector<double> collide_posX(bodies);
std::vector<double> collide_posY(bodies);
std::vector<double> collide_vx(bodies);
std::vector<double> collide_vy(bodies);
float elapse = 0.040;
float max_mass = 50;
int thread_number, iter;
BodyPool pool(static_cast<size_t>(bodies), space, max_mass);
int main(int argc, char **argv)
{
    // UNUSED(argc, argv);
#ifdef Get_Thread_Number
    if (argc < 3)
    {
        printf("Error: Invalid input! \nUseage: openmp <bodies> <iteration>\n");
        return 0;
    }
    thread_number = atoi(argv[1]);
    bodies = atoi(argv[2]);
    iter = atoi(argv[3]);
    pool = BodyPool{static_cast<size_t>(bodies), space, max_mass};
    collide_posX.resize(bodies);
    collide_posY.resize(bodies);
    collide_vx.resize(bodies);
    collide_vy.resize(bodies);
#endif // Get_Thread_Number
    // omp_set_num_threads(thread_number);
    size_t duration = 0;
    int current_iter = iter;
    auto begin = std::chrono::high_resolution_clock::now();
    while (current_iter > 0)
    {
        check_and_update();
        current_iter--;
    }
    auto end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
    printf("openmp_thread: %d \n", thread_number);
    printf("body: %d \n", bodies);
    printf("iterations: %d \n", iter);
    printf("duration(ns/iter): %lu \n", duration / iter);
}

void check_and_update()
{
    pool.clear_acceleration();
#pragma omp parallel for num_threads(thread_number) shared(pool, collide_vx, collide_vy, collide_posX, collide_posY)
    for (int i = 0; i < bodies; i++)
    {
        // printf("thread number: %d \n", thread_number);
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
#pragma omp parallel for num_threads(thread_number) shared(pool, collide_vx, collide_vy, collide_posX, collide_posY)
    for (int i = 0; i < bodies; i++)
    {
        pool.get_body(i).update_for_tick_thread(elapse, space, radius, collide_posX[i], collide_posY[i], collide_vx[i], collide_vy[i]);
    }
}
