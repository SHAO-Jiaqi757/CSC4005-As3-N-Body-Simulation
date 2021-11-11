#include <cstring>
#include <nbody/body.hpp>
#include <pthread.h>
#include "pthread_barrier.h"
#include <vector>
#include <chrono>
template <typename... Args>
void UNUSED(Args &&...args [[maybe_unused]]) {}
void *check_and_update_thread(void *args);
void *update_for_tick_thread(void *args);
int thread_number = 0;
pthread_barrier_t barrier;
static float gravity = 100;
static float space = 500;
static float radius = 5;
static int bodies = 20;
std::vector<double> collide_posX(bodies);
std::vector<double> collide_posY(bodies);
std::vector<double> collide_vx(bodies);
std::vector<double> collide_vy(bodies);
static float elapse = 0.040;
static float max_mass = 50;
BodyPool pool(static_cast<size_t>(bodies), space, max_mass);
void get_slice(int &start_body, int &end_body, int thread_id, int thread_number);
int iter;
int main(int argc, char **argv)
{
    // UNUSED(argc, argv);

    if (argc < 4)
    {
        printf("Error: Invalid Inputs \nUseage: pthread <thread_number> <bodies> <iteration> \n");
        return 0;
    }
    thread_number = atoi(argv[1]);
    bodies = atoi(argv[2]);
    iter = atoi(argv[3]);

    collide_posX.resize(bodies);
    collide_posY.resize(bodies);
    collide_vx.resize(bodies);
    collide_vy.resize(bodies);

    int current_iter = iter;
    size_t duration = 0;
    auto begin = std::chrono::high_resolution_clock::now();
    while (current_iter > 0)
    {
        std::vector<pthread_t> threads(thread_number);
        std::vector<int> thread_ids(thread_number);
        pthread_barrier_init(&barrier, NULL, thread_number);
        pool.clear_acceleration();
        for (int i = 0; i < thread_number; i++)
        {
            thread_ids[i] = i;
            pthread_create(&threads[i], nullptr, check_and_update_thread, (void *)&thread_ids[i]);
        }
        for (int i = 0; i < thread_number; i++)
        {
            pthread_join(threads[i], NULL);
        }
        current_iter--;
    }
    auto end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();

    printf("pthread_number: %d \n", thread_number);
    printf("body: %d \n", bodies);
    printf("iterations: %d \n", iter);
    printf("speed(ns/iter): %f \n", duration / iter);
}

void *check_and_update_thread(void *args)
{
    int thread_id = *((int *)args);
    int start_body, end_body;
    get_slice(start_body, end_body, thread_id, thread_number);
    if (start_body >= end_body)
        return NULL;
    // printf("I am thread %d, start body: %d, end body: %d \n (pool.size: %d)", thread_id, start_body, end_body, n);
    for (size_t i = (size_t)start_body; i < (size_t)end_body; i++)
    {
        collide_posX[i] = 0;
        collide_posY[i] = 0;
        collide_vx[i] = 0;
        collide_vy[i] = 0;
        for (size_t j = 0; j < pool.size(); j++)
        {
            if (i != j)
                pool.check_and_update_thread(pool.get_body(i), pool.get_body(j), radius, gravity, collide_posX, collide_posY, collide_vx, collide_vy);
        }
    }
    pthread_barrier_wait(&barrier);
    for (int i = start_body; i < end_body; i++)
    {
        pool.get_body(i).update_for_tick_thread(elapse, space, radius, collide_posX[i], collide_posY[i], collide_vx[i], collide_vy[i]);
    }
    return nullptr;
}

void get_slice(int &start_body, int &end_body, int thread_id, int thread_number)
{
    int m = pool.size() / thread_number;
    int rem = pool.size() % thread_number;
    start_body = (thread_id < rem) ? (m + 1) * thread_id : rem + m * thread_id;
    end_body = (thread_id < rem) ? start_body + (m + 1) : start_body + m;
}
