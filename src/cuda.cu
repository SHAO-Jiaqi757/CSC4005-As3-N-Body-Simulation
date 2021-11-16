#include <cstring>
#include <nbody/body.cuh>
#include <vector>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#define DEBUG
template <typename... Args>
void UNUSED(Args &&...args [[maybe_unused]]) {}
__device__ void get_slice(int &start_body, int &end_body, int thread_id, int thread_number);
__device__ __managed__ float gravity = 100;
__device__ __managed__ float space = 800;
__device__ __managed__ float radius = 5;
__device__ __managed__ int bodies = 200;
__device__ __managed__ float elapse = 0.1;
__device__ __managed__ float max_mass = 50;
__device__ __managed__ BodyPool *pool;
__device__ __managed__ int thread_number;

__global__ void check_and_update()
{
    int thread_id = threadIdx.x;

    int start_body, end_body;
    get_slice(start_body, end_body, thread_id, thread_number);
#ifdef DEBUG
    printf("threadIdx: %d, start: %d, end: %d\n", thread_id, start_body, end_body);
#endif
    for (size_t i = (size_t)start_body; i < (size_t)end_body; i++)
    {
        pool->get_body(i).clear_collision();
        for (size_t j = 0; j < pool->size; ++j)
        {
            if (i != j)
                pool->check_and_update_thread(pool->get_body(i), pool->get_body(j), radius, gravity);
        }
    }
    __syncthreads();

    for (size_t i = start_body; i < end_body; i++)
        pool->get_body(i).update_for_tick_thread(elapse, space, radius);
}

int main(int argc, char **argv)
{
    UNUSED(argc, argv);
    if (argc < 4)
    {
        printf("Error: Invalid Inputs \nUseage: pthread <thread_number> <bodies> <iteration> \n");
        return 0;
    }
    thread_number = atoi(argv[1]);
    bodies = atoi(argv[2]);

    int iter = atoi(argv[3]);
    int cur_iter = iter;
    pool = new BodyPool(static_cast<size_t>(bodies), space, max_mass);

    pool->clear_acceleration();
    auto begin = std::chrono::high_resolution_clock::now();
    while (cur_iter > 0)
    {
        check_and_update<<<1, bodies / thread_number>>>();
        cudaDeviceSynchronize();

        delete pool;
        cudaDeviceReset();
        cur_iter--;
#ifdef DEBUG
        for (size_t i = 0; i < pool->size; i++)
        {
            printf("%f ", pool->ax[i]);
        }
        printf("\n");
#endif
    }

    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();

    printf("thread_number(cuda): %d \n", thread_number);
    printf("body: %d \n", bodies);
    printf("iterations: %d \n", iter);
    printf("duration(ns/iter): %lu \n", duration / iter);

    return 0;
}

__device__ void get_slice(int &start_body, int &end_body, int thread_id, int thread_number)
{
    int m = pool->size / thread_number;
    int rem = pool->size % thread_number;
    start_body = (thread_id < rem) ? (m + 1) * thread_id : rem + m * thread_id;
    end_body = (thread_id < rem) ? start_body + (m + 1) : start_body + m;
}