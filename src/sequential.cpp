#include <cstring>
#include <nbody/body.hpp>

template <typename... Args>
void UNUSED(Args &&...args [[maybe_unused]]) {}

int main(int argc, char **argv)
{
    UNUSED(argc, argv);
    static float gravity = 100;
    static float space = 800;
    static float radius = 5;
    static int bodies = 20;
    static float elapse = 0.001;
    static float max_mass = 50;
    BodyPool pool(static_cast<size_t>(bodies), space, max_mass);
    pool.update_for_tick(elapse, gravity, space, radius);
}
