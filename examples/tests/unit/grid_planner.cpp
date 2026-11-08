#include <grid.hpp>
#include <cassert>

int main() {
    dsdecompx::DomainShape cube;
    cube.lengths = {10.0, 10.0, 10.0};
    const auto balanced = dsdecompx::choose_process_grid(
        8, cube, {}, dsdecompx::GridPolicy::Balanced);
    assert(balanced.px == 2 && balanced.py == 2 && balanced.pz == 2);

    dsdecompx::TopologyInfo unavailable;
    const auto fallback = dsdecompx::choose_process_grid(
        8, cube, unavailable, dsdecompx::GridPolicy::TopologyAware);
    assert(fallback.px == balanced.px && fallback.py == balanced.py &&
           fallback.pz == balanced.pz);

    const auto dynamic = dsdecompx::choose_process_grid(
        8, cube, {}, dsdecompx::GridPolicy::Dynamic);
    assert(dynamic.px == balanced.px && dynamic.py == balanced.py &&
           dynamic.pz == balanced.pz);
}
