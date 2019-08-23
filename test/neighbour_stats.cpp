#include "detail/neighbour_stats.hpp"
#include <cassert>
#include <cstdio>

template struct tcm::neighbour_stats_t<std::pair<size_t, size_t>, 8>;

auto main() -> int
{
    {
        alignas(64) std::array<int64_t, 8> xs = {
            {123, 2345, -32987, 9876, -10349, -4721, 2674, 22398}};
        for (auto x : xs) {
            std::printf("%u\n", tcm::detail::find_fast_impl<8>(xs.data(), x));
        }
        std::printf("%u\n", tcm::detail::find_fast_impl<8>(xs.data(), 2675));
        std::printf("%u\n", tcm::detail::find_fast_impl<8>(xs.data(), -1));

        std::printf("%u\n", tcm::detail::find_fast_impl<4>(xs.data(), 123));
        std::printf("%u\n", tcm::detail::find_fast_impl<4>(xs.data(), 2345));
        std::printf("%u\n", tcm::detail::find_fast_impl<4>(xs.data(), -32987));
        std::printf("%u\n", tcm::detail::find_fast_impl<4>(xs.data(), 9876));
        std::printf("%u\n", tcm::detail::find_fast_impl<4>(xs.data(), -1));
        std::printf("-------\n");
    }

    {
        struct dummy_cluster_t {};
        std::array<dummy_cluster_t, 10> cs = {};

        tcm::neighbour_stats_t<dummy_cluster_t, 10> stats;
        std::printf("%u==%u\n", 0, stats.size());

        stats.insert(11, &cs[1]);
        stats.insert(14, &cs[4]);
        stats.insert(151, &cs[5]);
        stats.insert(12, &cs[2]);
        stats.insert(16, &cs[6]);
        stats.insert(121, &cs[2]);

        for (auto const& x : stats) {
            std::printf("%li: [", x.cluster - cs.data());
            for (auto i : x.neighbours) {
                std::printf("%u, ", i);
            }
            std::printf("]\n");
        }
    }
}
