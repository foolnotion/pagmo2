/* Copyright 2017-2021 PaGMO development team

This file is part of the PaGMO library.

The PaGMO library is free software; you can redistribute it and/or modify
it under the terms of either:

  * the GNU Lesser General Public License as published by the Free
    Software Foundation; either version 3 of the License, or (at your
    option) any later version.

or

  * the GNU General Public License as published by the Free Software
    Foundation; either version 3 of the License, or (at your option) any
    later version.

or both in parallel, as here.

The PaGMO library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.

You should have received copies of the GNU General Public License and the
GNU Lesser General Public License along with the PaGMO library.  If not,
see https://www.gnu.org/licenses/. */

#include "pagmo/rng.hpp"
#include "pagmo/utils/multi_objective.hpp"
#define BOOST_TEST_MODULE nsga2_test
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <boost/lexical_cast.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <filesystem>
#include <scn/scn.h>
#include <ranges>
#include <string_view>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/nsga2.hpp>
#include <pagmo/io.hpp>
#include <pagmo/problems/dtlz.hpp>
#include <pagmo/problems/hock_schittkowski_71.hpp>
#include <pagmo/problems/inventory.hpp>
#include <pagmo/problems/rosenbrock.hpp>
#include <pagmo/problems/zdt.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/types.hpp>

#define ANKERL_NANOBENCH_IMPLEMENT
#include "nanobench.h"

using namespace pagmo;

BOOST_AUTO_TEST_CASE(nsga2_algorithm_construction)
{
    nsga2 user_algo{1u, 0.95, 10., 0.01, 50., 32u};
    BOOST_CHECK_NO_THROW(nsga2{});
    BOOST_CHECK(user_algo.get_verbosity() == 0u);
    BOOST_CHECK(user_algo.get_seed() == 32u);
    // BOOST_CHECK((user_algo.get_log() == moead::log_type{}));

    // Check the throws
    // Wrong cr
    BOOST_CHECK_THROW((nsga2{1u, 1., 10., 0.01, 50., 32u}), std::invalid_argument);
    BOOST_CHECK_THROW((nsga2{1u, -1., 10., 0.01, 50., 32u}), std::invalid_argument);
    // Wrong m
    BOOST_CHECK_THROW((nsga2{1u, .95, 10., 1.1, 50., 32u}), std::invalid_argument);
    BOOST_CHECK_THROW((nsga2{1u, .95, 10., -1.1, 50., 32u}), std::invalid_argument);
    // Wrong eta_m
    BOOST_CHECK_THROW((nsga2{1u, .95, 100.1, 0.01, 50., 32u}), std::invalid_argument);
    BOOST_CHECK_THROW((nsga2{1u, .95, .98, 0.01, 50., 32u}), std::invalid_argument);
    // Wrong eta_m
    BOOST_CHECK_THROW((nsga2{1u, .95, 10., 0.01, 100.1, 32u}), std::invalid_argument);
    BOOST_CHECK_THROW((nsga2{1u, .95, 10., 0.01, .98, 32u}), std::invalid_argument);
}

struct mo_equal_bounds {
    /// Fitness
    vector_double fitness(const vector_double &) const
    {
        return {0., 0.};
    }
    vector_double::size_type get_nobj() const
    {
        return 2u;
    }
    /// Problem bounds
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{0., 0.}, {1., 0.}};
    }
};

BOOST_AUTO_TEST_CASE(nsga2_evolve_test)
{
    // We check that the problem is checked to be suitable
    // Some bound is equal
    BOOST_CHECK_THROW(nsga2{10u}.evolve(population{problem{mo_equal_bounds{}}, 0u}), std::invalid_argument);
    // stochastic
    BOOST_CHECK_THROW((nsga2{}.evolve(population{inventory{}, 5u, 23u})), std::invalid_argument);
    // constrained prob
    BOOST_CHECK_THROW((nsga2{}.evolve(population{hock_schittkowski_71{}, 5u, 23u})), std::invalid_argument);
    // single objective prob
    BOOST_CHECK_THROW((nsga2{}.evolve(population{rosenbrock{}, 5u, 23u})), std::invalid_argument);
    // wrong population size
    BOOST_CHECK_THROW((nsga2{}.evolve(population{zdt{}, 3u, 23u})), std::invalid_argument);
    BOOST_CHECK_THROW((nsga2{}.evolve(population{zdt{}, 50u, 23u})), std::invalid_argument);

    // We check for deterministic behaviour if the seed is controlled
    // we treat the last three components of the decision vector as integers
    // to trigger all cases
    dtlz udp{1u, 10u, 3u};

    population pop1{udp, 52u, 23u};
    population pop2{udp, 52u, 23u};
    population pop3{udp, 52u, 23u};

    nsga2 user_algo1{10u, 0.95, 10., 0.01, 50., 32u};
    user_algo1.set_verbosity(1u);
    pop1 = user_algo1.evolve(pop1);

    BOOST_CHECK(user_algo1.get_log().size() > 0u);

    nsga2 user_algo2{10u, 0.95, 10., 0.01, 50., 32u};
    user_algo2.set_verbosity(1u);
    pop2 = user_algo2.evolve(pop2);

    BOOST_CHECK(user_algo1.get_log() == user_algo2.get_log());

    user_algo2.set_seed(32u);
    pop3 = user_algo2.evolve(pop3);

    BOOST_CHECK(user_algo1.get_log() == user_algo2.get_log());

    // We evolve for many-objectives and trigger the output with the ellipses
    udp = dtlz{1u, 12u, 7u};
    population pop4{udp, 52u, 23u};
    pop4 = user_algo2.evolve(pop4);
}

BOOST_AUTO_TEST_CASE(nsga2_setters_getters_test)
{
    nsga2 user_algo{1u, 0.95, 10., 0.01, 50., 32u};
    user_algo.set_verbosity(200u);
    BOOST_CHECK(user_algo.get_verbosity() == 200u);
    user_algo.set_seed(23456u);
    BOOST_CHECK(user_algo.get_seed() == 23456u);
    BOOST_CHECK(user_algo.get_name().find("NSGA-II") != std::string::npos);
    BOOST_CHECK(user_algo.get_extra_info().find("Verbosity") != std::string::npos);
    // BOOST_CHECK_NO_THROW(user_algo.get_log());
}

auto nsga = [](auto& random, pop_size_t n, auto problem, non_dominated_sorting_algorithm_type a, unsigned verbosity = 1u) {
    auto seed = std::uniform_int_distribution<uint32_t>()(random);
    algorithm algo{nsga2(100u, 0.95, 10., 0.01, 50., seed, a)};
    algo.set_verbosity(verbosity);
    population pop{problem, n, 32u};
    pop = algo.evolve(pop);
};

auto bench_problem = [](auto& rd, auto& bench, auto problem, auto n, auto a, unsigned verb = 1u) {
    std::vector<std::string> names = { "BOS", "DS", "EB", "ES", "HS", "MS", "FS", "RS", "RO" };
    auto name = names[static_cast<int>(a)] + ';' + std::to_string(n) + ';' + std::to_string(problem.get_nobj()) + ';' + problem.get_name();
    bench.run(name, [&](){  nsga(rd, n, problem, a, verb); });
};


BOOST_AUTO_TEST_CASE(nsga2_write_pop)
{
    for (auto n = 20000; n <= 20000; n += 1000) {
        for (auto m = 2; m <= 2; ++m) {
            pagmo::dtlz problem(2u, m+1, m);
            pagmo::detail::random_engine_type rd{1234};
            nsga(rd, n, problem, non_dominated_sorting_algorithm_type::merge_sort);
            std::cout << "n = " << n << ", m = " << m << " done.\n";
        }
    }
}

BOOST_AUTO_TEST_CASE(nsga2_write_pop_9000_9)
{
    auto const n{9000};
    auto const m{9};

    pagmo::dtlz problem(1u, m+1, m);
    pagmo::detail::random_engine_type rd{1234};
    nsga(rd, n, problem, non_dominated_sorting_algorithm_type::rank_intersect_sort);
    std::cout << "n = " << n << ", m = " << m << " done.\n";
}

BOOST_AUTO_TEST_CASE(nsga2_dtlz1_test_m2_rs)
{
    ankerl::nanobench::Bench bench;
    auto const n = 20'000;
    auto const m = 2;
    pagmo::dtlz problem(1u, m+1, m);
    pagmo::detail::random_engine_type rd{1234};
    nsga(rd, n, problem, non_dominated_sorting_algorithm_type::rank_intersect_sort, 1u);
    // bench_problem(rd, bench, problem, n, non_dominated_sorting_algorithm::rank_intersect_sort);
}

BOOST_AUTO_TEST_CASE(nsga2_dtlz1_test_m2_mnds)
{
    ankerl::nanobench::Bench bench;
    auto const n = 20'000;
    auto const m = 2;
    pagmo::dtlz problem(1u, m+1, m);
    pagmo::detail::random_engine_type rd{1234};
    nsga(rd, n, problem, non_dominated_sorting_algorithm_type::merge_sort, 1u);
    // bench_problem(rd, bench, problem, n, non_dominated_sorting_algorithm::merge_sort);
}

BOOST_AUTO_TEST_CASE(nsga2_dtlz1_test_m2_bos)
{
    ankerl::nanobench::Bench bench;
    auto const n = 20'000;
    auto const m = 2;
    pagmo::dtlz problem(1u, m+1, m);
    pagmo::detail::random_engine_type rd{1234};
    nsga(rd, n, problem, non_dominated_sorting_algorithm_type::best_order_sort, 1u);
    // bench_problem(rd, bench, problem, n, non_dominated_sorting_algorithm::merge_sort);
}

BOOST_AUTO_TEST_CASE(nsga2_dtlz1_test_m3_rs)
{
    ankerl::nanobench::Bench bench;
    auto const n = 20'000;
    auto const m = 3;
    pagmo::dtlz problem(1u, m+1, m);
    std::random_device rd;
    nsga(rd, n, problem, non_dominated_sorting_algorithm_type::rank_intersect_sort);
    // bench_problem(rd, bench, problem, n, non_dominated_sorting_algorithm::rank_intersect_sort);
}

BOOST_AUTO_TEST_CASE(nsga2_dtlz1_test_m3_mnds)
{
    ankerl::nanobench::Bench bench;
    auto const n = 20'000;
    auto const m = 3;
    pagmo::dtlz problem(1u, m+1, m);
    std::random_device rd;
    nsga(rd, n, problem, non_dominated_sorting_algorithm_type::merge_sort);
    // bench_problem(rd, bench, problem, n, non_dominated_sorting_algorithm::merge_sort);
}

BOOST_AUTO_TEST_CASE(nsga2_dtlz2_test_m2_rs)
{
    ankerl::nanobench::Bench bench;
    auto const n = 20'000;
    auto const m = 2;
    pagmo::dtlz problem(2u, m+1, m);
    pagmo::detail::random_engine_type rd{1234};
    nsga(rd, n, problem, non_dominated_sorting_algorithm_type::rank_intersect_sort);
    // bench_problem(rd, bench, problem, n, non_dominated_sorting_algorithm::rank_intersect_sort);
}

BOOST_AUTO_TEST_CASE(nsga2_dtlz2_test_m2_ro)
{
    ankerl::nanobench::Bench bench;
    auto const n = 20'000;
    auto const m = 2;
    pagmo::dtlz problem(2u, m+1, m);
    pagmo::detail::random_engine_type rd{1234};
    nsga(rd, n, problem, non_dominated_sorting_algorithm_type::rank_ordinal_sort);
    // bench_problem(rd, bench, problem, n, non_dominated_sorting_algorithm::rank_intersect_sort);
}

BOOST_AUTO_TEST_CASE(nsga2_dtlz2_test_m2_mnds)
{
    ankerl::nanobench::Bench bench;
    auto const n = 20'000;
    auto const m = 2;
    pagmo::dtlz problem(2u, m+1, m);
    pagmo::detail::random_engine_type rd{1234};
    nsga(rd, n, problem, non_dominated_sorting_algorithm_type::merge_sort);
    // bench_problem(rd, bench, problem, n, non_dominated_sorting_algorithm::merge_sort);
}

BOOST_AUTO_TEST_CASE(nsga2_dtlz2_test_m2_bos)
{
    ankerl::nanobench::Bench bench;
    auto const n = 20'000;
    auto const m = 2;
    pagmo::dtlz problem(2u, m+1, m);
    pagmo::detail::random_engine_type rd{1234};
    nsga(rd, n, problem, non_dominated_sorting_algorithm_type::best_order_sort);
    // bench_problem(rd, bench, problem, n, non_dominated_sorting_algorithm::merge_sort);
}

BOOST_AUTO_TEST_CASE(nsga2_dtlz2_test_m3_rs)
{
    ankerl::nanobench::Bench bench;
    auto const n = 20'000;
    auto const m = 3;
    pagmo::dtlz problem(2u, m+1, m);
    pagmo::detail::random_engine_type rd{1234};
    nsga(rd, n, problem, non_dominated_sorting_algorithm_type::rank_intersect_sort);
    // bench_problem(rd, bench, problem, n, non_dominated_sorting_algorithm::rank_intersect_sort);
}

BOOST_AUTO_TEST_CASE(nsga2_dtlz2_test_m3_mnds)
{
    ankerl::nanobench::Bench bench;
    auto const n = 20'000;
    auto const m = 3;
    pagmo::dtlz problem(2u, m+1, m);
    pagmo::detail::random_engine_type rd{1234};
    nsga(rd, n, problem, non_dominated_sorting_algorithm_type::merge_sort);
    // bench_problem(rd, bench, problem, n, non_dominated_sorting_algorithm::merge_sort);
}

BOOST_AUTO_TEST_CASE(nsga2_dtlz2_test_m4_rs)
{
    ankerl::nanobench::Bench bench;
    auto const n = 20'000;
    auto const m = 4;
    pagmo::dtlz problem(2u, m+1, m);
    pagmo::detail::random_engine_type rd{1234};
    nsga(rd, n, problem, non_dominated_sorting_algorithm_type::rank_intersect_sort);
    // bench_problem(rd, bench, problem, n, non_dominated_sorting_algorithm::rank_intersect_sort);
}

BOOST_AUTO_TEST_CASE(nsga2_dtlz2_test_m4_mnds)
{
    ankerl::nanobench::Bench bench;
    auto const n = 20'000;
    auto const m = 4;
    pagmo::dtlz problem(2u, m+1, m);
    pagmo::detail::random_engine_type rd{1234};
    nsga(rd, n, problem, non_dominated_sorting_algorithm_type::merge_sort);
    // bench_problem(rd, bench, problem, n, non_dominated_sorting_algorithm::merge_sort);
}

BOOST_AUTO_TEST_CASE(nsga2_dtlz1_test)
{
    ankerl::nanobench::Bench bench;
    std::random_device rd;
    std::vector<non_dominated_sorting_algorithm_type> algs {
        non_dominated_sorting_algorithm_type::rank_intersect_sort,
        non_dominated_sorting_algorithm_type::merge_sort
    };
    for (auto m = 2; m <= 20; ++m) {
        pagmo::dtlz problem(1u, m+1, m);
        for (auto n = 1000; n <= 20000; n += 1000) {
            for (auto a : algs) {
                bench_problem(rd, bench, problem, n, a);
            }
        }
    }
    std::ofstream f("dtlz1.csv");
    bench.render(ankerl::nanobench::templates::csv(), f);
    f.close();
}

BOOST_AUTO_TEST_CASE(nsga2_dtlz2_test)
{
    ankerl::nanobench::Bench bench;
    std::random_device rd;
    std::vector<non_dominated_sorting_algorithm_type> algs {
        non_dominated_sorting_algorithm_type::rank_intersect_sort,
        non_dominated_sorting_algorithm_type::merge_sort
    };
    for (auto m = 2; m <= 2; ++m) {
        pagmo::dtlz problem(2u, m+1, m);
        for (auto n = 1000; n <= 20000; n += 1000) {
            for (auto a : algs) {
                bench_problem(rd, bench, problem, n, a);
            }
        }
    }
    std::ofstream f("dtlz2.csv");
    bench.render(ankerl::nanobench::templates::csv(), f);
    f.close();
}

BOOST_AUTO_TEST_CASE(nsga2_zdt5_test)
{
    algorithm algo{nsga2(100u, 0.95, 10., 0.01, 50., 32u, non_dominated_sorting_algorithm_type::rank_intersect_sort)};
    algo.set_verbosity(1u);
    algo.set_seed(23456u);
    population pop{zdt(1u, 10u), 1000u, 32u};
    pop = algo.evolve(pop);
    // for (decltype(pop.size()) i = 0u; i < pop.size(); ++i) {
    //     auto x = pop.get_x()[i];
    //     BOOST_CHECK(std::all_of(x.begin(), x.end(), [](double el) { return (el == std::floor(el)); }));
    // }
}

BOOST_AUTO_TEST_CASE(nsga2_serialization_test)
{
    // Make one evolution
    problem prob{zdt{1u, 30u}};
    population pop{prob, 40u, 23u};
    algorithm algo{nsga2{10u, 0.95, 10., 0.01, 50., 32u}};
    algo.set_verbosity(1u);
    algo.set_seed(1234u);
    pop = algo.evolve(pop);

    // Store the string representation of p.
    std::stringstream ss;
    auto before_text = boost::lexical_cast<std::string>(algo);
    auto before_log = algo.extract<nsga2>()->get_log();
    // Now serialize, deserialize and compare the result.
    {
        boost::archive::binary_oarchive oarchive(ss);
        oarchive << algo;
    }
    // Change the content of p before deserializing.
    algo = algorithm{};
    {
        boost::archive::binary_iarchive iarchive(ss);
        iarchive >> algo;
    }
    auto after_text = boost::lexical_cast<std::string>(algo);
    auto after_log = algo.extract<nsga2>()->get_log();
    BOOST_CHECK_EQUAL(before_text, after_text);
    BOOST_CHECK(before_log == after_log);
    // so we implement a close check
    BOOST_CHECK(before_log.size() > 0u);
    for (auto i = 0u; i < before_log.size(); ++i) {
        BOOST_CHECK_EQUAL(std::get<0>(before_log[i]), std::get<0>(after_log[i]));
        BOOST_CHECK_EQUAL(std::get<1>(before_log[i]), std::get<1>(after_log[i]));
        for (auto j = 0u; j < 2u; ++j) {
            BOOST_CHECK_CLOSE(std::get<2>(before_log[i])[j], std::get<2>(after_log[i])[j], 1e-8);
        }
    }
}

BOOST_AUTO_TEST_CASE(bfe_usage_test)
{
    // 1 - Algorithm with bfe disabled
    problem prob{dtlz(1, 10, 2)};
    nsga2 uda1{nsga2{10}};
    uda1.set_verbosity(1u);
    uda1.set_seed(23u);
    // 2 - Instantiate
    algorithm algo1{uda1};

    // 3 - Instantiate populations
    population pop{prob, 24, 32u};
    population pop1{prob, 24, 456u};
    population pop2{prob, 24, 67345u};

    // 4 - Evolve the population
    pop1 = algo1.evolve(pop);

    // 5 - new algorithm that is bfe enabled
    nsga2 uda2{nsga2{10}};
    uda2.set_verbosity(1u);
    uda2.set_seed(23u);
    uda2.set_bfe(bfe{}); // This will use the default bfe.
    // 6 - Instantiate a pagmo algorithm
    algorithm algo2{uda2};

    // 7 - Evolve the population
    pop2 = algo2.evolve(pop);
    BOOST_CHECK(algo1.extract<nsga2>()->get_log() == algo2.extract<nsga2>()->get_log());
}

BOOST_AUTO_TEST_CASE(nsga2_dtlz2_point_benchmark)
{
    namespace fs = std::filesystem;
    namespace nb = ankerl::nanobench;

    std::string path = "./csv";

    nb::Bench bench;

    for (const auto& entry : fs::directory_iterator(path)) {
        std::string name = entry.path().string();
        if (name.find("rs") == std::string::npos) { continue; }
        std::cout << name << std::endl;
        std::size_t n{}; // population size
        std::size_t m{}; // number of objectives
        std::ignore = scn::scan(name, "./csv/nsga2_DTLZ2_{}_{}_rs.csv", n, m);

        // read part
        std::vector<std::vector<std::vector<double>>> gen;
        std::ifstream fs(name);
        std::string line;
        while (std::getline(fs, line)) {
            std::vector<std::vector<double>> pop;
            auto c = 0UL;
            std::vector<double> ind(m);
            for (auto const sv : std::views::split(line, ',')) {
                if (c == m-1) {
                    c = 0;
                    pop.push_back(std::move(ind));
                    ind = std::vector<double>(m);
                } else {
                    std::string s(sv.begin(), sv.end());
                    double v{}; std::ignore = scn::scan(s, "{}", v);
                    ind[c++] = v;
                }
            }
            gen.push_back(pop);
        }
        //values[name] = gen;

        // benchmark part
        for (auto i = 0; i < std::ssize(gen); ++i) {
            auto n = gen[i].size();
            auto m = gen[i].front().size();

            auto individuals = gen[i];

            // handle duplicate individuals
            std::vector<int> indices(n); std::iota(indices.begin(), indices.end(), 0);
            std::stable_sort(indices.begin(), indices.end(), [&](auto const& a, auto const& b){ return std::ranges::lexicographical_compare(individuals[a], individuals[b]); });

            std::vector<bool> duplicate(n, false);
            for (auto i = 0; i < n; ++i) {
                auto j = i+1;
                for (; j < n && individuals[indices[i]] == individuals[indices[j]]; ++j) {
                    duplicate[j] = true;
                }
                i = j;
            }

            //for(auto i = individuals.begin(); i < individuals.end(); ) {
            //    i->Rank = 0;
            //    auto j = i + 1;
            //    for (; j < individuals.end() && i->Fitness == j->Fitness; ++j) {
            //        j->Rank = 1;
            //    }
            //    i = j;
            //}
            auto r = std::stable_partition(indices.begin(), indices.end(), [&](auto i) { return !duplicate[i]; });
            std::vector<std::vector<double>> pop;
            for (auto it = indices.begin(); it < r; ++it) { pop.push_back(individuals[*it]); }

            auto label = std::to_string(n) + ";" + std::to_string(m) + ";" + std::to_string(i);
            bench.run("RS;" + label + ";DTLZ2", [&]() {
                rank_intersect_sorting(pop);
            });

            bench.run("MS;" + label + ";DTLZ2", [&]() {
                merge_sorting(pop);
            });
        }
    }


    std::ofstream out("./dtlz2-benchmark.csv");
    bench.render(nb::templates::csv(), out);
}
