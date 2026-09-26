/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */
#include <c50/c50.hpp>
#include "split_failure_hooks.hpp"
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

// Configuration is constant from task launch until every task has joined.
enum class failure_mode { none, workspace, launch, worker };
static failure_mode mode;
static std::size_t selected_worker, allocation_index, workspace_allocations;
static unsigned failed_workers;
static std::atomic<unsigned> started, finished, active;
static std::mutex rendezvous_mutex;
static std::condition_variable rendezvous;
static bool release_worker;
static thread_local bool armed;
static thread_local std::size_t allocation_calls;

extern "C" void *__real_calloc(std::size_t, std::size_t);
static bool allocation_fails() { return armed && ++allocation_calls == allocation_index; }
extern "C" void *__wrap_calloc(std::size_t n, std::size_t size)
{ return allocation_fails() ? nullptr : __real_calloc(n, size); }
void *operator new(std::size_t size)
{
    if (allocation_fails()) throw std::bad_alloc();
    if (void *p = std::malloc(size ? size : 1)) return p;
    throw std::bad_alloc();
}
void *operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void *p) noexcept { std::free(p); }
void operator delete[](void *p) noexcept { std::free(p); }
void operator delete(void *p, std::size_t) noexcept { std::free(p); }
void operator delete[](void *p, std::size_t) noexcept { std::free(p); }

static void require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}
struct worker_failure : std::exception {
    const char *what() const noexcept override { return "injected worker failure"; }
};

split_test_worker_scope::split_test_worker_scope(std::size_t)
{
    std::lock_guard<std::mutex> lock(rendezvous_mutex);
    ++active;
    ++started;
    rendezvous.notify_all();
}
split_test_worker_scope::~split_test_worker_scope()
{
    --active;
    ++finished;
}

void split_test_point(split_test_event event, std::size_t worker)
{
    if (event == split_test_event::workspace_release) {
        // Throwing from the production deleter would terminate. Abort makes an
        // early release an explicit test failure before touching worker data.
        if (active) std::abort();
    } else if (mode == failure_mode::workspace && worker == selected_worker) {
        if (event == split_test_event::workspace_begin) {
            allocation_calls = 0;
            armed = true;
        } else if (event == split_test_event::workspace_ready) {
            armed = false;
            workspace_allocations = allocation_calls;
        }
    } else if (mode == failure_mode::launch) {
        if (event == split_test_event::worker_begin && worker == 1) {
            std::unique_lock<std::mutex> lock(rendezvous_mutex);
            rendezvous.wait(lock, [] { return release_worker; });
        } else if (event == split_test_event::task_launch && worker == selected_worker) {
            if (worker > 1) {
                std::unique_lock<std::mutex> lock(rendezvous_mutex);
                rendezvous.wait(lock, [] { return started.load() > 0; });
                release_worker = true;
                rendezvous.notify_all();
            }
            throw std::system_error(std::make_error_code(std::errc::resource_unavailable_try_again));
        }
    } else if (mode == failure_mode::worker && event == split_test_event::worker_begin) {
        std::unique_lock<std::mutex> lock(rendezvous_mutex);
        rendezvous.wait(lock, [] { return started.load() == 4; });
        if (failed_workers & (1u << worker)) throw worker_failure();
    }
}

static std::string body(const c50::model &model)
{
    const auto serialized = model.serialized_data();
    return serialized.substr(serialized.find('\n') + 1);
}

int main()
{
    try {
        constexpr std::size_t rows = 10000, columns = 6;
        std::vector<double> values(rows * columns);
        std::vector<std::size_t> classes(rows);
        std::string names = "no, yes.\n";
        for (std::size_t column = 0; column < columns; ++column)
            names += "x" + std::to_string(column) + ": red, green, blue, yellow.\n";
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t column = 0; column < columns; ++column)
                values[row * columns + column] = (row / (column + 1)) % 4;
            classes[row] = values[row * columns] < 2 ? 0 : 1;
        }
        const c50::dense_dataset data(values.data(), rows, columns, classes.data());
        c50::context context;
        context.split_workers(4);
        c50::options options;
        options.subset_splits = true;
        const auto train = [&] {
            return c50::model::train(context, c50::model_kind::tree, names, data, options);
        };
        const auto expected = body(train());
        require(started > 0, "fixture did not use the parallel scheduler");
        const auto reset = [&] {
            require(active == 0 && started == finished, "operation returned before join");
            started = finished = 0;
            release_worker = false;
            armed = false;
        };
        const auto recover = [&] {
            reset();
            mode = failure_mode::none;
            auto model = train();
            require(body(model) == expected, "recovered classifier changed");
            auto predictions = model.predict(context, data);
            for (std::size_t row = 0; row < rows; ++row)
                require(predictions.class_index(row) == classes[row], "recovered prediction changed");
            reset();
        };
        for (selected_worker = 1; selected_worker < 4; ++selected_worker) {
            reset();
            mode = failure_mode::workspace;
            allocation_index = 0;
            train();
            const auto total = workspace_allocations;
            require(total > 10, "workspace allocations were not observed");
            for (allocation_index = 1; allocation_index <= total; ++allocation_index) {
                reset();
                mode = failure_mode::workspace;
                bool rejected = false;
                try { train(); } catch (const std::bad_alloc &) { rejected = true; }
                armed = false;
                require(rejected, "workspace allocation failure was ignored");
                recover();
            }
        }
        for (selected_worker = 1; selected_worker < 4; ++selected_worker) {
            mode = failure_mode::launch;
            bool rejected = false;
            try { train(); } catch (const std::system_error &error) {
                rejected = error.code() == std::errc::resource_unavailable_try_again;
            }
            require(rejected, "task launch failure was not propagated");
            if (selected_worker > 1) require(started > 0, "no preceding task was started");
            recover();
        }
        for (unsigned mask : {1u, 2u, 4u, 8u, 15u}) {
            mode = failure_mode::worker;
            failed_workers = mask;
            bool rejected = false;
            try { train(); } catch (const worker_failure &) { rejected = true; }
            require(rejected && started == 4, "worker failure was not propagated");
            recover();
        }
    } catch (const std::exception &error) {
        armed = false;
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
