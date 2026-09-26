/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <cstddef>

enum class split_test_event {
    workspace_begin, workspace_ready, workspace_release, task_launch, worker_begin
};
void split_test_point(split_test_event event, std::size_t worker);
struct split_test_worker_scope {
    explicit split_test_worker_scope(std::size_t worker);
    ~split_test_worker_scope();
};
#define C50_SPLIT_TEST_POINT(Event, Worker) \
    split_test_point(split_test_event::Event, Worker)
#define C50_SPLIT_TEST_WORKER(Worker) split_test_worker_scope TestWorkerScope(Worker)
