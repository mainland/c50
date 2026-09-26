/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */
#include <c50/c50.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>

static std::size_t fail_at, calls;
static bool armed;
extern "C" void *__real_calloc(std::size_t, std::size_t);
extern "C" void *__real_malloc(std::size_t);
extern "C" void *__real_realloc(void *, std::size_t);
extern "C" char *__real_strdup(const char *);
static bool fail() { return armed && ++calls == fail_at; }
extern "C" void *__wrap_calloc(std::size_t n, std::size_t size)
{ return fail() ? nullptr : __real_calloc(n, size); }
extern "C" void *__wrap_malloc(std::size_t size)
{ return fail() ? nullptr : __real_malloc(size); }
extern "C" void *__wrap_realloc(void *p, std::size_t size)
{ return fail() ? nullptr : __real_realloc(p, size); }
extern "C" char *__wrap_strdup(const char *s)
{ return fail() ? nullptr : __real_strdup(s); }
void *operator new(std::size_t size)
{
    if (fail()) throw std::bad_alloc();
    if (void *p = __real_malloc(size ? size : 1)) return p;
    throw std::bad_alloc();
}
void *operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void *p) noexcept { std::free(p); }
void operator delete[](void *p) noexcept { std::free(p); }
void operator delete(void *p, std::size_t) noexcept { std::free(p); }
void operator delete[](void *p, std::size_t) noexcept { std::free(p); }

static const std::string names = "no, yes.\nx: continuous.\n";
static const std::string data = "0, no\n1, no\n2, yes\n3, yes\n";
static const std::string tree = "entries=\"1\"\ntype=\"0\" class=\"no\" freq=\"2,0\"\n";

template<class F> static void exercise(const char *label, c50::context &context, F operation)
{
    context = c50::context();
    calls = fail_at = 0;
    armed = true;
    operation();
    armed = false;
    const auto total = calls;
    for (std::size_t index = 1; index <= total; ++index) {
        context = c50::context();
        calls = 0;
        fail_at = index;
        std::fprintf(stderr, "%s allocation %zu/%zu\n", label, index, total);
        bool rejected = false;
        armed = true;
        try { operation(); }
        catch (const std::bad_alloc &) { rejected = true; }
        catch (...) { armed = false; throw; }
        armed = false;
        if (!rejected) throw std::runtime_error("allocation failure was ignored");
        operation(); // The same workspace must remain usable after each failure.
    }
}

int main()
{
    try {
        c50::context context;
        exercise("load", context, [&] { c50::model::load(context, c50::model_kind::tree, names, tree); });
        exercise("train", context, [&] { c50::model::train(context, c50::model_kind::tree, names, data); });
        exercise("implicit boolean", context, [&] {
            c50::model::train(context, c50::model_kind::tree,
                "no, yes.\nx: continuous.\nlarge := x > 1.\n", data);
        });
        exercise("implicit string", context, [&] {
            c50::model::train(context, c50::model_kind::tree,
                "no, yes.\ncolor: red, blue.\nis red := color = \"red\".\n",
                "red, no\nred, no\nblue, yes\nblue, yes\n");
        });
        std::string nested = "color = \"red\"";
        for (unsigned level = 0; level < 60; ++level)
            nested = "color = \"red\" and (" + nested + ")";
        const std::string nested_names =
            "no, yes.\ncolor: red, blue.\nis red := " + nested + ".\n";
        exercise("implicit growth", context, [&] {
            c50::model::train(context, c50::model_kind::tree, nested_names,
                "red, no\nred, no\nblue, yes\nblue, yes\n");
        });
        std::string wide_names = "no, yes.\n";
        for (unsigned index = 0; index < 105; ++index)
            wide_names += "x" + std::to_string(index) + ": continuous.\n";
        exercise("wide schema", context, [&] {
            c50::model::load(context, c50::model_kind::tree, wide_names, tree);
        });
        std::string levels = "no, yes.\nx: ";
        for (unsigned index = 0; index < 105; ++index)
            levels += (index ? ", " : "") + std::string("v") + std::to_string(index);
        levels += ".\n";
        exercise("category schema", context, [&] {
            c50::model::load(context, c50::model_kind::tree, levels, tree);
        });
        std::string classes, frequencies;
        for (unsigned index = 0; index < 105; ++index) {
            classes += (index ? ", " : "") + std::string("c") + std::to_string(index);
            frequencies += index ? ",1" : "1";
        }
        classes += ".\nx: continuous.\n";
        const std::string class_tree = "entries=\"1\"\ntype=\"0\" class=\"c0\" freq=\"" + frequencies + "\"\n";
        exercise("class schema", context, [&] {
            c50::model::load(context, c50::model_kind::tree, classes, class_tree);
        });
        const std::string long_model = "id=\"" + std::string(30000, 'x') + "\"\n" + tree;
        exercise("property growth", context, [&] {
            c50::model::load(context, c50::model_kind::tree, names, long_model);
        });
        exercise("threshold schema", context, [&] {
            c50::model::load(context, c50::model_kind::tree,
                "y: 1, 2.\ny: continuous.\nx: continuous.\n",
                "entries=\"1\"\ntype=\"0\" class=\"y <= 1\" freq=\"1,0,0\"\n");
        });
        auto model = c50::model::load(context, c50::model_kind::tree, names, tree);
        exercise("predict", context, [&] { model.predict(context, data); });
    } catch (const std::exception &error) {
        armed = false;
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
