/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */
#include <c50/c50.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
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

static void exercise_predictor(const c50::model &model, const std::string &cases)
{
    c50::context reference;
    const auto expected = model.predict(reference, cases);
    auto predictor = model.prepare_predictor();
    predictor.predict(cases);
    calls = fail_at = 0;
    armed = true;
    predictor.predict(cases);
    armed = false;
    const auto total = calls;
    for (std::size_t index = 1; index <= total; ++index) {
        predictor = model.prepare_predictor();
        predictor.predict(cases);
        calls = 0;
        fail_at = index;
        bool rejected = false;
        armed = true;
        try { predictor.predict(cases); }
        catch (const std::bad_alloc &) { rejected = true; }
        catch (...) { armed = false; throw; }
        armed = false;
        if (!rejected) throw std::runtime_error("predictor allocation failure was ignored");
        const auto repeated = predictor.predict(cases);
        if (expected.size() != repeated.size())
            throw std::runtime_error("predictor recovery changed row count");
        for (std::size_t row = 0; row < expected.size(); ++row)
            for (std::size_t label = 0; label < expected.class_count(); ++label)
                if (expected.score(row, label) != repeated.score(row, label))
                    throw std::runtime_error("predictor recovery changed scores");
    }
}

static std::string read_file(const std::string &path)
{
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("cannot open fixture: " + path);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

int main(int argc, char **argv)
{
    try {
        if (argc != 2) throw std::runtime_error("expected boosting fixture directory");
        const auto boost_names = read_file(std::string(argv[1]) + "/boost.names");
        const auto boost_data = read_file(std::string(argv[1]) + "/boost.data");
        c50::context context;
        exercise("load", context, [&] { c50::model::load(context, c50::model_kind::tree, names, tree); });
        exercise("train", context, [&] { c50::model::train(context, c50::model_kind::tree, names, data); });
        exercise("rules", context, [&] {
            c50::model::train(context, c50::model_kind::rules, names, data);
        });
        const std::string category_names = "no, yes.\ncolor: red, blue, green, yellow.\n";
        const std::string category_data =
            "red, no\nblue, yes\ngreen, no\nyellow, yes\n"
            "red, no\nblue, yes\ngreen, no\nyellow, yes\n";
        exercise("branch compression", context, [&] {
            c50::model::train(context, c50::model_kind::tree, category_names, category_data);
        });
        c50::options subsets;
        subsets.subset_splits = true;
        exercise("subset rules", context, [&] {
            c50::model::train(context, c50::model_kind::rules,
                              category_names, category_data, subsets);
        });
        auto rules = c50::model::train(context, c50::model_kind::rules,
                                       category_names, category_data, subsets);
        const auto serialized_rules = rules.serialized_data();
        exercise("load rules", context, [&] {
            c50::model::load(context, c50::model_kind::rules,
                             category_names, serialized_rules);
        });
        exercise("predict rules", context, [&] { rules.predict(context, category_data); });
        exercise("inspect rules", context, [&] { rules.inspect(context); });
        c50::options boosted;
        boosted.trials = 3;
        for (auto kind : {c50::model_kind::tree, c50::model_kind::rules}) {
            auto ensemble = c50::model::train(context, kind, boost_names, boost_data, boosted);
            const auto serialized = ensemble.serialized_data();
            if (serialized.find("entries=\"3\"") == std::string::npos)
                throw std::runtime_error("boosting fixture did not build three classifiers");
            exercise(kind == c50::model_kind::tree ? "boosted tree" : "boosted rules", context, [&] {
                c50::model::train(context, kind, boost_names, boost_data, boosted);
            });
            exercise("load ensemble", context, [&] {
                c50::model::load(context, kind, boost_names, serialized);
            });
            exercise("predict ensemble", context, [&] { ensemble.predict(context, boost_data); });
            exercise("prepare predictor", context, [&] { ensemble.prepare_predictor(); });
            exercise_predictor(ensemble, boost_data);
            exercise("inspect ensemble", context, [&] { ensemble.inspect(context); });
        }
        const std::string dense_names = "no, yes.\nx: continuous.\ncolor: red, blue.\n";
        const double missing = std::numeric_limits<double>::quiet_NaN();
        const double values[] = {0, 0, 1, 0, 2, 1, 3, 1, missing, 1, 0, missing};
        const std::size_t dense_classes[] = {0, 0, 1, 1, 1, 0};
        const c50::dense_dataset dense(values, 6, 2, dense_classes);
        exercise("dense train", context, [&] {
            c50::model::train(context, c50::model_kind::tree, dense_names, dense);
        });
        auto dense_model = c50::model::train(context, c50::model_kind::tree, dense_names, dense);
        exercise("dense predict", context, [&] { dense_model.predict(context, dense); });
        exercise("inspect tree", context, [&] { dense_model.inspect(context); });
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
