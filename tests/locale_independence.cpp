/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

/* Operations must not depend on the host locale. The test installs a
 * decimal-comma locale as the process locale and as the calling thread's
 * locale, runs every operation on fractional data, and compares the results
 * with those of the C locale. It skips, with status 77, when the host has no
 * decimal-comma locale. */

#include <clocale>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include <c50/c50.hpp>

#ifdef _WIN32
int main()
{
    return 77;
}
#else

#include <locale.h>

namespace {
constexpr int skip_status = 77;

void require(bool value, const char *message)
{
    if (!value) throw std::runtime_error(message);
}

const std::string names =
    "low, high.\n\n"
    "signal: continuous.\n"
    "weight: continuous.\n";
const std::string training =
    "0.25, 1.5, low\n0.5, 1.25, low\n0.75, 2.5, low\n1.25, 0.5, low\n"
    "1.5, 3.25, low\n1.75, 2.75, low\n2.25, 0.75, low\n2.5, 3.5, low\n"
    "3.25, 1.25, high\n3.5, 2.25, high\n3.75, 0.25, high\n4.25, 3.75, high\n"
    "4.5, 1.75, high\n4.75, 2.5, high\n5.25, 0.5, high\n5.5, 3.25, high\n";
const std::string cases = "0.5, 1.5, ?\n5.25, 2.5, ?\n2.75, 0.75, ?\n";

const char *const candidates[] = {"de_DE.UTF-8", "de_DE.utf8", "fr_FR.UTF-8",
                                  "fr_FR.utf8", "de_DE", "fr_FR",
                                  "de_DE.ISO-8859-1", "fr_FR.ISO-8859-1"};

struct snapshot {
    std::string text;
    std::vector<std::size_t> classes;
    std::vector<double> cuts;
};

/* Run every operation once and collect what must not depend on the locale. */
snapshot run(c50::model_kind kind, const std::string &loaded_text)
{
    c50::context context;
    context.split_workers(4);
    c50::options options;
    options.trials = 2;
    options.confidence_factor = 0.375;
    options.sample_fraction = 0.75;
    snapshot result;
    c50::model model = c50::model::train(context, kind, names, training, options);
    result.text = model.serialized_data();
    if (!loaded_text.empty())
        model = c50::model::load(context, kind, names, loaded_text);
    const c50::predictions predicted = model.predict(context, cases);
    for (std::size_t index = 0; index < predicted.size(); ++index)
        result.classes.push_back(predicted.class_index(index));
    const c50::model_inspection inspected = model.inspect(context);
    for (const c50::tree_inspection &tree : inspected.trees)
        for (const auto &node : tree.nodes)
            if (node.threshold) result.cuts.push_back(node.threshold->cut);

    /* Fractional values in dense input take no part in text parsing, but the
     * prediction path still formats nothing: it must agree as well. */
    const double dense[] = {0.5, 1.5, 5.25, 2.5};
    c50::predictions dense_result = model.predict(
        context, c50::dense_dataset(dense, 2, 2));
    for (std::size_t index = 0; index < dense_result.size(); ++index)
        result.classes.push_back(dense_result.class_index(index));
    return result;
}

bool equal(const snapshot &left, const snapshot &right)
{
    return left.text == right.text && left.classes == right.classes &&
           left.cuts == right.cuts;
}

/* The caller's own formatting must survive each operation. */
bool caller_locale_intact()
{
    char buffer[32];
    std::snprintf(buffer, sizeof buffer, "%.1f", 1.5);
    return std::string(buffer) == "1,5";
}

void check_scenario(const char *label, const snapshot &expected,
                    const snapshot &expected_rules)
{
    require(caller_locale_intact(), label);
    const snapshot tree = run(c50::model_kind::tree, expected.text);
    require(equal(tree, expected), "tree results depend on the locale");
    for (std::size_t at = tree.text.find("cut=\""); at != std::string::npos;
         at = tree.text.find("cut=\"", at + 1)) {
        const std::size_t end = tree.text.find('"', at + 5);
        require(tree.text.substr(at + 5, end - at - 5).find(',') ==
                std::string::npos, "model text uses a decimal comma");
    }
    const snapshot rules = run(c50::model_kind::rules, expected_rules.text);
    require(equal(rules, expected_rules), "rule results depend on the locale");
    require(caller_locale_intact(), label);

    /* A failed operation restores the caller's locale as well. */
    c50::context context;
    try {
        static_cast<void>(c50::model::train(
            context, c50::model_kind::tree, names, "malformed", {}));
        require(false, "malformed data was accepted");
    } catch (const c50::exception &) {
    }
    require(caller_locale_intact(), label);
}

/* The caller's thread locale, and not only the process locale, must not
 * matter. The object converts to false when it cannot create the locale. */
class thread_locale {
public:
    /* Copy the process locale of that name. The process locale is set to the
     * name first and then restored to C. This avoids newlocale, which leaks
     * under LeakSanitizer when the locale comes from LOCPATH. */
    explicit thread_locale(const char *name)
    {
        if (std::setlocale(LC_ALL, name)) locale_ = duplocale(LC_GLOBAL_LOCALE);
        std::setlocale(LC_ALL, "C");
        if (locale_) previous_ = uselocale(locale_);
    }
    ~thread_locale()
    {
        if (locale_) {
            uselocale(previous_);
            freelocale(locale_);
        }
    }
    thread_locale(const thread_locale &) = delete;
    thread_locale &operator=(const thread_locale &) = delete;
    explicit operator bool() const { return locale_ != nullptr; }
private:
    locale_t locale_ = nullptr;
    locale_t previous_ = nullptr;
};

const char *find_decimal_comma_locale()
{
    for (const char *name : candidates) {
        if (!std::setlocale(LC_ALL, name)) continue;
        const char *point = std::localeconv()->decimal_point;
        if (point[0] == ',' && point[1] == '\0') {
            std::setlocale(LC_ALL, "C");
            return name;
        }
    }
    std::setlocale(LC_ALL, "C");
    return nullptr;
}
} // namespace

int main()
{
    const char *name = find_decimal_comma_locale();
    if (!name) {
        std::puts("skipped: no decimal-comma locale is installed");
        return skip_status;
    }
    try {
        std::setlocale(LC_ALL, "C");
        const snapshot expected = run(c50::model_kind::tree, "");
        const snapshot expected_rules = run(c50::model_kind::rules, "");
        require(expected.text.find("cut=\"") != std::string::npos,
                "the fixture has no continuous split");
        require(!expected.cuts.empty(), "the fixture has no threshold");

        /* The process locale. */
        require(std::setlocale(LC_ALL, name) != nullptr, "set the process locale");
        check_scenario("process locale", expected, expected_rules);

        /* The thread locale alone. */
        std::setlocale(LC_ALL, "C");
        {
            thread_locale guard(name);
            require(static_cast<bool>(guard), "set the thread locale");
            check_scenario("thread locale", expected, expected_rules);
        }

        /* Both. */
        {
            thread_locale guard(name);
            require(static_cast<bool>(guard), "set the thread locale");
            require(std::setlocale(LC_ALL, name) != nullptr,
                    "set the process locale");
            check_scenario("both locales", expected, expected_rules);
        }
        std::setlocale(LC_ALL, "C");
    } catch (const std::exception &error) {
        std::fprintf(stderr, "locale test failed: %s\n", error.what());
        return 1;
    }
    return 0;
}
#endif
