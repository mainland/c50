/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

// Compare model::attribute_usage with the "Attribute usage" reports that the
// command-line program printed for the regression fixtures.

#include <c50/c50.hpp>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool value, const std::string &message)
{
    if (!value) throw std::runtime_error(message);
}

std::string read_file(const std::string &path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot read fixture: " + path);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

// Parse the percentage lines that follow "Attribute usage:".
std::map<std::string, int> reported_usage(const std::string &output)
{
    std::map<std::string, int> usage;
    const auto start = output.find("\tAttribute usage:\n\n");
    if (start == std::string::npos) return usage;
    std::istringstream lines(output.substr(start));
    std::string line;
    std::getline(lines, line);
    std::getline(lines, line);
    while (std::getline(lines, line) && !line.empty()) {
        const auto percent = line.find('%');
        require(percent != std::string::npos, "malformed usage line: " + line);
        usage[line.substr(percent + 3)] = std::stoi(line.substr(0, percent));
    }
    return usage;
}

// Format counts as PrintUsageInfo does, omitting attributes below one percent.
std::map<std::string, int> computed_usage(const std::vector<std::string> &names,
                                          const std::vector<std::size_t> &counts,
                                          std::size_t cases)
{
    require(names.size() == counts.size(), "one count per attribute");
    const float tests = static_cast<float>(cases ? cases : 1);
    std::map<std::string, int> usage;
    for (std::size_t index = 0; index < counts.size(); ++index) {
        const auto count = static_cast<int>(counts[index]);
        if (count && count >= 0.01 * tests)
            usage[names[index]] = static_cast<int>((100 * count) / tests + 0.5);
    }
    return usage;
}

std::size_t line_count(const std::string &data)
{
    std::size_t count = 0;
    for (char character : data) count += character == '\n';
    return count;
}

struct usage_case {
    const char *fixture;
    const char *output;
    c50::model_kind kind;
    c50::options settings;
    bool costs;
};

usage_case make_case(const char *fixture, const char *output,
                     c50::model_kind kind = c50::model_kind::tree,
                     void (*configure)(c50::options &) = nullptr,
                     bool costs = false)
{
    usage_case result{fixture, output, kind, {}, costs};
    if (configure) configure(result.settings);
    return result;
}

void compare(const std::string &tests, const usage_case &test)
{
    const std::string fixture = tests + "/fixtures/" + test.fixture + "/" + test.fixture;
    const auto names = read_file(fixture + ".names");
    const auto data = read_file(fixture + ".data");
    const auto costs = test.costs ? read_file(fixture + ".costs") : std::string();
    const auto expected = reported_usage(read_file(
        tests + "/expected/" + test.fixture + "/" + test.output + ".output"));
    require(!expected.empty(), std::string("no usage report for ") + test.fixture);

    c50::context context;
    const auto model = c50::model::train(context, test.kind, names, data,
                                         test.settings, costs);
    const auto counts = model.attribute_usage(context, data);
    const auto actual = computed_usage(model.inspect(context).feature_names,
                                       counts, line_count(data));
    require(actual == expected, std::string("attribute usage differs for ") +
                                    test.fixture + "/" + test.output);
}

void dense_matches_text()
{
    const std::string names = "no, yes.\nsignal: continuous.\ncolor: red, blue.\n";
    const std::string data =
        "0, red, no\n1, red, no\n2, blue, no\n3, ?, yes\n4, blue, yes\n5, red, yes\n";
    const double missing = std::numeric_limits<double>::quiet_NaN();
    const std::vector<double> values = {0, 0, 1, 0, 2, 1, 3, missing, 4, 1, 5, 0};
    c50::context context;
    c50::options settings;
    settings.minimum_cases = 1;
    const auto model = c50::model::train(context, c50::model_kind::tree, names,
                                         data, settings);
    const auto text = model.attribute_usage(context, data);
    const auto dense = model.attribute_usage(
        context, c50::dense_dataset(values.data(), 6, 2));
    require(text == dense, "dense and text usage agree");
    require(text.size() == 2 && text[0] == 6, "the root split tests every case");
    require(model.attribute_usage(context, std::string()) ==
                std::vector<std::size_t>(2, 0),
            "no cases use no attributes");
}
} // namespace

int main(int argc, char **argv)
{
    try {
        require(argc == 2, "expected test directory");
        const std::vector<usage_case> cases = {
            make_case("basic", "tree"),
            make_case("basic", "rules", c50::model_kind::rules),
            make_case("basic", "subsets", c50::model_kind::tree,
                      [](c50::options &o) { o.subset_splits = true; }),
            make_case("basic", "winnow", c50::model_kind::tree,
                      [](c50::options &o) { o.winnow = true; }),
            make_case("basic", "soft-thresholds", c50::model_kind::tree,
                      [](c50::options &o) { o.probabilistic_thresholds = true; }),
            make_case("soft-threshold-ties", "soft-thresholds", c50::model_kind::tree,
                      [](c50::options &o) { o.probabilistic_thresholds = true; }),
            make_case("boost", "boost", c50::model_kind::tree,
                      [](c50::options &o) { o.trials = 5; }),
            make_case("implicit", "implicit"),
            make_case("multiclass", "tree"),
            make_case("multiclass", "rules", c50::model_kind::rules),
            make_case("winnow-case-weights", "winnow", c50::model_kind::tree,
                      [](c50::options &o) { o.winnow = true; }),
            make_case("winnow-case-weight-costs", "winnow", c50::model_kind::tree,
                      [](c50::options &o) { o.winnow = true; }, true),
        };
        for (const auto &test : cases) compare(argv[1], test);
        dense_matches_text();
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
