/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

// Load a `.tree` that Release 2.07 wrote for implicit definitions of the
// unknown and N/A literals, and check the classes against Release 2.07's
// evaluation of the same cases, which classified all of them correctly.

#include <c50/c50.hpp>

#include <cstddef>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string read(const std::string &path)
{
    std::ifstream file(path, std::ios::binary);
    if ( ! file ) throw std::runtime_error("cannot read fixture: " + path);
    return {std::istreambuf_iterator<char>(file),
            std::istreambuf_iterator<char>()};
}

} // namespace

int main(int argc, char **argv)
{
    if ( argc != 2 )
    {
        std::cerr << "usage: c50_implicit_legacy FIXTURE_DIRECTORY\n";
        return 2;
    }

    try
    {
        const std::string directory = std::string(argv[1]) + "/implicit-legacy/";
        const std::string names = read(directory + "legacy.names");
        const std::string tree = read(directory + "legacy.tree");
        std::istringstream labeled(read(directory + "legacy.test"));
        std::string cases;
        std::vector<std::string> expected;

        for ( std::string line; std::getline(labeled, line); )
        {
            const std::string::size_type comma = line.rfind(',');
            if ( comma == std::string::npos ) continue;
            std::string label = line.substr(comma + 1);
            label.erase(0, label.find_first_not_of(' '));
            if ( ! label.empty() && label.back() == '.' ) label.pop_back();
            expected.push_back(label);
            cases += line.substr(0, comma) + ", ?\n";
        }

        c50::context context;
        const c50::model model = c50::model::load(
            context, c50::model_kind::tree, names, tree);
        const c50::predictions predictions = model.predict(context, cases);

        if ( predictions.size() != expected.size() )
        {
            std::cerr << "expected " << expected.size() << " predictions, got "
                      << predictions.size() << '\n';
            return 1;
        }

        int failures = 0;
        for ( std::size_t row = 0; row < expected.size(); ++row )
        {
            const std::string actual =
                predictions.class_name(predictions.class_index(row));
            if ( actual != expected[row] )
            {
                std::cerr << "case " << row << ": expected " << expected[row]
                          << ", got " << actual << '\n';
                ++failures;
            }
        }
        return failures ? 1 : 0;
    }
    catch ( const std::exception &error )
    {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
