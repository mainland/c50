/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

// A name that an attribute list repeats must mean what a single occurrence
// means. Release 2.07 subtracted the SKIP bit from the status byte once for
// each occurrence, so a repeated name corrupted the attribute's type bits.

#include <c50/c50.hpp>

#include "model_text.hpp"

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct kind_case {
    std::string label;
    std::string declaration; // The declaration of attribute v.
    std::string data;        // Cases in which v determines the class.
};

std::string repeated(const std::string &cases, int times)
{
    std::string result;
    for (int copy = 0; copy < times; ++copy) result += cases;
    return result;
}

std::vector<kind_case> kinds()
{
    std::vector<kind_case> result{
        {"continuous", "v: continuous.",
         "1, 5, no\n2, 4, no\n3, 7, no\n4, 1, no\n"
         "5, 6, yes\n6, 2, yes\n7, 3, yes\n8, 9, yes\n"},
        {"date", "v: date.",
         "2024/01/01, 5, no\n2024/01/02, 4, no\n2024/01/03, 7, no\n"
         "2024/01/04, 1, no\n2024/02/05, 6, yes\n2024/02/06, 2, yes\n"
         "2024/02/07, 3, yes\n2024/02/08, 9, yes\n"},
        {"time", "v: time.",
         "01:00:00, 5, no\n02:00:00, 4, no\n03:00:00, 7, no\n"
         "04:00:00, 1, no\n15:00:00, 6, yes\n16:00:00, 2, yes\n"
         "17:00:00, 3, yes\n18:00:00, 9, yes\n"},
        {"timestamp", "v: timestamp.",
         "2024/01/01 01:00:00, 5, no\n2024/01/02 02:00:00, 4, no\n"
         "2024/01/03 03:00:00, 7, no\n2024/01/04 04:00:00, 1, no\n"
         "2024/02/05 15:00:00, 6, yes\n2024/02/06 16:00:00, 2, yes\n"
         "2024/02/07 17:00:00, 3, yes\n2024/02/08 18:00:00, 9, yes\n"},
        {"discrete", "v: red, green, blue, black.",
         "red, 5, no\ngreen, 4, no\nred, 7, no\ngreen, 1, no\n"
         "blue, 6, yes\nblack, 2, yes\nblue, 3, yes\nblack, 9, yes\n"},
        {"ordered", "v: [ordered] lo, mid, hi, top.",
         "lo, 5, no\nmid, 4, no\nlo, 7, no\nmid, 1, no\n"
         "hi, 6, yes\ntop, 2, yes\nhi, 3, yes\ntop, 9, yes\n"},
        {"discrete count", "v: discrete 4.",
         "a, 5, no\nb, 4, no\na, 7, no\nb, 1, no\n"
         "c, 6, yes\nd, 2, yes\nc, 3, yes\nd, 9, yes\n"}};
    for (kind_case &kind : result) kind.data = repeated(kind.data, 4);
    return result;
}

std::string schema(const kind_case &kind, const std::string &list)
{
    return "no, yes.\n" + kind.declaration + "\nw: continuous.\n" + list;
}

void same_prediction(const c50::predictions &left,
                     const c50::predictions &right, const std::string &label)
{
    if (left.size() != right.size())
        throw std::runtime_error(label + ": prediction counts differ");
    for (std::size_t row = 0; row < left.size(); ++row)
        if (left.class_index(row) != right.class_index(row) ||
            left.confidence(row) != right.confidence(row))
            throw std::runtime_error(label + ": predictions differ");
}

// Train with a list that repeats a name and with the list that names it once.
// The models must match, and the repeated schema must load that model.
void same_model(c50::context &context, c50::model_kind kind,
                const std::string &once, const std::string &twice,
                const std::string &data, const std::string &used,
                const std::string &label)
{
    const c50::model single = c50::model::train(context, kind, once, data);
    const c50::model repeated = c50::model::train(context, kind, twice, data);
    const std::string text = without_stamp(repeated.serialized_data());
    if (text != without_stamp(single.serialized_data()))
        throw std::runtime_error(label + ": repeated name changed the model");
    if (text.find(used) == std::string::npos)
        throw std::runtime_error(label + ": model does not use the attribute");

    const c50::model loaded = c50::model::load(
        context, kind, twice, repeated.serialized_data());
    same_prediction(loaded.predict(context, data),
                    single.predict(context, data), label + " loaded");
    same_prediction(repeated.predict(context, data),
                    single.predict(context, data), label);
    if (loaded.inspect(context).feature_names !=
        single.inspect(context).feature_names)
        throw std::runtime_error(label + ": inspection differs");
}

void rejects_unknown(c50::context &context, const std::string &names,
                     const std::string &data, const std::string &label)
{
    try {
        c50::model::train(context, c50::model_kind::tree, names, data);
    } catch (const c50::exception &error) {
        if (error.code() == c50::error_code::parse_error) return;
        throw;
    }
    throw std::runtime_error(label + ": unknown name accepted");
}

} // namespace

int main()
{
    try {
        c50::context context;
        for (const kind_case &kind : kinds())
            for (const auto model : {c50::model_kind::tree,
                                     c50::model_kind::rules}) {
                same_model(context, model,
                           schema(kind, "attributes included: v.\n"),
                           schema(kind, "attributes included: v, v.\n"),
                           kind.data, "att=\"v\"", kind.label + " included");
                same_model(context, model,
                           schema(kind, "attributes included: v, w, w, v.\n"),
                           schema(kind, "attributes included: w, v.\n"),
                           kind.data, "att=\"v\"", kind.label + " both");
                same_model(context, model,
                           schema(kind, "attributes excluded: w.\n"),
                           schema(kind, "attributes excluded: w, w.\n"),
                           kind.data, "att=\"v\"", kind.label + " excluded");
            }

        // The class attribute may be an attribute of the schema, and a list
        // may name it more than once.
        const std::string outcome =
            "outcome.\nv: continuous.\noutcome: no, yes.\n";
        const std::string outcome_data =
            "1, no\n2, no\n3, no\n4, no\n5, yes\n6, yes\n7, yes\n8, yes\n";
        same_model(context, c50::model_kind::tree,
                   outcome + "attributes included: v, outcome.\n",
                   outcome + "attributes included: v, outcome, outcome.\n",
                   outcome_data, "att=\"v\"", "class attribute");
        same_model(context, c50::model_kind::rules,
                   outcome + "attributes included: v, outcome.\n",
                   outcome + "attributes included: outcome, v, outcome.\n",
                   outcome_data, "att=\"v\"", "class attribute rules");

        // An unknown name is an error however often it appears.
        const kind_case first = kinds().front();
        rejects_unknown(context,
                        schema(first, "attributes included: v, zz.\n"),
                        first.data, "included once");
        rejects_unknown(context,
                        schema(first, "attributes included: v, zz, zz.\n"),
                        first.data, "included twice");
        rejects_unknown(context,
                        schema(first, "attributes excluded: zz, zz.\n"),
                        first.data, "excluded twice");
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
