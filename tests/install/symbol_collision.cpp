/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

/* Define global functions that the C5.0 learning core also uses internally.
   A consumer owns the global namespace, so linking the installed archive must
   not conflict with these definitions, and they must not replace the core's
   own. */

#include <string>

#include <c50/c50.hpp>

double ExecTime()
{
    return 42.0;
}

int DateToDay(char *)
{
    return 17;
}

int main()
{
    const std::string names =
        "low, high.\n\n"
        "signal: continuous.\n";
    const std::string training =
        "0, low\n1, low\n2, low\n3, high\n4, high\n5, high\n";
    const std::string cases = "5, ?\n";
    c50::context context;
    c50::model model = c50::model::train(
        context, c50::model_kind::tree, names, training);
    c50::predictions predictions = model.predict(context, cases);

    return ExecTime() == 42.0 && DateToDay(nullptr) == 17 &&
           predictions.size() == 1 && predictions.class_index(0) == 1 ?
        0 : 1;
}
