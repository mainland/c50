/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Compare serialized classifiers without their training date.

   Each training run writes its local date into the id line, so two models
   trained from the same inputs differ on that line when the runs straddle
   midnight.  Compare without_stamp(model.serialized_data()) for models that
   were trained separately.  A model compared with its own load or round trip
   keeps its id line and may be compared as it is. */
#pragma once
#include <cctype>
#include <stdexcept>
#include <string>

/* Replace the date in the id line by "<date>" and keep every other
   character.  Throw if the text has no dated id line, because the date would
   then stay in the comparison. */
inline std::string without_stamp(const std::string &text)
{
    const std::string prefix = "id=\"See5/C5.0 ";
    const std::string::size_type line_end = text.find('\n');
    const std::string::size_type quote =
        text.rfind('"', line_end == std::string::npos ? text.size() : line_end);
    if (text.compare(0, prefix.size(), prefix) != 0 ||
        quote == std::string::npos || quote < 11)
        throw std::runtime_error("serialized model lacks an id line");
    const std::string::size_type date = quote - 10;
    bool dated = text[date - 1] == ' ';
    for (std::string::size_type at = 0; dated && at < 10; ++at) {
        const char c = text[date + at];
        dated = at == 4 || at == 7 ? c == '-'
                                   : std::isdigit(static_cast<unsigned char>(c));
    }
    if (!dated)
        throw std::runtime_error("serialized model id line lacks a date");
    std::string result = text;
    result.replace(date, 10, "<date>");
    return result;
}
