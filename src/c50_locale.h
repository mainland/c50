/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#ifndef C50_LOCALE_H
#define C50_LOCALE_H

#ifdef _WIN32
#include <string>
#else
#include <locale.h>
#endif

namespace c50::core {

/* The core formats and parses numbers with C library functions that follow the
 * calling thread's locale. A c50_c_locale_scope makes the calling thread use
 * the C locale, so that model text, data, names, and costs read and write `.`
 * as the decimal point whatever the host locale is.
 *
 * The scope covers every locale category, not only LC_NUMERIC. The parsers
 * also classify bytes with isspace and its relatives, whose result for bytes
 * above 127 depends on LC_CTYPE in an eight-bit locale, and the saved models
 * must not depend on that either. */

/* Owner of a C locale object for one context. Construction throws
 * std::bad_alloc when the platform reports that it is out of memory, and a
 * c50::exception with error_code::internal_error for any other failure. A
 * context shares it among the threads of one operation, which only read it. */
class c50_c_locale
{
public:
    c50_c_locale();
    ~c50_c_locale();
    c50_c_locale(const c50_c_locale &) = delete;
    c50_c_locale &operator=(const c50_c_locale &) = delete;

private:
    friend class c50_c_locale_scope;
#ifndef _WIN32
    locale_t handle_;
#endif
};

/* Select the C locale for the calling thread until destruction, then restore
 * the thread's earlier locale, including when an exception unwinds the scope.
 * The scope applies to one thread and must be destroyed on it. Every thread
 * that an operation starts needs its own scope.
 *
 * On POSIX systems it uses uselocale, which leaves the process locale alone.
 * On Windows it enables the per-thread locale with _configthreadlocale and
 * calls setlocale, then restores both. The Windows code has not run in this
 * repository's local checks. */
class c50_c_locale_scope
{
public:
    explicit c50_c_locale_scope(const c50_c_locale &locale);
    ~c50_c_locale_scope();
    c50_c_locale_scope(const c50_c_locale_scope &) = delete;
    c50_c_locale_scope &operator=(const c50_c_locale_scope &) = delete;

private:
#ifdef _WIN32
    int previous_configuration_;
    std::string previous_locale_;
#else
    locale_t previous_;
#endif
};

}  // namespace c50::core

#endif
