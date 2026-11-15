/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <cerrno>
#include <new>
#include <string>

#include <c50/c50.hpp>

#include "c50_locale.h"

#ifdef _WIN32

#include <locale.h>
#include <stdlib.h>

namespace c50::core {

c50_c_locale::c50_c_locale() {}
c50_c_locale::~c50_c_locale() {}

c50_c_locale_scope::c50_c_locale_scope(const c50_c_locale &)
{
    const char *current = setlocale(LC_ALL, nullptr);
    if (!current)
        throw c50::exception(c50::error_code::internal_error,
                             "cannot read the thread locale");
    previous_locale_ = current;
    previous_configuration_ = _configthreadlocale(_ENABLE_PER_THREAD_LOCALE);
    if (previous_configuration_ == -1)
        throw c50::exception(c50::error_code::internal_error,
                             "cannot enable the per-thread locale");
    if (!setlocale(LC_ALL, "C")) {
        _configthreadlocale(previous_configuration_);
        throw c50::exception(c50::error_code::internal_error,
                             "cannot select the C locale");
    }
}

c50_c_locale_scope::~c50_c_locale_scope()
{
    setlocale(LC_ALL, previous_locale_.c_str());
    _configthreadlocale(previous_configuration_);
}

}  // namespace c50::core

#else

namespace c50::core {

c50_c_locale::c50_c_locale() : handle_(newlocale(LC_ALL_MASK, "C", nullptr))
{
    if (!handle_) {
        const int failure = errno;
        if (failure == ENOMEM) throw std::bad_alloc();
        throw c50::exception(c50::error_code::internal_error,
                             "cannot create the C locale (errno " +
                             std::to_string(failure) + ")");
    }
}

c50_c_locale::~c50_c_locale()
{
    freelocale(handle_);
}

c50_c_locale_scope::c50_c_locale_scope(const c50_c_locale &locale)
    : previous_(uselocale(locale.handle_))
{
    if (!previous_)
        throw c50::exception(c50::error_code::internal_error,
                             "cannot select the C locale (errno " +
                             std::to_string(errno) + ")");
}

c50_c_locale_scope::~c50_c_locale_scope()
{
    uselocale(previous_);
}

}  // namespace c50::core

#endif
