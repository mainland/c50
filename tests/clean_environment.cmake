# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Arguments for `cmake -E env` that remove the caller's build settings from a
# nested configure, so that a test sees only the settings that it passes.
# Include this file from a script run with `cmake -P`.
set(C50_CLEAN_ENVIRONMENT "")
foreach(c50_variable
        CMAKE_BUILD_TYPE CMAKE_CXX_FLAGS CMAKE_C_FLAGS CMAKE_EXE_LINKER_FLAGS
        CMAKE_SHARED_LINKER_FLAGS CMAKE_MODULE_LINKER_FLAGS
        CMAKE_CXX_COMPILER_LAUNCHER CMAKE_C_COMPILER_LAUNCHER
        CMAKE_TOOLCHAIN_FILE CMAKE_GENERATOR CMAKE_GENERATOR_PLATFORM
        CMAKE_GENERATOR_TOOLSET CMAKE_PREFIX_PATH CMAKE_BUILD_PARALLEL_LEVEL
        CXX CC CXXFLAGS CFLAGS CPPFLAGS LDFLAGS)
    list(APPEND C50_CLEAN_ENVIRONMENT "--unset=${c50_variable}")
endforeach()
unset(c50_variable)
