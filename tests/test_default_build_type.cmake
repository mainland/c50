# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Configure the project in fresh directories and check the effective build
# type and the optimization flags of the core, and that no C compiler is
# needed. Run with:
#   cmake -DC50_SOURCE_DIR=... -DC50_WORK_DIR=... -DC50_GENERATOR=...
#         -DC50_CXX_COMPILER=... -P test_default_build_type.cmake

include("${CMAKE_CURRENT_LIST_DIR}/clean_environment.cmake")

function(configure_case name)
    set(directory "${C50_WORK_DIR}/${name}")
    file(REMOVE_RECURSE "${directory}")
    # A case sets the CMAKE_BUILD_TYPE environment variable by defining
    # ENV_BUILD_TYPE.
    # The nested configure ignores the caller's own build environment.
    set(environment_command "${CMAKE_COMMAND}" -E env
                            ${C50_CLEAN_ENVIRONMENT})
    if(DEFINED ENV_BUILD_TYPE)
        list(APPEND environment_command "CMAKE_BUILD_TYPE=${ENV_BUILD_TYPE}")
    endif()
    execute_process(
        COMMAND ${environment_command} "${CMAKE_COMMAND}"
                -S "${C50_SOURCE_DIR}" -B "${directory}"
                -G "${C50_GENERATOR}"
                "-DCMAKE_CXX_COMPILER=${C50_CXX_COMPILER}"
                -DBUILD_TESTING=OFF
                -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
                ${ARGN}
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error
    )
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "configuring ${name} failed\n${output}${error}")
    endif()
    file(STRINGS "${directory}/CMakeCache.txt" type_line
         REGEX "^CMAKE_BUILD_TYPE:")
    string(REGEX REPLACE "^[^=]*=" "" type "${type_line}")
    file(READ "${directory}/compile_commands.json" commands)
    string(REGEX MATCH "[^\n]*src/c50_api\\.cpp[^\n]*" core_command
           "${commands}")
    if(NOT core_command)
        # The database puts the command before the file name.
        string(REGEX MATCH "\"command\": \"[^\n]*\n[^\n]*c50_api\\.cpp"
               core_command "${commands}")
    endif()
    set(c50_type "${type}" PARENT_SCOPE)
    set(c50_command "${core_command}" PARENT_SCOPE)
endfunction()

function(expect_equal what actual expected)
    if(NOT "${actual}" STREQUAL "${expected}")
        message(FATAL_ERROR
                "${what}: expected '${expected}', found '${actual}'")
    endif()
endfunction()

# With no build type, the build chooses Release and optimizes the core.
configure_case(unset)
expect_equal("default build type" "${c50_type}" "Release")
if(NOT c50_command MATCHES " -O3 ")
    message(FATAL_ERROR "the default build does not optimize:\n${c50_command}")
endif()

# An explicit type is kept.
configure_case(debug -DCMAKE_BUILD_TYPE=Debug)
expect_equal("explicit build type" "${c50_type}" "Debug")
if(c50_command MATCHES " -O3 ")
    message(FATAL_ERROR "the Debug build optimizes:\n${c50_command}")
endif()

# CMake 3.22 and newer read the build type from the environment, and the
# build keeps it. Older versions ignore the variable, so the build chooses
# Release as if it were unset.
set(ENV_BUILD_TYPE Debug)
configure_case(environment)
unset(ENV_BUILD_TYPE)
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.22)
    expect_equal("environment build type" "${c50_type}" "Debug")
    if(c50_command MATCHES " -O3 ")
        message(FATAL_ERROR
                "the environment Debug build optimizes:\n${c50_command}")
    endif()
else()
    expect_equal("ignored environment build type" "${c50_type}" "Release")
    if(NOT c50_command MATCHES " -O3 ")
        message(FATAL_ERROR
                "the build with an ignored environment type does not "
                "optimize:\n${c50_command}")
    endif()
endif()

# No version uses an empty environment variable, so the build chooses Release.
set(ENV_BUILD_TYPE "")
configure_case(environment-empty)
unset(ENV_BUILD_TYPE)
expect_equal("empty environment build type" "${c50_type}" "Release")
if(NOT c50_command MATCHES " -O3 ")
    message(FATAL_ERROR
            "the build with an empty environment type does not optimize:\n"
            "${c50_command}")
endif()

# An explicitly empty type is kept, so users can choose their own flags.
configure_case(empty -DCMAKE_BUILD_TYPE=)
expect_equal("explicit empty build type" "${c50_type}" "")
if(c50_command MATCHES " -O3 ")
    message(FATAL_ERROR "the empty-type build optimizes:\n${c50_command}")
endif()

# Nothing compiles C, so a missing C compiler must not stop configuration.
configure_case(no-c-compiler -DCMAKE_C_COMPILER=/nonexistent/cc)
