# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Check that CMake registers `writable-global-inventory` for a native core and
# omits it for a core compiled with link-time optimization, whose objects hold
# no sections for `nm` to read. Run with:
#   cmake -DC50_SOURCE_DIR=... -DC50_WORK_DIR=... -DC50_GENERATOR=...
#         -DC50_CXX_COMPILER=... -DC50_CTEST=... -P test_inventory_lto.cmake

include("${CMAKE_CURRENT_LIST_DIR}/clean_environment.cmake")

function(list_tests name)
    set(directory "${C50_WORK_DIR}/${name}")
    file(REMOVE_RECURSE "${directory}")
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env ${C50_CLEAN_ENVIRONMENT}
                "${CMAKE_COMMAND}"
                -S "${C50_SOURCE_DIR}" -B "${directory}"
                -G "${C50_GENERATOR}"
                "-DCMAKE_CXX_COMPILER=${C50_CXX_COMPILER}"
                -DCMAKE_BUILD_TYPE=Release
                ${ARGN}
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error
    )
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "configuring ${name} failed\n${output}${error}")
    endif()
    execute_process(
        COMMAND "${C50_CTEST}" --test-dir "${directory}" -N
        RESULT_VARIABLE result
        OUTPUT_VARIABLE listing
        ERROR_VARIABLE error
    )
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "listing the tests of ${name} failed\n${error}")
    endif()
    set(c50_listing "${listing}" PARENT_SCOPE)
endfunction()

function(expect_inventory name expected)
    if(c50_listing MATCHES "writable-global-inventory\r?\n")
        set(found ON)
    else()
        set(found OFF)
    endif()
    if(NOT found STREQUAL expected)
        message(FATAL_ERROR
                "${name}: writable-global-inventory registered is ${found}, "
                "expected ${expected}\n${c50_listing}")
    endif()
endfunction()

list_tests(native)
expect_inventory("native core" ON)

# GCC writes slim LTO objects by default, and Clang writes bitcode.
list_tests(flags "-DCMAKE_CXX_FLAGS=-flto")
expect_inventory("core built with -flto" OFF)

list_tests(slim "-DCMAKE_CXX_FLAGS=-flto -fno-fat-lto-objects")
expect_inventory("core built with slim LTO objects" OFF)

list_tests(variable -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON)
expect_inventory("core built with CMAKE_INTERPROCEDURAL_OPTIMIZATION" OFF)

list_tests(private -DC50_PRIVATE_CORE=ON)
expect_inventory("private core" OFF)
