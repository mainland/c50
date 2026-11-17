# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Check the argument handling of the training fuzz driver. Run with
# `cmake -DDRIVER=<executable> -DWORKDIR=<directory> -P driver_arguments.cmake`.
# An invalid number must end the driver with status 2 before it creates the
# input file. A zero count must succeed without creating it either.
if(NOT DRIVER OR NOT WORKDIR)
    message(FATAL_ERROR "DRIVER and WORKDIR are required")
endif()

set(input "${WORKDIR}/driver-arguments-input")
file(MAKE_DIRECTORY "${WORKDIR}")

# Run the driver with the given SEED COUNT INPUT [FIRST] and check its status
# and that the input file is absent. Pass each argument quoted, because an
# unquoted list expansion drops empty elements.
function(check_driver expected seed count path)
    file(REMOVE "${input}")
    if(ARGC GREATER 4)
        set(arguments "${seed}" "${count}" "${path}" "${ARGV4}")
        execute_process(
            COMMAND "${DRIVER}" "${seed}" "${count}" "${path}" "${ARGV4}"
            RESULT_VARIABLE status
            OUTPUT_QUIET
            ERROR_QUIET
        )
    else()
        set(arguments "${seed}" "${count}" "${path}")
        execute_process(
            COMMAND "${DRIVER}" "${seed}" "${count}" "${path}"
            RESULT_VARIABLE status
            OUTPUT_QUIET
            ERROR_QUIET
        )
    endif()
    if(NOT status STREQUAL "${expected}")
        message(SEND_ERROR
                "driver arguments [${arguments}]: status ${status}, expected "
                "${expected}")
    endif()
    if(EXISTS "${input}")
        message(SEND_ERROR
                "driver arguments [${arguments}] created the input file")
    endif()
endfunction()

foreach(bad "-1" "18446744073709551616" "+1" " 1" "" "1 " "0x1" "1a")
    check_driver(2 "${bad}" 1 "${input}")
    check_driver(2 1 "${bad}" "${input}")
    check_driver(2 1 1 "${input}" "${bad}")
endforeach()

# The largest 64-bit value is valid, and a zero count trains nothing.
check_driver(0 18446744073709551615 0 "${input}")
check_driver(0 1 0 "${input}" 18446744073709551615)
check_driver(0 0 0 "${input}")
