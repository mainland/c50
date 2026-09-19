function(run_checked description)
    execute_process(
        COMMAND ${ARGN}
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error
    )
    if(NOT result EQUAL 0)
        message(FATAL_ERROR
                "${description} failed (${result})\n${output}${error}")
    endif()
endfunction()

set(test_root "${C50_BINARY_DIR}/install-consumer")
set(prefix "${test_root}/prefix")
set(consumer_build "${test_root}/build")

file(REMOVE_RECURSE "${test_root}")

set(install_command
    "${CMAKE_COMMAND}" --install "${C50_BINARY_DIR}" --prefix "${prefix}")
if(C50_CONFIG)
    list(APPEND install_command --config "${C50_CONFIG}")
endif()
run_checked("C50 installation" ${install_command})

set(configure_command
    "${CMAKE_COMMAND}"
    -S "${C50_SOURCE_DIR}/tests/install"
    -B "${consumer_build}"
    -G "${C50_GENERATOR}"
    "-DCMAKE_PREFIX_PATH=${prefix}")
if(C50_GENERATOR_PLATFORM)
    list(APPEND configure_command -A "${C50_GENERATOR_PLATFORM}")
endif()
if(C50_GENERATOR_TOOLSET)
    list(APPEND configure_command -T "${C50_GENERATOR_TOOLSET}")
endif()
run_checked("installed-package configuration" ${configure_command})

set(build_command "${CMAKE_COMMAND}" --build "${consumer_build}")
if(C50_CONFIG)
    list(APPEND build_command --config "${C50_CONFIG}")
endif()
run_checked("installed-package build" ${build_command})

set(test_command
    "${CMAKE_CTEST_COMMAND}" --test-dir "${consumer_build}"
    --output-on-failure)
if(C50_CONFIG)
    list(APPEND test_command -C "${C50_CONFIG}")
endif()
run_checked("installed-package tests" ${test_command})
