cmake_minimum_required(VERSION 3.20)

project(C50 VERSION 2.07 LANGUAGES C)

option(C50_ENABLE_SANITIZERS
       "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

include(CTest)

set(C50_CORE_SOURCES
    src/attwinnow.c
    src/classify.c
    src/confmat.c
    src/construct.c
    src/contin.c
    src/discr.c
    src/formrules.c
    src/formtree.c
    src/getdata.c
    src/getnames.c
    src/global.c
    src/implicitatt.c
    src/info.c
    src/mcost.c
    src/modelfiles.c
    src/p-thresh.c
    src/prune.c
    src/rules.c
    src/ruletree.c
    src/siftrules.c
    src/sort.c
    src/subset.c
    src/trees.c
    src/update.c
    src/utility.c
    src/xval.c
)

add_executable(c50_cli src/c50.c ${C50_CORE_SOURCES})
set_target_properties(c50_cli PROPERTIES OUTPUT_NAME c5.0)

add_executable(c50_report src/report.c)
set_target_properties(c50_report PROPERTIES OUTPUT_NAME report)

set(C50_TARGETS c50_cli c50_report)

if(BUILD_TESTING)
    add_executable(
        c50_prediction_probe
        tests/prediction_probe.c
        ${C50_CORE_SOURCES}
    )
    target_include_directories(
        c50_prediction_probe
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    set_target_properties(
        c50_prediction_probe
        PROPERTIES OUTPUT_NAME prediction-probe
    )
    list(APPEND C50_TARGETS c50_prediction_probe)
endif()

if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
    foreach(target IN LISTS C50_TARGETS)
        target_compile_options(${target} PRIVATE -ffloat-store)
    endforeach()
endif()

if(NOT MSVC)
    foreach(target IN LISTS C50_TARGETS)
        target_link_libraries(${target} PRIVATE m)
    endforeach()
endif()

if(C50_ENABLE_SANITIZERS)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        foreach(target IN LISTS C50_TARGETS)
            target_compile_options(
                ${target}
                PRIVATE
                    -fsanitize=address,undefined
                    -fno-omit-frame-pointer
                    -fno-sanitize-recover=all
            )
            target_link_options(
                ${target}
                PRIVATE
                    -fsanitize=address,undefined
                    -fno-sanitize-recover=all
            )
        endforeach()
    else()
        message(FATAL_ERROR
                "C50_ENABLE_SANITIZERS requires GCC, Clang, or AppleClang")
    endif()
endif()

if(BUILD_TESTING)
    add_test(
        NAME cli-regression
        COMMAND
            ${CMAKE_COMMAND} -E env
            "C50_BINARY=$<TARGET_FILE:c50_cli>"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cli.sh"
    )
    add_test(
        NAME report-regression
        COMMAND
            ${CMAKE_COMMAND} -E env
            "REPORT_BINARY=$<TARGET_FILE:c50_report>"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_report.sh"
    )
    add_test(
        NAME prediction-regression
        COMMAND
            ${CMAKE_COMMAND} -E env
            "C50_BINARY=$<TARGET_FILE:c50_cli>"
            "C50_PREDICTION_BINARY=$<TARGET_FILE:c50_prediction_probe>"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_predictions.sh"
    )
endif()
