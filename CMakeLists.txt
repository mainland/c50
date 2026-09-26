cmake_minimum_required(VERSION 3.20)

project(C50 VERSION 2.07 LANGUAGES C)

option(C50_ENABLE_SANITIZERS
       "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

set(C50_SOURCES
    src/attwinnow.c
    src/c50.c
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

add_executable(c50_cli ${C50_SOURCES})
set_target_properties(c50_cli PROPERTIES OUTPUT_NAME c5.0)

add_executable(c50_report src/report.c)
set_target_properties(c50_report PROPERTIES OUTPUT_NAME report)

if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
    target_compile_options(c50_cli PRIVATE -ffloat-store)
    target_compile_options(c50_report PRIVATE -ffloat-store)
endif()

if(NOT MSVC)
    target_link_libraries(c50_cli PRIVATE m)
    target_link_libraries(c50_report PRIVATE m)
endif()

if(C50_ENABLE_SANITIZERS)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        foreach(target IN ITEMS c50_cli c50_report)
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

include(CTest)

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
endif()
