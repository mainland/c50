cmake_minimum_required(VERSION 3.20)

project(C50 VERSION 2.07 LANGUAGES C CXX)

option(C50_ENABLE_SANITIZERS
       "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

include(CTest)
find_package(Threads REQUIRED)

set(C50_CORE_SOURCES
    src/attwinnow.cpp
    src/c50_api.cpp
    src/c50_input.cpp
    src/c50_model.cpp
    src/c50_output.cpp
    src/classify.cpp
    src/confmat.cpp
    src/construct.cpp
    src/contin.cpp
    src/discr.cpp
    src/formrules.cpp
    src/formtree.cpp
    src/getdata.cpp
    src/getnames.cpp
    src/global.cpp
    src/implicitatt.cpp
    src/info.cpp
    src/mcost.cpp
    src/modelfiles.cpp
    src/p-thresh.cpp
    src/prune.cpp
    src/rules.cpp
    src/ruletree.cpp
    src/siftrules.cpp
    src/sort.cpp
    src/subset.cpp
    src/trees.cpp
    src/update.cpp
    src/utility.cpp
    src/xval.cpp
)

add_library(c50_core STATIC ${C50_CORE_SOURCES})
add_library(C50::core ALIAS c50_core)
target_include_directories(c50_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(c50_core PUBLIC Threads::Threads)
add_library(c50_cpp INTERFACE)
add_library(C50::cpp ALIAS c50_cpp)
target_link_libraries(c50_cpp INTERFACE C50::core)
target_compile_features(c50_core PUBLIC cxx_std_17)
set_target_properties(c50_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(c50_cli src/c50.cpp)
target_link_libraries(c50_cli PRIVATE C50::core)
set_target_properties(c50_cli PROPERTIES OUTPUT_NAME c5.0)

add_executable(c50_report src/report.c)
set_target_properties(c50_report PROPERTIES OUTPUT_NAME report)

set(C50_COMPILE_TARGETS c50_core c50_cli c50_report)
set(C50_EXECUTABLE_TARGETS c50_cli c50_report)

if(BUILD_TESTING)
    add_executable(
        c50_prediction_probe
        tests/prediction_probe.cpp
    )
    target_include_directories(
        c50_prediction_probe
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    set_target_properties(
        c50_prediction_probe
        PROPERTIES OUTPUT_NAME prediction-probe
    )
    target_link_libraries(c50_prediction_probe PRIVATE C50::core)
    list(APPEND C50_COMPILE_TARGETS c50_prediction_probe)
    list(APPEND C50_EXECUTABLE_TARGETS c50_prediction_probe)
endif()

if(BUILD_TESTING)
    add_executable(c50_input tests/c50_input.cpp)
    target_include_directories(c50_input PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(c50_input PRIVATE C50::core)
    list(APPEND C50_COMPILE_TARGETS c50_input)
    list(APPEND C50_EXECUTABLE_TARGETS c50_input)
    add_test(NAME c50_input COMMAND c50_input)
endif()

if(BUILD_TESTING)
    add_executable(c50_data_input tests/c50_data_input.cpp)
    target_include_directories(c50_data_input PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(c50_data_input PRIVATE C50::core)
    list(APPEND C50_COMPILE_TARGETS c50_data_input)
    list(APPEND C50_EXECUTABLE_TARGETS c50_data_input)
    add_test(NAME c50_data_input COMMAND c50_data_input)
endif()

if(BUILD_TESTING)
    add_executable(c50_output tests/c50_output.cpp)
    target_include_directories(c50_output PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(c50_output PRIVATE C50::core)
    list(APPEND C50_COMPILE_TARGETS c50_output)
    list(APPEND C50_EXECUTABLE_TARGETS c50_output)
    add_test(NAME c50_output COMMAND c50_output)
endif()

if(BUILD_TESTING)
    add_executable(c50_operation_failure tests/operation_failure.cpp)
    target_include_directories(c50_operation_failure PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(c50_operation_failure PRIVATE C50::cpp)
    list(APPEND C50_COMPILE_TARGETS c50_operation_failure)
    list(APPEND C50_EXECUTABLE_TARGETS c50_operation_failure)
    add_test(NAME operation-failure COMMAND c50_operation_failure)
endif()

if(BUILD_TESTING)
    add_executable(c50_cpp_api tests/cpp_api.cpp)
    target_include_directories(c50_cpp_api PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(c50_cpp_api PRIVATE C50::cpp)
    list(APPEND C50_COMPILE_TARGETS c50_cpp_api)
    list(APPEND C50_EXECUTABLE_TARGETS c50_cpp_api)
    add_test(NAME cpp-api COMMAND c50_cpp_api)
endif()

if(BUILD_TESTING)
    add_executable(c50_cpp_contracts tests/cpp_contracts.cpp)
    target_include_directories(c50_cpp_contracts PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(c50_cpp_contracts PRIVATE C50::cpp)
    list(APPEND C50_COMPILE_TARGETS c50_cpp_contracts)
    list(APPEND C50_EXECUTABLE_TARGETS c50_cpp_contracts)
    add_test(NAME cpp-contracts COMMAND c50_cpp_contracts)
endif()

if(BUILD_TESTING)
    add_executable(c50_cpp_validation tests/cpp_validation.cpp)
    target_include_directories(c50_cpp_validation PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(c50_cpp_validation PRIVATE C50::cpp)
    list(APPEND C50_COMPILE_TARGETS c50_cpp_validation)
    list(APPEND C50_EXECUTABLE_TARGETS c50_cpp_validation)
    add_test(NAME cpp-validation COMMAND c50_cpp_validation)
endif()

if(BUILD_TESTING)
    add_executable(c50_reference_compatibility tests/reference_compatibility.cpp)
    target_include_directories(c50_reference_compatibility PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(c50_reference_compatibility PRIVATE C50::cpp)
    list(APPEND C50_COMPILE_TARGETS c50_reference_compatibility)
    list(APPEND C50_EXECUTABLE_TARGETS c50_reference_compatibility)
    add_test(NAME reference-compatibility COMMAND c50_reference_compatibility)
endif()

if(BUILD_TESTING)
    add_executable(c50_api_prediction_probe tests/api_prediction_probe.cpp)
    target_link_libraries(c50_api_prediction_probe PRIVATE C50::cpp)
    list(APPEND C50_COMPILE_TARGETS c50_api_prediction_probe)
    list(APPEND C50_EXECUTABLE_TARGETS c50_api_prediction_probe)
    if(CMAKE_NM AND UNIX)
        add_test(NAME writable-global-inventory COMMAND
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_writable_globals.sh"
            "${CMAKE_NM}" "$<TARGET_FILE:c50_core>"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/expected/writable-globals.txt")
    endif()
endif()

if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
    foreach(target IN LISTS C50_COMPILE_TARGETS)
        target_compile_options(${target} PRIVATE -ffloat-store)
    endforeach()
endif()

if(NOT MSVC)
    target_link_libraries(c50_core PUBLIC m)
    target_link_libraries(c50_report PRIVATE m)
endif()

if(C50_ENABLE_SANITIZERS)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        foreach(target IN LISTS C50_COMPILE_TARGETS)
            target_compile_options(
                ${target}
                PRIVATE
                    -fsanitize=address,undefined
                    -fno-omit-frame-pointer
                    -fno-sanitize-recover=all
            )
        endforeach()
        foreach(target IN LISTS C50_EXECUTABLE_TARGETS)
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
            "C50_API_PREDICTION_BINARY=$<TARGET_FILE:c50_api_prediction_probe>"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_predictions.sh"
    )
endif()
