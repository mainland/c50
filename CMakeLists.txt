cmake_minimum_required(VERSION 3.20)

project(C50 VERSION 2.07 LANGUAGES C CXX)

option(C50_ENABLE_SANITIZERS
       "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(C50_ENABLE_THREAD_SANITIZER "Enable ThreadSanitizer" OFF)

if(C50_ENABLE_SANITIZERS AND C50_ENABLE_THREAD_SANITIZER)
    message(FATAL_ERROR "AddressSanitizer and ThreadSanitizer cannot be combined")
endif()

include(CTest)

set(C50_CORE_SOURCES
    src/attwinnow.c
    src/c50_api.c
    src/c50_input.c
    src/c50_model.c
    src/c50_output.c
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

add_library(c50_core STATIC ${C50_CORE_SOURCES})
add_library(C50::core ALIAS c50_core)
set_target_properties(c50_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(
    c50_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

add_executable(c50_cli src/c50.c)
target_link_libraries(c50_cli PRIVATE C50::core)
set_target_properties(c50_cli PROPERTIES OUTPUT_NAME c5.0)

add_executable(c50_report src/report.c)
set_target_properties(c50_report PROPERTIES OUTPUT_NAME report)

set(C50_COMPILE_TARGETS c50_core c50_cli c50_report)
set(C50_EXECUTABLE_TARGETS c50_cli c50_report)

if(BUILD_TESTING)
    find_package(Threads REQUIRED)
    add_executable(
        c50_prediction_probe
        tests/prediction_probe.c
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

    add_executable(c50_api_prediction_probe tests/api_prediction_probe.c)
    target_link_libraries(c50_api_prediction_probe PRIVATE C50::core)
    set_target_properties(
        c50_api_prediction_probe
        PROPERTIES OUTPUT_NAME api-prediction-probe
    )
    list(APPEND C50_COMPILE_TARGETS c50_api_prediction_probe)
    list(APPEND C50_EXECUTABLE_TARGETS c50_api_prediction_probe)

    add_executable(c50_c_api_compile tests/c_api_compile.c)
    target_link_libraries(c50_c_api_compile PRIVATE C50::core)
    add_executable(c50_input_test tests/c50_input.c)
    target_include_directories(
        c50_input_test
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(c50_input_test PRIVATE C50::core)
    add_executable(c50_output_test tests/c50_output.c)
    target_include_directories(
        c50_output_test
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(c50_output_test PRIVATE C50::core)
    add_executable(c50_data_input_test tests/c50_data_input.c)
    target_include_directories(
        c50_data_input_test
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(c50_data_input_test PRIVATE C50::core)
    add_executable(c50_c_api_failure tests/c_api_failure.c)
    target_include_directories(
        c50_c_api_failure
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(c50_c_api_failure PRIVATE C50::core)
    add_executable(c50_c_api_model tests/c_api_model.c)
    target_link_libraries(c50_c_api_model PRIVATE C50::core)
    add_executable(c50_c_api_prediction tests/c_api_prediction.c)
    target_link_libraries(c50_c_api_prediction PRIVATE C50::core)
    add_executable(c50_c_api_training tests/c_api_training.c)
    target_link_libraries(c50_c_api_training PRIVATE C50::core)
    add_executable(
        c50_c_api_context_isolation
        tests/c_api_context_isolation.c
    )
    target_link_libraries(c50_c_api_context_isolation PRIVATE C50::core)
    add_executable(c50_c_api_concurrency tests/c_api_concurrency.c)
    target_link_libraries(
        c50_c_api_concurrency
        PRIVATE C50::core Threads::Threads
    )
    add_executable(
        c50_c_api_training_concurrency
        tests/c_api_training_concurrency.c
    )
    target_link_libraries(
        c50_c_api_training_concurrency
        PRIVATE C50::core Threads::Threads
    )
    add_executable(c50_cpp_api_compile tests/c_api_compile.cpp)
    target_compile_features(c50_cpp_api_compile PRIVATE cxx_std_11)
    target_link_libraries(c50_cpp_api_compile PRIVATE C50::core)
    list(
        APPEND C50_COMPILE_TARGETS
        c50_c_api_compile
        c50_input_test
        c50_output_test
        c50_data_input_test
        c50_c_api_failure
        c50_c_api_model
        c50_c_api_prediction
        c50_c_api_training
        c50_c_api_context_isolation
        c50_c_api_concurrency
        c50_c_api_training_concurrency
        c50_cpp_api_compile
    )
    list(
        APPEND C50_EXECUTABLE_TARGETS
        c50_c_api_compile
        c50_input_test
        c50_output_test
        c50_data_input_test
        c50_c_api_failure
        c50_c_api_model
        c50_c_api_prediction
        c50_c_api_training
        c50_c_api_context_isolation
        c50_c_api_concurrency
        c50_c_api_training_concurrency
        c50_cpp_api_compile
    )
endif()

if(C50_ENABLE_THREAD_SANITIZER)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        foreach(target IN LISTS C50_COMPILE_TARGETS)
            target_compile_options(
                ${target} PRIVATE -fsanitize=thread -fno-omit-frame-pointer
            )
        endforeach()
        foreach(target IN LISTS C50_EXECUTABLE_TARGETS)
            target_link_options(${target} PRIVATE -fsanitize=thread)
        endforeach()
    else()
        message(FATAL_ERROR
                "C50_ENABLE_THREAD_SANITIZER requires GCC, Clang, or AppleClang")
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
    add_test(NAME c-api-compile COMMAND c50_c_api_compile)
    add_test(NAME input-stream COMMAND c50_input_test)
    add_test(NAME output-stream COMMAND c50_output_test)
    add_test(NAME data-input-stream COMMAND c50_data_input_test)
    add_test(NAME c-api-failure COMMAND c50_c_api_failure)
    add_test(NAME c-api-model COMMAND c50_c_api_model)
    add_test(NAME c-api-prediction COMMAND c50_c_api_prediction)
    add_test(NAME c-api-training COMMAND c50_c_api_training)
    add_test(NAME c-api-context-isolation COMMAND c50_c_api_context_isolation)
    add_test(NAME c-api-concurrency COMMAND c50_c_api_concurrency)
    add_test(
        NAME c-api-training-concurrency
        COMMAND c50_c_api_training_concurrency
    )
    add_test(NAME cpp-api-compile COMMAND c50_cpp_api_compile)
    if(CMAKE_NM AND UNIX)
        add_test(
            NAME writable-global-inventory
            COMMAND
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_writable_globals.sh"
                "${CMAKE_NM}"
                "$<TARGET_FILE:c50_core>"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/expected/writable-globals.txt"
        )
    endif()
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
