cmake_minimum_required(VERSION 3.20)

project(C50 VERSION 2.07 LANGUAGES C CXX)

option(C50_ENABLE_SANITIZERS
       "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(C50_ENABLE_THREAD_SANITIZER "Enable ThreadSanitizer" OFF)
option(C50_BUILD_PYTHON "Build the Python extension module" OFF)
option(C50_BUILD_DOCUMENTATION "Build the Doxygen API documentation" OFF)
option(C50_BUILD_FUZZERS "Build AFL++ parser harnesses" OFF)
option(C50_BUILD_BENCHMARKS "Build native correctness tests and benchmarks" OFF)

if(C50_ENABLE_SANITIZERS AND C50_ENABLE_THREAD_SANITIZER)
    message(FATAL_ERROR "AddressSanitizer and ThreadSanitizer cannot be combined")
endif()

include(CTest)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
find_package(Threads REQUIRED)

if(C50_BUILD_DOCUMENTATION)
    find_package(Doxygen 1.9 REQUIRED)
    set(DOXYGEN_PROJECT_NAME "C5.0 GPL")
    set(DOXYGEN_PROJECT_NUMBER "${PROJECT_VERSION}")
    set(
        DOXYGEN_OUTPUT_DIRECTORY
        "${CMAKE_CURRENT_BINARY_DIR}/docs/doxygen"
    )
    set(DOXYGEN_EXTRACT_ALL NO)
    set(DOXYGEN_EXTRACT_PRIVATE NO)
    set(DOXYGEN_EXTRACT_STATIC NO)
    set(DOXYGEN_FULL_PATH_NAMES NO)
    set(DOXYGEN_GENERATE_HTML YES)
    set(DOXYGEN_GENERATE_TREEVIEW YES)
    set(DOXYGEN_GENERATE_XML YES)
    set(DOXYGEN_JAVADOC_AUTOBRIEF YES)
    set(DOXYGEN_MARKDOWN_SUPPORT YES)
    set(DOXYGEN_QUIET YES)
    set(DOXYGEN_RECURSIVE NO)
    set(DOXYGEN_STRIP_FROM_PATH "${CMAKE_CURRENT_SOURCE_DIR}/include")
    set(DOXYGEN_TYPEDEF_HIDES_STRUCT YES)
    set(DOXYGEN_WARN_AS_ERROR YES)
    set(DOXYGEN_WARN_IF_DOC_ERROR YES)
    set(DOXYGEN_WARN_IF_UNDOCUMENTED YES)
    set(DOXYGEN_WARN_NO_PARAMDOC YES)
    doxygen_add_docs(
        c50-docs
        "${CMAKE_CURRENT_SOURCE_DIR}/include/c50/c50.hpp"
        COMMENT "Build the C5.0 C++ API documentation"
    )
endif()

set(C50_CORE_SOURCES
    src/attwinnow.cpp
    src/c50_api.cpp
    src/c50_inspection.cpp
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
set_target_properties(
    c50_core
    PROPERTIES
        EXPORT_NAME core
        POSITION_INDEPENDENT_CODE ON
)
target_include_directories(
    c50_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_compile_features(c50_core PUBLIC cxx_std_17)
target_link_libraries(c50_core PUBLIC Threads::Threads)

add_library(c50_cpp INTERFACE)
add_library(C50::cpp ALIAS c50_cpp)
set_target_properties(c50_cpp PROPERTIES EXPORT_NAME cpp)
target_compile_features(c50_cpp INTERFACE cxx_std_17)
target_link_libraries(c50_cpp INTERFACE C50::core)

if(C50_BUILD_PYTHON)
    find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(c50_python MODULE python/c50/_c50module.cpp)
    target_compile_features(c50_python PRIVATE cxx_std_17)
    target_link_libraries(c50_python PRIVATE C50::cpp)
    set_target_properties(
        c50_python
        PROPERTIES
            OUTPUT_NAME _c50
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/python/c50
    )
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/python/c50)
    configure_file(
        python/c50/__init__.py
        ${CMAKE_CURRENT_BINARY_DIR}/python/c50/__init__.py
        COPYONLY
    )
    configure_file(
        python/c50/_c50.pyi
        ${CMAKE_CURRENT_BINARY_DIR}/python/c50/_c50.pyi
        COPYONLY
    )
    configure_file(
        python/c50/inspection.py
        ${CMAKE_CURRENT_BINARY_DIR}/python/c50/inspection.py
        COPYONLY
    )
    configure_file(
        python/c50/_data.py
        ${CMAKE_CURRENT_BINARY_DIR}/python/c50/_data.py
        COPYONLY
    )
    configure_file(
        python/c50/sklearn.py
        ${CMAKE_CURRENT_BINARY_DIR}/python/c50/sklearn.py
        COPYONLY
    )
    configure_file(
        python/c50/py.typed
        ${CMAKE_CURRENT_BINARY_DIR}/python/c50/py.typed
        COPYONLY
    )
    install(
        TARGETS c50_python
        LIBRARY DESTINATION c50 COMPONENT python
        RUNTIME DESTINATION c50 COMPONENT python
    )
endif()

add_executable(c50_cli src/c50.cpp)
target_link_libraries(c50_cli PRIVATE C50::core)
set_target_properties(c50_cli PROPERTIES OUTPUT_NAME c5.0)

add_executable(c50_report src/report.cpp)
target_compile_features(c50_report PRIVATE cxx_std_17)
set_target_properties(c50_report PROPERTIES OUTPUT_NAME report)

set(C50_COMPILE_TARGETS c50_core c50_cli c50_report)
set(C50_EXECUTABLE_TARGETS c50_cli c50_report)
if(C50_BUILD_PYTHON)
    list(APPEND C50_COMPILE_TARGETS c50_python)
    list(APPEND C50_EXECUTABLE_TARGETS c50_python)
endif()

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

    add_executable(c50_api_prediction_probe tests/api_prediction_probe.cpp)
    target_link_libraries(c50_api_prediction_probe PRIVATE C50::core)
    set_target_properties(
        c50_api_prediction_probe
        PROPERTIES OUTPUT_NAME api-prediction-probe
    )
    list(APPEND C50_COMPILE_TARGETS c50_api_prediction_probe)
    list(APPEND C50_EXECUTABLE_TARGETS c50_api_prediction_probe)

    add_executable(c50_input_test tests/c50_input.cpp)
    target_include_directories(
        c50_input_test
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(c50_input_test PRIVATE C50::core)
    add_executable(c50_output_test tests/c50_output.cpp)
    target_include_directories(
        c50_output_test
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(c50_output_test PRIVATE C50::core)
    add_executable(c50_data_input_test tests/c50_data_input.cpp)
    target_include_directories(
        c50_data_input_test
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(c50_data_input_test PRIVATE C50::core)
    add_executable(c50_reference_compatibility tests/reference_compatibility.cpp)
    target_compile_features(c50_reference_compatibility PRIVATE cxx_std_17)
    target_link_libraries(c50_reference_compatibility PRIVATE C50::core)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_executable(c50_allocation_failure tests/allocation_failure.cpp)
        target_link_libraries(c50_allocation_failure PRIVATE C50::cpp)
        target_link_options(c50_allocation_failure PRIVATE
                            -Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=strdup)
        list(APPEND C50_COMPILE_TARGETS c50_allocation_failure)
        list(APPEND C50_EXECUTABLE_TARGETS c50_allocation_failure)
        add_test(NAME allocation-failure COMMAND c50_allocation_failure
                 "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/boost")
        # Compile the production scheduler with private failure probes only in
        # this executable. The installed core contains no test hooks.
        add_executable(c50_parallel_failure tests/parallel_failure.cpp src/formtree.cpp)
        target_include_directories(c50_parallel_failure PRIVATE
                                   ${CMAKE_CURRENT_SOURCE_DIR}/tests)
        target_compile_definitions(c50_parallel_failure PRIVATE C50_TEST_SPLIT_FAILURE)
        target_link_libraries(c50_parallel_failure PRIVATE C50::cpp)
        target_link_options(c50_parallel_failure PRIVATE -Wl,--wrap=calloc)
        list(APPEND C50_COMPILE_TARGETS c50_parallel_failure)
        list(APPEND C50_EXECUTABLE_TARGETS c50_parallel_failure)
        add_test(NAME parallel-failure COMMAND c50_parallel_failure)
        set_tests_properties(parallel-failure PROPERTIES TIMEOUT 120)
    endif()
    add_executable(c50_cost_validation tests/cost_validation.cpp)
    target_link_libraries(c50_cost_validation PRIVATE C50::cpp)
    list(APPEND C50_COMPILE_TARGETS c50_cost_validation)
    list(APPEND C50_EXECUTABLE_TARGETS c50_cost_validation)
    add_executable(c50_parser_validation tests/parser_validation.cpp)
    target_link_libraries(c50_parser_validation PRIVATE C50::cpp)
    list(APPEND C50_COMPILE_TARGETS c50_parser_validation)
    list(APPEND C50_EXECUTABLE_TARGETS c50_parser_validation)
    add_executable(c50_model_inspection tests/model_inspection.cpp)
    target_link_libraries(c50_model_inspection PRIVATE C50::cpp)
    list(APPEND C50_COMPILE_TARGETS c50_model_inspection)
    list(APPEND C50_EXECUTABLE_TARGETS c50_model_inspection)
    add_test(NAME model-inspection COMMAND c50_model_inspection)
    add_executable(c50_cpp_validation tests/cpp_validation.cpp)
    target_link_libraries(c50_cpp_validation PRIVATE C50::cpp)
    add_executable(c50_operation_failure tests/operation_failure.cpp)
    target_include_directories(c50_operation_failure PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(c50_operation_failure PRIVATE C50::cpp)
    list(APPEND C50_COMPILE_TARGETS c50_cpp_validation c50_operation_failure)
    list(APPEND C50_EXECUTABLE_TARGETS c50_cpp_validation c50_operation_failure)
    add_executable(c50_cpp_contracts tests/cpp_contracts.cpp)
    target_link_libraries(c50_cpp_contracts PRIVATE C50::cpp)
    list(APPEND C50_COMPILE_TARGETS c50_cpp_contracts)
    list(APPEND C50_EXECUTABLE_TARGETS c50_cpp_contracts)
    add_executable(c50_cpp_api tests/cpp_api.cpp)
    target_compile_features(c50_cpp_api PRIVATE cxx_std_17)
    target_link_libraries(c50_cpp_api PRIVATE C50::cpp)
    list(
        APPEND C50_COMPILE_TARGETS
        c50_input_test
        c50_output_test
        c50_data_input_test
        c50_reference_compatibility
        c50_cpp_api
    )
    list(
        APPEND C50_EXECUTABLE_TARGETS
        c50_input_test
        c50_output_test
        c50_data_input_test
        c50_reference_compatibility
        c50_cpp_api
    )
endif()

if(C50_BUILD_BENCHMARKS)
    find_package(Catch2 3 CONFIG REQUIRED)

    add_executable(c50_native_workload benchmarks/native_workload.cpp)
    target_compile_features(c50_native_workload PRIVATE cxx_std_17)
    target_link_libraries(c50_native_workload PRIVATE C50::core)
    set_target_properties(
        c50_native_workload
        PROPERTIES OUTPUT_NAME c50-native-workload
    )

    add_executable(c50_native_benchmarks benchmarks/native_benchmarks.cpp)
    target_compile_features(c50_native_benchmarks PRIVATE cxx_std_17)
    target_include_directories(
        c50_native_benchmarks
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(
        c50_native_benchmarks
        PRIVATE C50::core Catch2::Catch2WithMain
    )
    set_target_properties(
        c50_native_benchmarks
        PROPERTIES OUTPUT_NAME c50-native-benchmarks
    )

    list(
        APPEND C50_COMPILE_TARGETS
        c50_native_workload
        c50_native_benchmarks
    )
    list(
        APPEND C50_EXECUTABLE_TARGETS
        c50_native_workload
        c50_native_benchmarks
    )

    if(BUILD_TESTING)
        add_test(
            NAME native-sort-properties
            COMMAND c50_native_benchmarks "[sort][property]"
        )
        add_test(
            NAME native-workload-smoke
            COMMAND c50_native_workload
                    --rows 2000
                    --features 12
                    --categorical-features 2
        )
        add_test(
            NAME parallel-workload-equivalence
            COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_parallel_workload.sh"
                    "$<TARGET_FILE:c50_native_workload>"
        )
    endif()
endif()

install(
    TARGETS c50_core c50_cpp
    EXPORT C50Targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(
    DIRECTORY include/c50
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

configure_package_config_file(
    cmake/C50Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/C50Config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/C50
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/C50ConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/C50Config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/C50ConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/C50
)
install(
    EXPORT C50Targets
    FILE C50Targets.cmake
    NAMESPACE C50::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/C50
)

if(BUILD_TESTING OR C50_BUILD_FUZZERS)
    add_library(c50_fuzz_target STATIC tests/fuzz/parser_target.cpp)
    target_link_libraries(c50_fuzz_target PUBLIC C50::cpp)
    list(APPEND C50_COMPILE_TARGETS c50_fuzz_target)

    add_executable(c50_fuzz_seed_replay tests/fuzz/replay.cpp)
    target_link_libraries(c50_fuzz_seed_replay PRIVATE c50_fuzz_target)
    list(APPEND C50_COMPILE_TARGETS c50_fuzz_seed_replay)
    list(APPEND C50_EXECUTABLE_TARGETS c50_fuzz_seed_replay)
endif()

if(C50_BUILD_FUZZERS)
    get_filename_component(c50_fuzz_compiler "${CMAKE_CXX_COMPILER}" NAME)
    if(NOT c50_fuzz_compiler MATCHES "^afl-")
        message(FATAL_ERROR
                "Parser fuzzing requires an AFL++ C++ compiler wrapper")
    endif()
    find_program(C50_AFL_FUZZ_EXECUTABLE NAMES afl-fuzz REQUIRED)
    find_program(C50_SH_EXECUTABLE NAMES sh REQUIRED)

    add_custom_target(
        fuzz-seeds
        COMMAND $<TARGET_FILE:c50_fuzz_seed_replay>
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/fuzz/corpus"
        DEPENDS c50_fuzz_seed_replay
        VERBATIM
    )
    add_custom_target(fuzz-smoke)
    if(C50_ENABLE_SANITIZERS)
        set(c50_fuzz_memory_limit none)
    else()
        set(c50_fuzz_memory_limit 512)
    endif()
    set(fuzz_kind 0)
    foreach(parser schema implicit tree rules cases costs)
        math(EXPR fuzz_kind "${fuzz_kind} + 1")
        set(target c50_fuzz_${parser})
        add_executable(${target} tests/fuzz/parser.cpp)
        target_compile_definitions(${target} PRIVATE C50_FUZZ_KIND=${fuzz_kind})
        # AFL++ interprets this compatibility option by linking aflpp_driver.
        target_link_options(${target} PRIVATE -fsanitize=fuzzer)
        target_link_libraries(${target} PRIVATE c50_fuzz_target)
        list(APPEND C50_COMPILE_TARGETS ${target})
        list(APPEND C50_EXECUTABLE_TARGETS ${target})
        add_custom_target(
            fuzz-smoke-${parser}
            COMMAND
                ${CMAKE_COMMAND} -E env
                AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1
                AFL_NO_UI=1
                AFL_SKIP_CPUFREQ=1
                C50_AFL_MEMORY_LIMIT=${c50_fuzz_memory_limit}
                ${C50_SH_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/fuzz/run-afl.sh"
                "${C50_AFL_FUZZ_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/fuzz/corpus/${parser}"
                "${CMAKE_CURRENT_BINARY_DIR}/afl-output/${parser}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/fuzz/parser.dict"
                5
                $<TARGET_FILE:${target}>
            DEPENDS ${target}
            USES_TERMINAL
            VERBATIM
        )
        add_dependencies(fuzz-smoke fuzz-smoke-${parser})
    endforeach()
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
    add_test(NAME input-stream COMMAND c50_input_test)
    add_test(NAME output-stream COMMAND c50_output_test)
    add_test(NAME data-input-stream COMMAND c50_data_input_test)
    add_test(NAME reference-compatibility COMMAND c50_reference_compatibility)
    add_test(NAME cost-validation COMMAND c50_cost_validation)
    add_test(NAME parser-validation COMMAND c50_parser_validation)
    set_tests_properties(parser-validation PROPERTIES TIMEOUT 120)
    add_test(
        NAME parser-fuzz-seeds
        COMMAND c50_fuzz_seed_replay
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/fuzz/corpus"
    )
    add_test(NAME cpp-validation COMMAND c50_cpp_validation)
    add_test(NAME operation-failure COMMAND c50_operation_failure)
    add_test(NAME cpp-api COMMAND c50_cpp_api)
    add_test(NAME cpp-contracts COMMAND c50_cpp_contracts)
    if(C50_BUILD_PYTHON)
        add_test(
            NAME python-api
            COMMAND ${Python_EXECUTABLE} -m pytest
                    "${CMAKE_CURRENT_SOURCE_DIR}/tests/python"
        )
        set_tests_properties(
            python-api
            PROPERTIES
                ENVIRONMENT
                    "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python;SCIPY_ARRAY_API=1"
        )
        add_test(
            NAME benchmark-smoke
            COMMAND ${Python_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/tests/benchmark_smoke.py"
        )
        set_tests_properties(
            benchmark-smoke
            PROPERTIES
                ENVIRONMENT
                    "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python"
        )
        if(NOT C50_ENABLE_SANITIZERS AND NOT C50_ENABLE_THREAD_SANITIZER)
            add_test(
                NAME python-typecheck
                COMMAND ${Python_EXECUTABLE} -m mypy
            )
            set_tests_properties(
                python-typecheck
                PROPERTIES
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            )
            add_test(
                NAME python-stubtest
                COMMAND
                    ${Python_EXECUTABLE} -m mypy.stubtest
                    --allowlist
                    "${CMAKE_CURRENT_SOURCE_DIR}/tests/typing/stubtest-allowlist.txt"
                    c50._c50
            )
            set_tests_properties(
                python-stubtest
                PROPERTIES
                    ENVIRONMENT
                        "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python"
            )
        endif()
    endif()
    if(NOT C50_ENABLE_SANITIZERS AND NOT C50_ENABLE_THREAD_SANITIZER)
        add_test(
            NAME install-consumer
            COMMAND
                ${CMAKE_COMMAND}
                "-DC50_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
                "-DC50_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}"
                "-DC50_GENERATOR=${CMAKE_GENERATOR}"
                "-DC50_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}"
                "-DC50_GENERATOR_TOOLSET=${CMAKE_GENERATOR_TOOLSET}"
                "-DC50_CONFIG=$<CONFIG>"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_install.cmake"
        )
    endif()
    # This inventory uses GNU nm output and ELF symbol conventions.
    if(CMAKE_NM AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
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
    if(CMAKE_CXX_FLAGS MATCHES "(^|[ ;])-DVerbOpt($|[ ;])")
        add_test(
            NAME cli-verbose-regression
            COMMAND
                ${CMAKE_COMMAND} -E env
                "C50_BINARY=$<TARGET_FILE:c50_cli>"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_verbose.sh"
        )
    endif()
    add_test(
        NAME report-regression
        COMMAND
            ${CMAKE_COMMAND} -E env
            "REPORT_BINARY=$<TARGET_FILE:c50_report>"
            "C50_BINARY=$<TARGET_FILE:c50_cli>"
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
