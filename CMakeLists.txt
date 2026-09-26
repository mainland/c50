cmake_minimum_required(VERSION 3.20)

project(C50 VERSION 2.07 LANGUAGES C CXX)

option(C50_ENABLE_SANITIZERS
       "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(C50_ENABLE_THREAD_SANITIZER "Enable ThreadSanitizer" OFF)
option(C50_BUILD_PYTHON "Build the Python extension module" OFF)
option(C50_BUILD_BENCHMARKS "Build native correctness tests and benchmarks" OFF)

if(C50_ENABLE_SANITIZERS AND C50_ENABLE_THREAD_SANITIZER)
    message(FATAL_ERROR "AddressSanitizer and ThreadSanitizer cannot be combined")
endif()

include(CTest)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
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
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
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
        python/c50/_data.py
        ${CMAKE_CURRENT_BINARY_DIR}/python/c50/_data.py
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

add_executable(c50_report src/report.c)
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
        add_test(NAME allocation-failure COMMAND c50_allocation_failure)
    endif()
    add_executable(c50_cost_validation tests/cost_validation.cpp)
    target_link_libraries(c50_cost_validation PRIVATE C50::cpp)
    list(APPEND C50_COMPILE_TARGETS c50_cost_validation)
    list(APPEND C50_EXECUTABLE_TARGETS c50_cost_validation)
    add_executable(c50_parser_validation tests/parser_validation.cpp)
    target_link_libraries(c50_parser_validation PRIVATE C50::cpp)
    list(APPEND C50_COMPILE_TARGETS c50_parser_validation)
    list(APPEND C50_EXECUTABLE_TARGETS c50_parser_validation)
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
