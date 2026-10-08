cmake_minimum_required(VERSION 3.20)

# The library and Python package share a version, which differs from the
# imported C5.0 release number. CMake versions have no prerelease labels, so
# prereleases such as 1.0.0a1 use the version they precede.
project(C50 VERSION 1.0.0 LANGUAGES CXX)

# A Python build from a release tag must match the library version. Builds
# between tags have development versions and are not checked.
if(DEFINED SKBUILD_PROJECT_VERSION_FULL
   AND NOT SKBUILD_PROJECT_VERSION_FULL MATCHES "dev|\\+"
   AND NOT SKBUILD_PROJECT_VERSION VERSION_EQUAL PROJECT_VERSION)
    message(FATAL_ERROR
            "Python package version ${SKBUILD_PROJECT_VERSION_FULL} does not "
            "match the library version ${PROJECT_VERSION} in CMakeLists.txt")
endif()

option(C50_ENABLE_SANITIZERS
       "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(C50_ENABLE_THREAD_SANITIZER "Enable ThreadSanitizer" OFF)
option(C50_BUILD_PYTHON "Build the Python extension module" OFF)
option(C50_BUILD_DOCUMENTATION "Build the Doxygen API documentation" OFF)
option(C50_BUILD_FUZZERS "Build AFL++ parser harnesses" OFF)
option(C50_PRIVATE_CORE
       "Compile an uninstalled core with link-time optimization and hidden symbols"
       OFF)

if(C50_ENABLE_SANITIZERS AND C50_ENABLE_THREAD_SANITIZER)
    message(FATAL_ERROR "AddressSanitizer and ThreadSanitizer cannot be combined")
endif()

# Wheels contain only the Python extension, which links the core statically.
# A shared object that exports the core's functions lets the dynamic linker
# replace them, so the compiler cannot inline across them. Hidden symbols make
# them private to the extension, and link-time optimization then inlines across
# the core's source files. A core compiled this way is not installed, because
# its objects hold intermediate code that only the same compiler can link.
# The core and the extension get these properties where they are created. An
# explicit CMAKE_INTERPROCEDURAL_OPTIMIZATION setting takes precedence.
set(c50_core_ipo OFF)
if(C50_PRIVATE_CORE AND NOT DEFINED CMAKE_INTERPROCEDURAL_OPTIMIZATION)
    include(CheckIPOSupported)
    check_ipo_supported(LANGUAGES CXX RESULT c50_ipo_supported
                        OUTPUT c50_ipo_output)
    if(c50_ipo_supported)
        set(c50_core_ipo ON)
    else()
        message(WARNING
                "Compiling the private core without link-time "
                "optimization: ${c50_ipo_output}")
    endif()
endif()

# MSVC deprecates standard C functions such as fopen and sscanf in favor of
# its bounds-checked extensions, and POSIX names such as strdup in favor of
# underscored ones. The sources use the standard and POSIX names.
if(MSVC)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS _CRT_NONSTDC_NO_WARNINGS)
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
    src/c50_locale.cpp
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
if(C50_PRIVATE_CORE)
    set_target_properties(
        c50_core
        PROPERTIES
            CXX_VISIBILITY_PRESET hidden
            VISIBILITY_INLINES_HIDDEN ON
    )
    if(c50_core_ipo)
        set_target_properties(c50_core PROPERTIES
                              INTERPROCEDURAL_OPTIMIZATION ON)
    endif()
endif()
target_include_directories(
    c50_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_features(c50_core PUBLIC cxx_std_17)
target_link_libraries(c50_core PUBLIC Threads::Threads)

# The core is position-independent so that the installed archive can be
# linked into shared objects. GCC then assumes that the dynamic linker may
# replace any function the core exports, and it does not inline across them.
# Nothing replaces the core's functions, so drop that assumption where the
# compiler supports the option.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-fno-semantic-interposition
                        C50_HAS_NO_SEMANTIC_INTERPOSITION)
if(C50_HAS_NO_SEMANTIC_INTERPOSITION)
    target_compile_options(c50_core PRIVATE -fno-semantic-interposition)
endif()

add_library(c50_cpp INTERFACE)
add_library(C50::cpp ALIAS c50_cpp)
set_target_properties(c50_cpp PROPERTIES EXPORT_NAME cpp)
target_compile_features(c50_cpp INTERFACE cxx_std_17)
target_link_libraries(c50_cpp INTERFACE C50::core)

if(C50_BUILD_PYTHON)
    find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(pybind11 CONFIG REQUIRED)
    # pybind11_add_module adds its own link-time optimization flags unless
    # CMAKE_INTERPROCEDURAL_OPTIMIZATION is defined. Define it only for this
    # call, which also initializes the target's INTERPROCEDURAL_OPTIMIZATION
    # property, so that CMake's flags are the only ones.
    if(c50_core_ipo)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    endif()
    pybind11_add_module(c50_python MODULE python/c50/_c50module.cpp)
    if(c50_core_ipo)
        unset(CMAKE_INTERPROCEDURAL_OPTIMIZATION)
    endif()
    if(C50_PRIVATE_CORE)
        set_target_properties(
            c50_python
            PROPERTIES
                CXX_VISIBILITY_PRESET hidden
                VISIBILITY_INLINES_HIDDEN ON
        )
    endif()
    target_compile_features(c50_python PRIVATE cxx_std_17)
    target_link_libraries(c50_python PRIVATE C50::cpp)
    # Multi-configuration generators append a configuration directory to an
    # output directory unless it contains a generator expression. The
    # expression below keeps the extension next to the package files that the
    # build tree copies, for every generator. CMake treats a MODULE library as
    # a library artifact on every platform, so only the library property
    # places it.
    set(c50_python_package_dir "$<1:${CMAKE_CURRENT_BINARY_DIR}/python/c50>")
    set_target_properties(
        c50_python
        PROPERTIES
            OUTPUT_NAME _c50
            LIBRARY_OUTPUT_DIRECTORY "${c50_python_package_dir}"
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
# The POSIX program raises its stack limit to 20 MiB at startup. Windows fixes
# the main-thread stack size at link time instead.
if(MSVC)
    target_link_options(c50_cli PRIVATE /STACK:20971520)
elseif(WIN32)
    target_link_options(c50_cli PRIVATE -Wl,--stack,20971520)
endif()

add_executable(c50_report src/report.cpp)
target_compile_features(c50_report PRIVATE cxx_std_17)
set_target_properties(c50_report PROPERTIES OUTPUT_NAME report)

# Windows cannot execute the shell tests directly. Run them with the POSIX shell
# from Git for Windows.
set(C50_TEST_SHELL_COMMAND)
if(WIN32 AND BUILD_TESTING)
    find_program(
        C50_TEST_SHELL
        NAMES sh
        PATHS "$ENV{ProgramFiles}/Git/usr/bin" "$ENV{ProgramFiles}/Git/bin"
        REQUIRED
    )
    set(C50_TEST_SHELL_COMMAND "${C50_TEST_SHELL}")
endif()

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
        add_executable(c50_no_files tests/no_files.cpp)
        target_link_libraries(c50_no_files PRIVATE C50::cpp)
        target_link_options(c50_no_files PRIVATE -Wl,--wrap=tmpfile,--wrap=fopen)
        list(APPEND C50_COMPILE_TARGETS c50_no_files)
        list(APPEND C50_EXECUTABLE_TARGETS c50_no_files)
        add_test(NAME in-memory-without-files COMMAND c50_no_files)
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
    add_test(NAME model-inspection COMMAND c50_model_inspection
             "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures")
    add_executable(c50_attribute_usage tests/attribute_usage.cpp)
    target_link_libraries(c50_attribute_usage PRIVATE C50::cpp)
    list(APPEND C50_COMPILE_TARGETS c50_attribute_usage)
    list(APPEND C50_EXECUTABLE_TARGETS c50_attribute_usage)
    add_test(NAME attribute-usage COMMAND c50_attribute_usage
             "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    add_executable(c50_predictor tests/predictor.cpp)
    target_link_libraries(c50_predictor PRIVATE C50::cpp)
    list(APPEND C50_COMPILE_TARGETS c50_predictor)
    list(APPEND C50_EXECUTABLE_TARGETS c50_predictor)
    add_test(NAME predictor COMMAND c50_predictor "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures")
    add_executable(c50_implicit_legacy tests/implicit_legacy.cpp)
    target_link_libraries(c50_implicit_legacy PRIVATE C50::cpp)
    list(APPEND C50_COMPILE_TARGETS c50_implicit_legacy)
    list(APPEND C50_EXECUTABLE_TARGETS c50_implicit_legacy)
    add_test(NAME implicit-legacy COMMAND c50_implicit_legacy
             "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures")
    add_executable(c50_cpp_validation tests/cpp_validation.cpp)
    target_link_libraries(c50_cpp_validation PRIVATE C50::cpp)
    add_executable(c50_operation_failure tests/operation_failure.cpp)
    target_include_directories(c50_operation_failure PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(c50_operation_failure PRIVATE C50::cpp)
    list(APPEND C50_COMPILE_TARGETS c50_cpp_validation c50_operation_failure)
    list(APPEND C50_EXECUTABLE_TARGETS c50_cpp_validation c50_operation_failure)
    add_executable(c50_operation_state tests/operation_state.cpp)
    target_link_libraries(c50_operation_state PRIVATE C50::cpp)
    list(APPEND C50_COMPILE_TARGETS c50_operation_state)
    list(APPEND C50_EXECUTABLE_TARGETS c50_operation_state)
    add_executable(c50_locale_independence tests/locale_independence.cpp)
    target_link_libraries(c50_locale_independence PRIVATE C50::cpp)
    list(APPEND C50_COMPILE_TARGETS c50_locale_independence)
    list(APPEND C50_EXECUTABLE_TARGETS c50_locale_independence)
    add_executable(c50_cpp_contracts tests/cpp_contracts.cpp)
    target_link_libraries(c50_cpp_contracts PRIVATE C50::cpp)
    list(APPEND C50_COMPILE_TARGETS c50_cpp_contracts)
    list(APPEND C50_EXECUTABLE_TARGETS c50_cpp_contracts)
    add_executable(c50_cpp_api tests/cpp_api.cpp)
    target_compile_features(c50_cpp_api PRIVATE cxx_std_17)
    target_link_libraries(c50_cpp_api PRIVATE C50::cpp)
    add_executable(c50_sort_properties tests/sort_properties.cpp)
    target_compile_features(c50_sort_properties PRIVATE cxx_std_17)
    target_include_directories(
        c50_sort_properties
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(c50_sort_properties PRIVATE C50::core)
    add_executable(c50_index_overflow tests/index_overflow.cpp)
    target_compile_features(c50_index_overflow PRIVATE cxx_std_17)
    target_include_directories(
        c50_index_overflow
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    add_executable(c50_parallel_workload tests/parallel_workload.cpp)
    target_compile_features(c50_parallel_workload PRIVATE cxx_std_17)
    target_link_libraries(c50_parallel_workload PRIVATE C50::core)
    list(
        APPEND C50_COMPILE_TARGETS
        c50_input_test
        c50_output_test
        c50_data_input_test
        c50_reference_compatibility
        c50_cpp_api
        c50_sort_properties
        c50_index_overflow
        c50_parallel_workload
    )
    list(
        APPEND C50_EXECUTABLE_TARGETS
        c50_input_test
        c50_output_test
        c50_data_input_test
        c50_reference_compatibility
        c50_cpp_api
        c50_sort_properties
        c50_index_overflow
        c50_parallel_workload
    )
endif()

if(NOT C50_PRIVATE_CORE)
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
endif()

# The class-count index check trains a schema of 46,341 classes, which reserves
# about 8.6 GB of address space, so it is not part of the default tests. Run it
# explicitly with `cmake --build build --target index-overflow-large`.
if(UNIX)
    add_executable(
        c50_index_overflow_train EXCLUDE_FROM_ALL
        tests/slow/index_overflow_train.cpp
    )
    target_compile_features(c50_index_overflow_train PRIVATE cxx_std_17)
    target_link_libraries(c50_index_overflow_train PRIVATE C50::cpp)
    list(APPEND C50_COMPILE_TARGETS c50_index_overflow_train)
    add_custom_target(
        index-overflow-large
        COMMAND sh "${CMAKE_CURRENT_SOURCE_DIR}/tests/slow/index_overflow.sh"
                $<TARGET_FILE:c50_cli> $<TARGET_FILE:c50_index_overflow_train>
        DEPENDS c50_cli c50_index_overflow_train
        USES_TERMINAL
        VERBATIM
    )
endif()

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
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
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

# -ffloat-store affects only x87 arithmetic, which keeps excess precision in
# registers. 64-bit targets evaluate at the declared precision, so the flag is
# kept only for 32-bit GCC builds, as the imported Makefile used it.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_SIZEOF_VOID_P EQUAL 4)
    foreach(target IN LISTS C50_COMPILE_TARGETS)
        target_compile_options(${target} PRIVATE -ffloat-store)
    endforeach()
endif()

if(NOT MSVC)
    target_link_libraries(c50_core PUBLIC m)
    target_link_libraries(c50_report PRIVATE m)
endif()

if(C50_ENABLE_SANITIZERS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
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
    add_test(NAME operation-state COMMAND c50_operation_state)
    add_test(NAME locale-independence COMMAND c50_locale_independence)
    set_tests_properties(locale-independence PROPERTIES SKIP_RETURN_CODE 77)
    add_test(NAME cpp-contracts COMMAND c50_cpp_contracts)
    add_test(NAME sort-properties COMMAND c50_sort_properties)
    add_test(NAME index-overflow COMMAND c50_index_overflow)
    add_test(
        NAME parallel-workload-equivalence
        COMMAND ${C50_TEST_SHELL_COMMAND}
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_parallel_workload.sh"
                "$<TARGET_FILE:c50_parallel_workload>"
    )
    if(C50_BUILD_PYTHON)
        add_test(
            NAME python-api
            COMMAND ${Python_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_built_python.py"
                    "${CMAKE_CURRENT_BINARY_DIR}/python" pytest
                    "${CMAKE_CURRENT_SOURCE_DIR}/tests/python"
        )
        set_tests_properties(
            python-api
            PROPERTIES
                ENVIRONMENT
                    "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python;SCIPY_ARRAY_API=1"
        )
        # The benchmark suite requires Linux or macOS.
        if(UNIX)
            add_test(
                NAME benchmark-smoke
                COMMAND ${Python_EXECUTABLE}
                        "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_built_python.py"
                        "${CMAKE_CURRENT_BINARY_DIR}/python"
                        "${CMAKE_CURRENT_SOURCE_DIR}/tests/benchmark_smoke.py"
            )
            set_tests_properties(
                benchmark-smoke
                PROPERTIES
                    ENVIRONMENT
                        "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python;C50_BENCHMARK_MODULE_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}/python"
            )
        endif()
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
                    ${Python_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_built_python.py"
                    "${CMAKE_CURRENT_BINARY_DIR}/python" mypy.stubtest
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
    if(NOT C50_ENABLE_SANITIZERS AND NOT C50_ENABLE_THREAD_SANITIZER
       AND NOT C50_PRIVATE_CORE)
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
    # This inventory uses GNU nm output and ELF symbol conventions. A core
    # compiled with link-time optimization holds intermediate code, which is
    # LLVM bitcode for Clang and slim GNU LTO objects for GCC. Neither records
    # data sections, so nm cannot tell constants from writable data, and the
    # inventory would pass without checking anything. Whether a global is
    # writable does not depend on link-time optimization, and native builds of
    # the same sources run the inventory.
    set(c50_core_lto OFF)
    get_target_property(c50_core_ipo_property c50_core
                        INTERPROCEDURAL_OPTIMIZATION)
    set(c50_core_flags "${CMAKE_CXX_FLAGS}")
    foreach(c50_config IN ITEMS DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
        string(APPEND c50_core_flags " ${CMAKE_CXX_FLAGS_${c50_config}}")
        get_target_property(c50_config_ipo c50_core
                            INTERPROCEDURAL_OPTIMIZATION_${c50_config})
        if(c50_config_ipo)
            set(c50_core_lto ON)
        endif()
    endforeach()
    if(c50_core_ipo OR CMAKE_INTERPROCEDURAL_OPTIMIZATION
       OR c50_core_ipo_property
       OR c50_core_flags MATCHES "(^|[ ;])-flto([ =;]|$)")
        set(c50_core_lto ON)
    endif()
    if(CMAKE_NM AND CMAKE_SYSTEM_NAME STREQUAL "Linux"
       AND NOT c50_core_lto)
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
        NAME makefile-headers
        COMMAND
            ${C50_TEST_SHELL_COMMAND}
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_makefile_headers.sh"
    )
    add_test(
        NAME makefile-float-store
        COMMAND
            ${C50_TEST_SHELL_COMMAND}
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_makefile_float_store.sh"
    )
    set_tests_properties(makefile-float-store PROPERTIES SKIP_RETURN_CODE 77)
    add_test(
        NAME cli-regression
        COMMAND
            ${CMAKE_COMMAND} -E env
            "C50_BINARY=$<TARGET_FILE:c50_cli>"
            ${C50_TEST_SHELL_COMMAND}
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cli.sh"
    )
    if(CMAKE_CXX_FLAGS MATCHES "(^|[ ;])-DVerbOpt($|[ ;])")
        add_test(
            NAME cli-verbose-regression
            COMMAND
                ${CMAKE_COMMAND} -E env
                "C50_BINARY=$<TARGET_FILE:c50_cli>"
                ${C50_TEST_SHELL_COMMAND}
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_verbose.sh"
        )
    endif()
    add_test(
        NAME report-regression
        COMMAND
            ${CMAKE_COMMAND} -E env
            "REPORT_BINARY=$<TARGET_FILE:c50_report>"
            "C50_BINARY=$<TARGET_FILE:c50_cli>"
            ${C50_TEST_SHELL_COMMAND}
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_report.sh"
    )
    add_test(
        NAME prediction-regression
        COMMAND
            ${CMAKE_COMMAND} -E env
            "C50_BINARY=$<TARGET_FILE:c50_cli>"
            "C50_PREDICTION_BINARY=$<TARGET_FILE:c50_prediction_probe>"
            "C50_API_PREDICTION_BINARY=$<TARGET_FILE:c50_api_prediction_probe>"
            ${C50_TEST_SHELL_COMMAND}
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_predictions.sh"
    )
endif()

# The core's objects hold intermediate code, so the link step that consumes
# the archive must load the compiler's link-time optimization plugin. GNU ld
# loads GCC's plugin on its own but not Clang's, and CMake adds the flag that
# loads it only to targets that request link-time optimization themselves.
# Request it for every program that links the private core.
if(c50_core_ipo)
    get_directory_property(c50_all_targets BUILDSYSTEM_TARGETS)
    foreach(c50_target IN LISTS c50_all_targets)
        get_target_property(c50_target_type ${c50_target} TYPE)
        if(c50_target_type STREQUAL "EXECUTABLE")
            set_target_properties(${c50_target} PROPERTIES
                                  INTERPROCEDURAL_OPTIMIZATION ON)
        endif()
    endforeach()
endif()
