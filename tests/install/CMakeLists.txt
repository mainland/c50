# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

cmake_minimum_required(VERSION 3.20)

project(C50InstallConsumer LANGUAGES C CXX)

include(CTest)

find_package(C50 1.0 CONFIG REQUIRED)

foreach(standard IN ITEMS 17 20)
    add_executable(c50_install_cpp${standard} cpp_consumer.cpp)
    set_target_properties(c50_install_cpp${standard} PROPERTIES
                         CXX_STANDARD ${standard} CXX_STANDARD_REQUIRED YES
                         CXX_EXTENSIONS NO)
    target_link_libraries(c50_install_cpp${standard} PRIVATE C50::cpp)
    add_test(NAME installed-cpp${standard}-api COMMAND c50_install_cpp${standard})
endforeach()

# The consumer defines global names that the core uses internally. Linking
# fails with a multiple definition if the core leaves them in the global
# namespace.
add_executable(c50_install_symbol_collision symbol_collision.cpp)
set_target_properties(c50_install_symbol_collision PROPERTIES
                     CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES
                     CXX_EXTENSIONS NO)
target_link_libraries(c50_install_symbol_collision PRIVATE C50::cpp)
add_test(NAME installed-symbol-collision COMMAND c50_install_symbol_collision)
