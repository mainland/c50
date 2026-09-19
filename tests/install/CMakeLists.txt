cmake_minimum_required(VERSION 3.20)

project(C50InstallConsumer LANGUAGES C CXX)

include(CTest)

find_package(C50 2.07 CONFIG REQUIRED)

add_executable(c50_install_c c_consumer.c)
target_link_libraries(c50_install_c PRIVATE C50::core)

add_executable(c50_install_cpp cpp_consumer.cpp)
target_link_libraries(c50_install_cpp PRIVATE C50::cpp)

add_test(NAME installed-c-api COMMAND c50_install_c)
add_test(NAME installed-cpp-api COMMAND c50_install_cpp)
