cmake_minimum_required(VERSION 3.20)

project(C50InstallConsumer LANGUAGES C CXX)

include(CTest)

find_package(C50 2.07 CONFIG REQUIRED)

foreach(standard IN ITEMS 17 20)
    add_executable(c50_install_cpp${standard} cpp_consumer.cpp)
    set_target_properties(c50_install_cpp${standard} PROPERTIES
                         CXX_STANDARD ${standard} CXX_STANDARD_REQUIRED YES
                         CXX_EXTENSIONS NO)
    target_link_libraries(c50_install_cpp${standard} PRIVATE C50::cpp)
    add_test(NAME installed-cpp${standard}-api COMMAND c50_install_cpp${standard})
endforeach()
