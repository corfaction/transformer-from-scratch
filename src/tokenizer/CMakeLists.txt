cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)

project(tokenizer)

set(JSON_INCLUDE_DIR "C:/Users/LocalAdmin/libs" CACHE PATH "Path to nlohmann json")

find_package(pybind11 REQUIRED)
pybind11_add_module(tokenizer tokenizer.cpp)

target_include_directories(tokenizer PRIVATE ${JSON_INCLUDE_DIR})