# The following lines of boilerplate have to be in your project's
# CMakeLists in this exact order for cmake to work correctly
cmake_minimum_required(VERSION 3.16)

set(CMAKE_USE_EMBEDDED_CLASSIFIER TRUE) #uncomment this line to enable classifier

if(CMAKE_USE_EMBEDDED_CLASSIFIER)
    add_compile_definitions(USE_EMBEDDED_CLASSIFIER=1)
    list(APPEND EXTRA_COMPONENT_DIRS ./ai_components/)
endif()

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(workbench)
