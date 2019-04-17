cmake_minimum_required(VERSION 2.8.8 FATAL_ERROR)


project(Material)

SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/libs)


############
#group files
############
file(GLOB_RECURSE Material_CURRENT_HEADERS  *.h *.hpp)
file(GLOB Material_CURRENT_SOURCES  *.cpp)
source_group("Include" FILES ${Material_CURRENT_HEADERS}) 
source_group("Source" FILES ${Material_CURRENT_SOURCES})


############
#include dir
############
set(Material_INCLUDE_DIR CMAKE_CURRENT_SOURCE_DIR)
INCLUDE_DIRECTORIES(Material_INCLUDE_DIR)


############
#  feed back
############
mark_as_advanced(Material_INCLUDE_DIR)

ADD_LIBRARY(Material STATIC ${Material_CURRENT_HEADERS} ${Material_CURRENT_SOURCES})