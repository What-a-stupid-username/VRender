project(project)

# required cmake version
cmake_minimum_required(VERSION 3.9)

enable_language(CUDA)

file(GLOB_RECURSE CURRENT_HEADERS  *.h *.hpp *.cuh)
file(GLOB CURRENT_SOURCES  *.cpp *.cu)

source_group("Include" FILES ${CURRENT_HEADERS}) 
source_group("Source" FILES ${CURRENT_SOURCES}) 

add_executable(project ${CURRENT_HEADERS} ${CURRENT_SOURCES})