include_directories(${GLUT_INCLUDE_DIR})
add_definitions(-DGLUT_FOUND -DGLUT_NO_LIB_PRAGMA)

add_sample_executable(Renderer 
    main.cpp
    path_tracer.cu
    photon_map.cu
    optixPathTracer.h
	
	CommonInclude.h
	GlutEvent.h
	OptiXLayer.h
	OptiXLayer.cpp
	VDebug.h
	GUI.hpp
)

set(CMAKE_INCLUDE_CURRENT_DIR ON)