include_directories(${GLUT_INCLUDE_DIR})
add_definitions(-DGLUT_FOUND -DGLUT_NO_LIB_PRAGMA)

add_sample_executable(Renderer 
    main.cpp
    optixPathTracer.cu
    optixPathTracer.h
	
	CommonInclude.h
	GlutEvent.h
	OptiXLayer.h
	OptiXLayer.cpp
	VDebug.h


    glviewer.h
    glviewer.cpp
    vector_widget.h
    vector_widget.cpp
    transform_widget.h
    transform_widget.cpp


    # These files are common among multiple samples
    parallelogram.cu
    random.h
)

set(CMAKE_INCLUDE_CURRENT_DIR ON)