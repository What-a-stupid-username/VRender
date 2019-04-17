include_directories(${GLUT_INCLUDE_DIR})
add_definitions(-DGLUT_FOUND -DGLUT_NO_LIB_PRAGMA)

add_sample_executable(Renderer 
	CommonInclude.h
	VDebug.h

    main.cpp

    DataBridge.h

	OptiXLayer.h
	OptiXLayer.cpp
	
	GlutEvent.h
	GUI.hpp

	Support/Material.h
	Support/Material.cpp

	Support/Basic.h
	Support/Basic.cpp
)

set(CMAKE_INCLUDE_CURRENT_DIR ON)