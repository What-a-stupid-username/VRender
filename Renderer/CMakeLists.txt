include_directories(${GLUT_INCLUDE_DIR})
add_definitions(-DGLUT_FOUND -DGLUT_NO_LIB_PRAGMA)

add_sample_executable(Renderer 
	CommonInclude.h
	VDebug.h

    main.cpp

	OptiXLayer.h
	OptiXLayer.cpp
	
	GlutEvent.h
	GUI.hpp

	Support/Basic.h
	Support/Basic.cpp

	Support/Components.h
	Support/Components.cpp

	Support/Scene.h
	Support/Scene.cpp
	
	Support/objLoader.h

	Support/objLoader.h

	Support/stb_image.h
	Support/stb_image.cpp

	resource.rc
)

set(CMAKE_INCLUDE_CURRENT_DIR ON)