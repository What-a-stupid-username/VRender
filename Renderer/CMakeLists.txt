include_directories(${GLUT_INCLUDE_DIR})
add_definitions(-DGLUT_FOUND -DGLUT_NO_LIB_PRAGMA)

add_sample_executable(Renderer 

	VRender.hpp
	VRender.cpp

	default_pipeline.hpp

    main.cpp

	Support/CommonInclude.h
	Support/VDebug.h

	Support/Basic.h
	Support/Basic.cpp

	Support/objLoader.h

	Support/stb_image.h
	Support/stb_image.cpp
	
	Component/BasicComponent.hpp
	Component/BasicComponent.cpp

	Component/MeshFilter.hpp
	Component/MeshFilter.cpp

	Component/MeshRenderer.hpp
	Component/MeshRenderer.cpp
	
	Component/Transform.hpp
	Component/Transform.cpp
		
	Component/Object.hpp
	Component/Object.cpp
	
	Component/Light.hpp
	Component/Light.cpp
	
	Manager/Manager.hpp
	Manager/Manager.cpp	
	
	Pipeline/Pipeline.hpp
	Pipeline/Pipeline.cpp
	
	Pipeline/PipelineUtility.hpp
	Pipeline/PipelineUtility.cpp
	
	GUI/resource.rc
	GUI/renderView.hpp
	GUI/UIView.hpp
)

set(CMAKE_INCLUDE_CURRENT_DIR ON)