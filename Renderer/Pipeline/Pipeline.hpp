#pragma once

#include "PipelineUtility.hpp"
#include "../Component/Light.hpp"

namespace VRender {

	struct VRenderContext{
		optix::Group root;
		optix::Buffer lights;
		optix::Buffer target;
	};


	class VPipeline
	{
	protected:
		optix::Context context;
	public:

		VPipeline() { this->context = OptixInstance::Instance().Context(); };

		virtual void Render(VCamera& camera, const VRenderContext& renderContext) = 0;
	};

}
