#pragma once

#include "Pipeline/Pipeline.hpp"

namespace VRender {

	class DefaultPipeline : public VPipeline
	{
		int ray_trace_index, blit_index;

		optix::Buffer helper_Buffer;

		VTexture test;

	public:

		DefaultPipeline() : VPipeline() {
			ray_trace_index = PipelineUtility::AddDispatch<true>("path_tracer_camera");
			blit_index = PipelineUtility::AddDispatch("blit");
			
			helper_Buffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT4, 512, 512, false);

			test = VTextureManager::Find("baseColor.sRGB.bmp");
		};

		void Render(VCamera& camera, const VRenderContext& renderContext) {
			//change helper_buffer to target_buffer's size
			ChangeBufferSize(helper_Buffer, renderContext.target);

			//bind scene data
			PipelineUtility::SetGlobalProperties("top_object", renderContext.root);
			PipelineUtility::SetGlobalProperties("lights", renderContext.lights);

			//set render target
			vector<optix::Buffer> rts;
			rts.push_back(renderContext.target);
			rts.push_back(helper_Buffer);

			PipelineUtility::SetRenderTarget(rts);

			//setup camera data
			PipelineUtility::SetupCameraProperties(camera);

			PipelineUtility::Dispatch<true>(ray_trace_index);


			PipelineUtility::SetRenderTarget(renderContext.target);

			//set texture
			//PipelineUtility::SetGlobalTexture("mainTex", test);
			//post
			//PipelineUtility::Dispatch<false>(blit_index);
		}

		void ChangeBufferSize(optix::Buffer buffer, optix::Buffer to_buffer) {
			
			RTsize a, b, c, d;
			buffer->getSize(a, b);
			to_buffer->getSize(c, d);
			if (a != c || b != d) {
				buffer->setSize(c, d);
			}
		}
	};
}
