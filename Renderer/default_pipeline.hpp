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
			ray_trace_index = PipelineUtility::AddDispatch("path_tracer_camera");
			blit_index = PipelineUtility::AddDispatch("blit");
			
			helper_Buffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT4, 512, 512, false);

			test = VTextureManager::Find("baseColor.sRGB.bmp");
		};

		void Render(VCamera& camera, const VRenderContext& renderContext) {

			ChangeBufferSize(helper_Buffer, renderContext.target);

			PipelineUtility::SetGlobalProperties("top_object", renderContext.root);
			PipelineUtility::SetGlobalProperties("lights", renderContext.lights);

			vector<optix::Buffer> rts;
			rts.push_back(renderContext.target);
			rts.push_back(helper_Buffer);

			PipelineUtility::SetRenderTarget(rts);

			PipelineUtility::SetupCameraProperties(camera);

			PipelineUtility::Dispatch<true>(ray_trace_index);


			PipelineUtility::SetRenderTarget(renderContext.target);

			PipelineUtility::SetGlobalTexture("mainTex", test);

			PipelineUtility::Dispatch<false>(blit_index);
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
