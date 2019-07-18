#pragma once

#include "Pipeline/Pipeline.hpp"
#include "Component/Light.hpp"
#include <mutex>

namespace VRender {

	class VRenderer {

		optix::Context context;

		unique_ptr<VPipeline> pipeline;

		optix::Buffer result;

		mutex mut;

		uint2 resultSize;

		bool render = true, end = false;

		VCamera camera;

		int delta = 10;

		VRenderContext render_context;

		thread render_thread;

	protected:

		void PrepareRenderContext() {
			prime::PrimeObjectManager::RebindObjectComponents();

			prime::PrimeComponentManager::ApplyChanges();

			prime::PrimeLightManager::RegenerateLightBuffer();

			render_context.target = result;
			render_context.root = prime::PrimeObjectManager::Root();
			render_context.lights = prime::PrimeLightManager::LightBuffer();
		}

	public:

		static VRenderer& Instance();

		VRenderer() {
			camera = VCamera(new prime::VCameraObj());

			context = OptixInstance::Instance().Context();

			PipelineUtility::SetupContext();

			SetResultSize(make_uint2(512, 512));

			render_thread = thread([&]() {
				int last_time = -1000;
				Sleep(10);

				while (!end) {
					while (render)
					{
						int now_time = clock();
						if (now_time - last_time < delta) {
							Sleep(1);
							continue;
						}
						last_time = now_time;
						if (pipeline != nullptr) {
							mut.lock();
							try
							{
								PrepareRenderContext();

								pipeline->Render(camera, render_context);
							}
							catch (const Exception & e)
							{
								cout << e.getErrorString() << endl;
							}
							mut.unlock();
						}
						else
						{
							Sleep(100);
						}
					}
					Sleep(100);
				}
			});
			render_thread.detach();
		}

		template<typename T>
		void SetupPipeline() {
			mut.lock();

			pipeline = unique_ptr<T>(new T());

			mut.unlock();
		}

		optix::Buffer GetRenderResult() {
			return result;
		}

		void SetResultSize(uint2 size) {
			mut.lock();

			SAFE_RELEASE_OPTIX_OBJ(result);

			resultSize = size;
			result = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT4, resultSize.x, resultSize.y, false);

			mut.unlock();
		}

		void EnableRenderer(bool enable) {
			render = enable;
		}

		VCamera Camera() { return camera; }

		void Lock() { mut.lock(); }
		void Unlock() { mut.unlock(); }
		void Join() { end = true; render_thread.join(); }
	};
}
