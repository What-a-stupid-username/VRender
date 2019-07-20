#pragma once

#include "../Support/Basic.h"
#include "../Component/Object.hpp"

namespace VRender {

	namespace prime{
		struct VCameraObj{
			// Camera state
			float3		position;
			float3      up;
			float3      forward;
			float3      right;
			float2		fov;

			bool dirty;

			uint staticFrameNum = 1;

			VCameraObj() {
				position = make_float3(0, 0, -2);
				up = make_float3(0, 1, 0);
				forward = make_float3(0, 0, 1);
				right = make_float3(1, 0, 0);
				fov = make_float2(45, 45);
				dirty = true;
			};
		};
	}
	typedef shared_ptr<prime::VCameraObj> VCamera;


	class PipelineUtility {
		static optix::Context context;

		static RTsize target_width, target_height;

		static list<VDispatch> dispatchs;

		static VCamera camera;
	public:

		static void SetupContext() {
			context = OptixInstance::Instance().Context();
			
			context->setStackSize(2000);
		}

		static void SetRenderTarget(vector<optix::Buffer> buffers) {
			int i = 0;
			for each (auto buffer in buffers)
			{
				context["V_TARGET" + to_string(i++)]->set(buffer);
			}
			buffers[0]->getSize(target_width, target_height);
		}

		static void SetRenderTarget(optix::Buffer buffer) {
			context["V_TARGET0"]->set(buffer);
			buffer->getSize(target_width, target_height);
		}

		static int AddDispatch(string name) {
			
			VDispatch dispatch = VResources::Find<VDispatch>(name);
			dispatchs.push_back(dispatch);
			
			unsigned int size = dispatchs.size();
			size = max(size, context->getEntryPointCount());

			context->setEntryPointCount(size);
			
			size--;
			
			context->setRayGenerationProgram(size, dispatch->rayGenerationProgram);
			context->setExceptionProgram(size, dispatch->exceptionProgram);
			context->setMissProgram(size, dispatch->missProgram);

			return size;
		}

		template<bool cameraDispatch = false>
		static void Dispatch(int dispatchIndex) {}
		template<>
		static void Dispatch<true>(int dispatchIndex) {
			context->launch(dispatchIndex, target_width, target_height);
			camera->staticFrameNum++;
		}
		template<>
		static void Dispatch<false>(int dispatchIndex) {
			context->launch(dispatchIndex, target_width, target_height);
		}


		template<typename T>
		static void SetGlobalProperties(string name, T value) {
			context[name]->set(value);
		}

		static void SetGlobalTexture(string name, VTexture texture) {
			int id = texture->ID();
			context[name]->setUserData(sizeof(int), (void*)& id);
		}
		
		static void SetupCameraProperties(VCamera& cam) {
			camera = cam;
			if (cam->dirty) {
				cam->staticFrameNum = 0;
				cam->dirty = false;
			}
			context["camera_position"]->setFloat(camera->position);
			context["camera_up"]->setFloat(camera->up);
			context["camera_forward"]->setFloat(camera->forward);
			context["camera_right"]->setFloat(camera->right);
			context["camera_fov"]->setFloat(make_float2(tan(camera->fov.x / 180 * 3.14159265389), tan(camera->fov.y / 180 * 3.14159265389)));
			context["camera_staticFrameNum"]->setUint(camera->staticFrameNum);
		}
	};
}
