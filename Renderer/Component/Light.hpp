#pragma once

#include "../Support/Basic.h"
#include "../cuda/DataBridge.h"
#include <unordered_set>


namespace VRender {

	namespace prime {

		class PrimeLightManager {
			static vector<shared_ptr<ParallelogramLight>> parallelogram_lights;
			static unordered_set<int> dirty_lights;
			static Buffer light_buffer;
		public:
			static void TryInit() {
				if (light_buffer == NULL) {
					light_buffer = OptixInstance::Instance().Context()->createBuffer(RT_BUFFER_INPUT);
					light_buffer->setFormat(RT_FORMAT_USER);
					light_buffer->setElementSize(sizeof(ParallelogramLight));
					light_buffer->setSize(0u);
				}
			}

			static int CreateLight() {
				TryInit();
				const float3 light_em = make_float3(15, 15, 15);
				shared_ptr<ParallelogramLight> light = shared_ptr<ParallelogramLight>(new ParallelogramLight());
				light->corner = make_float3(1, 5, 1);
				light->v1 = make_float3(0, 0, 2);
				light->v2 = make_float3(-2, 0, 0);
				light->normal = normalize(cross(light->v1, light->v2));
				light->emission = light_em;
				parallelogram_lights.push_back(light);
				dirty_lights.insert(parallelogram_lights.size() - 1);
				return parallelogram_lights.size() - 1;
			}

			static shared_ptr<ParallelogramLight> GetLight(const int& id) {
				return parallelogram_lights[id];
			}

			static void MarkDirty(const int& id) {
				dirty_lights.insert(id);
			}

			static void RegenerateLightBuffer() {
				TryInit();
				if (dirty_lights.empty()) return;

				RTsize size;
				light_buffer->getSize(size);
				if (size != parallelogram_lights.size())
					light_buffer->setSize(parallelogram_lights.size());

				{
					auto ptr = (ParallelogramLight*)light_buffer->map();
					for each (auto & light in dirty_lights)
					{
						memcpy(ptr + light, &*parallelogram_lights[light], sizeof(ParallelogramLight));
					}
					light_buffer->unmap();
				}

				dirty_lights.clear();
			}

			static optix::Buffer LightBuffer() {
				return light_buffer;
			}
		};
	}	

	class VLightManager {
	public:
		static int CreateLight() { return prime::PrimeLightManager::CreateLight(); }
		static void MarkDirty(const int& id) { prime::PrimeLightManager::MarkDirty(id); }
		static shared_ptr<ParallelogramLight> GetLight(const int& id) { return prime::PrimeLightManager::GetLight(id); }
	};

}