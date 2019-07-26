#pragma once

#include "../Support/Basic.h"
#include "../cuda/DataBridge.h"
#include "Component/Object.hpp"
#include <unordered_set>


namespace VRender {

	namespace prime {

		struct VLightObj {
			enum Type { ParallelogramLight = 0 };
			string name;
			Type type;
			float3 position;
			float3 rotation;
			float3 scale;
			float3 color;
			float emission;
		};
		typedef shared_ptr<VLightObj> VLight;

		class PrimeLightManager {
			static vector<VLight> lights;
			static vector<VObject> light_objects;
			static unordered_set<int> dirty_lights;
			static Buffer light_buffer;
		public:
			static void TryInit() {
				if (light_buffer == NULL) {
					light_buffer = OptixInstance::Instance().Context()->createBuffer(RT_BUFFER_INPUT);
					light_buffer->setFormat(RT_FORMAT_USER);
					light_buffer->setElementSize(sizeof(Light));
					light_buffer->setSize(0u);
				}
			}

			static int CreateLight(int type, string mat) {
				TryInit();

				if (type == 0) {

					VLight light = VLight(new VLightObj());
					light->type = VLightObj::Type(type);
					light->position = make_float3(0, 0, 0);
					light->rotation = make_float3(0, 0, 0);
					light->scale = make_float3(1, 1, 1);
					light->emission = 15;
					light->color = make_float3(1, 1, 1);
					lights.push_back(light);
					int id = lights.size() - 1;

					VObject obj = VObjectManager::CreateNewObject();
					obj->light_id = id;
					auto filter = VComponents::Create<PtrVMeshFilter>();
					filter->SetMesh(VResources::Find<VMesh>("Quad.obj"));
					obj->SetComponent(filter);
					auto renderer = VComponents::Create<PtrVMeshRenderer>();
					renderer->SetMaterial(VResources::Find<VMaterial>(mat));
					obj->SetComponent(renderer);
					obj->GetPrimeInstance()["light_id"]->setInt(id);

					light_objects.push_back(obj);

					MarkDirty(id);
					return id;
				}

			}

			static VLight GetLight(const int& id) {
				return lights[id];
			}

			static void MarkDirty(const int& id) {
				dirty_lights.insert(id);
			}

			static bool RegenerateLightBuffer() {
				TryInit();
				if (dirty_lights.empty()) return false;

				RTsize size;
				light_buffer->getSize(size);
				if (size != lights.size())
					light_buffer->setSize(lights.size());

				{
					auto ptr = (Light*)light_buffer->map();
					for each (auto & id in dirty_lights)
					{
						VLight vlight = lights[id];

						light_objects[id]->name = vlight->name;
						PtrVTransform light_obj_trans = light_objects[id]->Transform();
						*light_obj_trans->Position<float3>() = vlight->position;
						*light_obj_trans->Rotation<float3>() = vlight->rotation;
						*light_obj_trans->Scale<float3>() = vlight->scale;
						light_obj_trans->MarkDirty();

						Light light;
						light.type = vlight->type;
						light.emission = vlight->emission * vlight->color;

						Matrix4x4 mat;
						mat = Matrix4x4::scale(vlight->scale);
						mat = Matrix4x4::rotate(vlight->rotation.x / 180 * M_PI, make_float3(1, 0, 0)) * mat;
						mat = Matrix4x4::rotate(vlight->rotation.y / 180 * M_PI, make_float3(0, 1, 0)) * mat;
						mat = Matrix4x4::rotate(vlight->rotation.z / 180 * M_PI, make_float3(0, 0, 1)) * mat;
						mat = Matrix4x4::translate(vlight->position) * mat;

						switch (light.type)
						{
						case 0:
							light.a = make_float3(mat * make_float4(-0.5, 0, -0.5, 1));
							light.b = make_float3(mat * make_float4(1, 0, 0, 0));
							light.c = make_float3(mat * make_float4(0, 0, 1, 0));
							light.d = normalize(make_float3(mat * make_float4(0, 1, 0, 0)));
						default:
							break;
						}

						memcpy(ptr + id, &light, sizeof(Light));
					}
					light_buffer->unmap();
				}

				dirty_lights.clear();
				return true;
			}

			static optix::Buffer LightBuffer() {
				return light_buffer;
			}

			static void RemoveAll() {
				lights.clear();
				light_objects.clear();
				dirty_lights.clear();
			}
		};
	}
	typedef prime::VLight VLight;

	class VLightManager {
	public:
		static int CreateLight(int type, string material) { return prime::PrimeLightManager::CreateLight(type, material); }
		static void MarkDirty(const int& id) { prime::PrimeLightManager::MarkDirty(id); }
		static VLight GetLight(const int& id) { return prime::PrimeLightManager::GetLight(id); }
		static void RemoveAll() { prime::PrimeLightManager::RemoveAll(); }
	};

}