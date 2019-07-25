#pragma once

#include "Pipeline/Pipeline.hpp"
#include "Component/Light.hpp"
#include <mutex>

namespace VRender {

	namespace prime {
		class VTransformHandle {
			VMesh mesh;
			VMaterial mat;
			optix::Transform trans;
			optix::GeometryGroup group;
			optix::Acceleration acc;
			optix::GeometryInstance instance;
			optix::GeometryTriangles triangle;

		public:
			optix::Transform GetPrime() {
				return trans;
			}

			VTransformHandle() {
				auto& context = OptixInstance::Instance().Context();
				trans = context->createTransform();
				group = context->createGroup();
				acc = context->createAcceleration("Trbvh");
				instance = context->createGeometryInstance();
				triangle = context->createGeometryTriangles();
				trans->setChild(group);
				group->setAcceleration(acc);
				group->addChild(instance);
				instance->setGeometryTriangles(triangle);
				instance->addMaterial(mat->material);
				
				RTsize size = -1; mesh->v_index_buffer->getSize(size);
				if (size == -1) throw Exception("Error mesh!");
				triangle->setPrimitiveCount(size);
				mesh->vert_buffer->getSize(size);
				triangle->setVertices(size, mesh->vert_buffer, RT_FORMAT_FLOAT3);
				triangle->setTriangleIndices(mesh->v_index_buffer, RT_FORMAT_UNSIGNED_INT3);
				triangle->setBuildFlags(RTgeometrybuildflags::RT_GEOMETRY_BUILD_FLAG_NONE);

				triangle["vertex_buffer"]->setBuffer(mesh->vert_buffer);
				triangle["v_index_buffer"]->setBuffer(mesh->v_index_buffer);

				auto& instance = OptixInstance::Instance();

				triangle["normal_buffer"]->setBuffer(instance.float3_default);
				triangle["n_index_buffer"]->setBuffer(instance.int3_default);
				triangle["texcoord_buffer"]->setBuffer(mesh->tex_buffer);
				triangle["t_index_buffer"]->setBuffer(mesh->t_index_buffer);
			}
			~VTransformHandle() {
				SAFE_RELEASE_OPTIX_OBJ(trans);
				SAFE_RELEASE_OPTIX_OBJ(group);
				SAFE_RELEASE_OPTIX_OBJ(acc);
				SAFE_RELEASE_OPTIX_OBJ(instance);
				SAFE_RELEASE_OPTIX_OBJ(triangle);
			}
		};
	}




	class VRenderer {

		optix::Context context;

		unique_ptr<VPipeline> pipeline;
		unique_ptr<prime::VTransformHandle> transform_handle;

		optix::Buffer result = NULL;
		optix::Buffer target = NULL;
		optix::Buffer id_mask = NULL;

		mutex mut;

		uint2 resultSize;

		bool render = true, end = false;

		VCamera camera;

		int delta = 5;

		VRenderContext render_context;

		thread render_thread;

		int renderer_post_dispatch_index;

		int selected_object_id = -1;

		int frame = 0;

	protected:

		void PrepareRenderContext() {

			bool changed = VMaterialManager::ApplyAllPropertiesChanged();

			changed |= prime::PrimeLightManager::RegenerateLightBuffer();

			changed |= prime::PrimeObjectManager::RebindObjectComponents();

			changed |= prime::PrimeComponentManager::ApplyChanges();

			if (changed) {
				prime::PrimeObjectManager::Root()->getAcceleration()->markDirty();
				camera->staticFrameNum = 0;
			}
			
			render_context.target = result;
			render_context.root = prime::PrimeObjectManager::Root();
			render_context.lights = prime::PrimeLightManager::LightBuffer();

			context["selected_object_id"]->setInt(selected_object_id);
		}

	public:

		static VRenderer& Instance();

		VRenderer() {
			camera = VCamera(new prime::VCameraObj());

			context = OptixInstance::Instance().Context();

			transform_handle = unique_ptr<prime::VTransformHandle>(new prime::VTransformHandle());

			PipelineUtility::SetupContext();

			SetResultSize(make_uint2(512, 512));

			renderer_post_dispatch_index = PipelineUtility::AddDispatch("renderer_post");

			PipelineUtility::SetGlobalProperties("ID_MASK", id_mask);
			PipelineUtility::SetGlobalProperties("TARGET", target);
			PipelineUtility::SetGlobalProperties("handle_object", transform_handle->GetPrime());

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

								PipelineUtility::SetRenderTarget(result);
								PipelineUtility::Dispatch<false>(renderer_post_dispatch_index);
								frame++;
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
					if (pipeline != nullptr) {
						mut.lock();
						try
						{
							PrepareRenderContext();
							PipelineUtility::SetRenderTarget(result);
							PipelineUtility::Dispatch<false>(renderer_post_dispatch_index);
						}
						catch (const Exception& e)
						{
							cout << e.getErrorString() << endl;
						}
						mut.unlock();
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

		optix::Buffer GetRenderTarget() {
			return target;
		}
		optix::Buffer GetIDMask() {
			return id_mask;
		}

		void SetSelectedObject(const int& id) {
			selected_object_id = id;
		}
		int GetSelectedObject() {
			return  selected_object_id;
		}

		int GlobalFrameNumber() {
			return frame;
		}

		void SetResultSize(uint2 size) {
			mut.lock();

			SAFE_RELEASE_OPTIX_OBJ(result);
			SAFE_RELEASE_OPTIX_OBJ(id_mask);

			resultSize = size;
			result = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT4, resultSize.x, resultSize.y, false);
			target = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT4, resultSize.x, resultSize.y, false);
			id_mask = sutil::createOutputBuffer(context, RT_FORMAT_INT, resultSize.x, resultSize.y, false);

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
