#pragma once

#include "BasicComponent.hpp"

#include "Transform.hpp"
#include "MeshFilter.hpp"
#include "MeshRenderer.hpp"


namespace VRender {
	

	namespace prime {

		void* Create();
		
		class VObjectObj {
			friend class PrimeObjectManager;
			friend void* Create();
		private:

		#pragma region Compile Time Check
			template<typename T>
			struct Is_Shared_Ptr {
				template<typename U, int P>
				struct matcher;
				template<typename U>
				static char helper(matcher<U, sizeof(U::element_type)>*);
				template<typename U>
				static int helper(...);
				enum { value = sizeof(helper<T>(NULL)) == 1 };
			};

			template <bool>
			struct SharedPtrWrapper {};
			template<>
			struct SharedPtrWrapper<true> {
				template<typename T>
				static void Check() {
					static_assert(std::is_base_of<VComponent, T::element_type>::value, "Only subclasses of VComponent can be add to object.");
					static_assert(!std::is_same<T::element_type, VComponent>::value, "VComponent can't be add to object.");
				}
			};

			template<>
			struct SharedPtrWrapper<false> {
				template<typename T>
				static void Check() {
					static_assert(std::is_base_of<VComponent, T>::value, "Only subclasses of VComponent can be add to object.");
				}
			};
	#pragma endregion

		protected:			

			optix::GeometryGroup group;
			optix::Acceleration acc;
			optix::GeometryInstance instance;

			VRender::VTransform transform = nullptr;
			VRender::VMeshFilter meshFilter = nullptr;
			VRender::VMeshRenderer meshRenderer = nullptr;

			bool trans_dirty = true, filter_dirty = true, render_dirty = true;

			enum CType { Unkown, Trans, Filter, Renderer};

			template<typename T>
			CType AddComponentWrapper(T comp) { return Unkown; }
	 		template<>
			CType AddComponentWrapper<VRender::VTransform>(VRender::VTransform comp) abandon;
			template<>
			CType AddComponentWrapper<VRender::VMeshFilter>(VRender::VMeshFilter comp) { 
				meshFilter = comp;
				filter_dirty = true;
				return Filter;
			}
			template<>
			CType AddComponentWrapper<VRender::VMeshRenderer>(VRender::VMeshRenderer comp) { 
				meshRenderer = comp;
				render_dirty = true;
				return Renderer;
			}

			VObjectObj() {
				auto context = OptixInstance::Instance().Context();

				group = context->createGeometryGroup();
				acc = context->createAcceleration("Trbvh");
				instance = context->createGeometryInstance();

				group->addChild(instance);
				group->setAcceleration(acc);


				transform = VRender::VComponents::Create<VRender::VTransform>();
			};

			void ApplyRebind() {

				if (transform == nullptr && meshFilter == nullptr || meshRenderer == nullptr) {
					throw Exception("invalid component!");
				}

				if (trans_dirty) {
					transform->Rebind(group);
					trans_dirty = false;
				}
				if (filter_dirty) {
					meshFilter->Rebind(instance);
					filter_dirty = false;
				}
				if (render_dirty) {
					meshRenderer->Rebind(instance);
					render_dirty = false;
				}
			}

			void MarkDirty();

			optix::Transform GetPrimeObj() { return transform->GetPrimeObj(); }
		public:
			string name;
			int light_id = -1;

			VRender::VTransform Transform() { return transform; }
			VRender::VMeshFilter MeshFilter() { return meshFilter; }
			VRender::VMeshRenderer MeshRenderer() { return meshRenderer; }

			
			template<typename T>
			void SetComponent(T component) {
				SharedPtrWrapper<Is_Shared_Ptr<T>::value>::Check<T>();

				CType type = AddComponentWrapper(component);
				MarkDirty();
			}


			~VObjectObj() {
				SAFE_RELEASE_OPTIX_OBJ(group);
				SAFE_RELEASE_OPTIX_OBJ(acc);
				SAFE_RELEASE_OPTIX_OBJ(instance);
			}
		};
		typedef shared_ptr<VObjectObj> PtrVObjectObj;


		class PrimeObjectManager {

			static unordered_map<VObjectObj* , shared_ptr<VObjectObj>> all_objects;

			static unordered_set<shared_ptr<VObjectObj>> need_rebind_objects;
			
			static optix::Group root;
			static optix::Acceleration acc;

		public:

			static optix::Group Root() { TryInit(); return root; }

			static void MarkDiry(VObjectObj* obj) {
				need_rebind_objects.insert(all_objects[obj]);
			}

			static bool RebindObjectComponents() {
				if (need_rebind_objects.empty()) return false;
				for each (auto obj in need_rebind_objects) {
					obj->ApplyRebind();
				}
				need_rebind_objects.clear();
				return true;
			}

			static void RegenerateGraph() {
				SAFE_RELEASE_OPTIX_OBJ(root);
				SAFE_RELEASE_OPTIX_OBJ(acc);

				auto context = OptixInstance::Instance().Context();
				root = context->createGroup();
				acc = context->createAcceleration("Trbvh");
				root->setAcceleration(acc);

				for each (auto obj in all_objects)
				{
					root->addChild(obj.second->GetPrimeObj());
				}
			}

			static void RemoveAll() {
				all_objects.clear();
				need_rebind_objects.clear();
				SAFE_RELEASE_OPTIX_OBJ(root);
				SAFE_RELEASE_OPTIX_OBJ(acc);
			}

			static void TryInit() {
				if (root == NULL) {
					auto context = OptixInstance::Instance().Context();
					root = context->createGroup();
					acc = context->createAcceleration("Trbvh");
					root->setAcceleration(acc);
				}
			}

			static PtrVObjectObj CreateNewObject();
		};
	}

	typedef prime::PtrVObjectObj VObject;

	class VObjectManager {
	public:
		static VObject CreateNewObject() { return prime::PrimeObjectManager::CreateNewObject(); };
		static void RemoveAll() { prime::PrimeObjectManager::RemoveAll(); }
	};
}