#pragma once



#include "../Support/CommonInclude.h"
#include "../Support/Basic.h"
#include "../Support/objLoader.h"

#include "../Manager/Manager.hpp"

#include <regex>
#include <sstream> 
#include <set>
#include <memory>



namespace VRender {

	namespace prime {

		class VObjectObj;

		#define RegisterComponent(x) x; typedef shared_ptr<x> Ptr##x; class x : public VComponent 

		class VComponent;
		typedef shared_ptr<VComponent> PtrVComponent;
		class VComponent {

			friend class VObjectObj;

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
				static T Create() {
					static_assert(std::is_base_of<VComponent, T::element_type>::value, "Only subclasses of VComponent can be created.");
					static_assert(!std::is_same<T::element_type, VComponent>::value, "VComponent can't be created.");

					auto ptr = T(new T::element_type());
					return ptr;
				}
			};

			template<>
			struct SharedPtrWrapper<false> {
				template<typename T>
				static T Create() {
					static_assert(std::is_base_of<VComponent, T>::value, "Only subclasses of VComponent can be created.");
					return T();
				}
			};
		#pragma endregion

		protected:
			
			bool dirty = true;

		public:
			template<typename T>
			static T Create() {
				return SharedPtrWrapper<Is_Shared_Ptr<T>::value>::Create<T>();
			};

			void MarkDirty();

			virtual void Rebind(optix::GeometryInstance) = 0;

			virtual void ApplyPropertiesChanged() = 0;

			~VComponent() {	}

		protected:
			VComponent() { MarkDirty(); };
		};

		class PrimeComponentManager {
			static set<VComponent*> dirt_comps;
		public:
			template<typename T>
			static void MarkDirty(T comp) {
				dirt_comps.insert(comp);
			}

			static void ApplyChanges() {
				for each (auto comp in dirt_comps)
				{
					comp->ApplyPropertiesChanged();
				}
				dirt_comps.clear();
			}
		};
	}
	
	#define PublicClass(x) typedef prime::Ptr##x x;
	
	class VComponents {
	public:
		template<typename T>
		static T Create() {
			return prime::VComponent::Create<T>();
		};
	};
}