#include "Object.hpp"

namespace VRender {

	namespace prime {

		void* Create() {
			return new VObjectObj();
		}

		unordered_map<VObjectObj*, shared_ptr<VObjectObj>> PrimeObjectManager::all_objects;
		unordered_set<shared_ptr<VObjectObj>> PrimeObjectManager::need_rebind_objects;
		optix::Group PrimeObjectManager::root;
		optix::Acceleration PrimeObjectManager::acc;

		PtrVObjectObj PrimeObjectManager::CreateNewObject()
		{
			TryInit();

			auto ptr = (VObjectObj*)Create();
			auto res = PtrVObjectObj(ptr);
			
			all_objects[ptr] = res;

			res->MarkDirty();

			root->addChild(res->GetPrimeObj());

			return res;
		}
		void VObjectObj::MarkDirty() { PrimeObjectManager::MarkDiry(this); }
	}
}
