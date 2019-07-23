#include "Object.hpp"

namespace VRender {

	namespace prime {

		static int objectID = 0;
		void* Create() {
			auto ptr = new VObjectObj(objectID);
			objectID++;
			return ptr;
		}


		unordered_map<int, shared_ptr<VObjectObj>> PrimeObjectManager::all_objects;
		unordered_set<shared_ptr<VObjectObj>> PrimeObjectManager::need_rebind_objects;
		optix::Group PrimeObjectManager::root;
		optix::Acceleration PrimeObjectManager::acc;

		PtrVObjectObj PrimeObjectManager::CreateNewObject()
		{
			TryInit();

			auto ptr = (VObjectObj*)Create();
			auto res = PtrVObjectObj(ptr);
			
			all_objects[ptr->id] = res;

			res->MarkDirty();

			root->addChild(res->GetPrimeObj());

			return res;
		}
		void VObjectObj::MarkDirty() { PrimeObjectManager::MarkDiry(this); }
	}
}
