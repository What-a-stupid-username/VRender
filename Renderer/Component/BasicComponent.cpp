#include "BasicComponent.hpp"

namespace VRender {

	namespace prime {

		set<VComponent*> PrimeComponentManager::dirt_comps;

		void VComponent::MarkDirty() { PrimeComponentManager::MarkDirty(this); }
	}

}