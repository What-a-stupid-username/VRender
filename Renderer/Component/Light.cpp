#pragma once

#include "Light.hpp"

namespace VRender {
	namespace prime {
		vector<VLight> PrimeLightManager::lights;
		vector<VObject> PrimeLightManager::light_objects;
		unordered_set<int> PrimeLightManager::dirty_lights;
		Buffer PrimeLightManager::light_buffer;
	}
}