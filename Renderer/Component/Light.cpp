#pragma once

#include "Light.hpp"

namespace VRender {
	namespace prime {
		vector<shared_ptr<ParallelogramLight>> PrimeLightManager::parallelogram_lights;
		unordered_set<int> PrimeLightManager::dirty_lights;
		Buffer PrimeLightManager::light_buffer;
	}
}