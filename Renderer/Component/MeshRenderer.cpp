#include "MeshRenderer.hpp"

namespace VRender {
	namespace prime {

		VMeshRenderer::VMeshRenderer() : VComponent() {
			material = VMaterialManager::Find("error");
			
			MarkDirty();
		}
	}
}