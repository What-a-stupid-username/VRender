#include "MeshRenderer.hpp"

namespace VRender {
	namespace prime {

		VMeshRenderer::VMeshRenderer() : VComponent() {
			optiXmaterial = OptixInstance::Instance().Context()->createMaterial();
			material = VMaterialManager::Find("error");
			
			material->shader->ApplyShader(optiXmaterial);

			dirty = true;
		}
	}
}