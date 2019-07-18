#include "Transform.hpp"

namespace VRender {
	namespace prime {

		static float identity[] = { 1,0,0,0,
									0,1,0,0,
									0,0,1,0,
									0,0,0,1 };

		VTransform::VTransform() : VComponent() {
			transform = OptixInstance::Instance().Context()->createTransform();
			transform->setMatrix(false, identity, identity);

			pos = rotate = make_float3(0);
			scale = make_float3(1);
		}
		

	}
}