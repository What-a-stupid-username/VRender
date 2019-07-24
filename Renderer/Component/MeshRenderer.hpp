#pragma once

#include "BasicComponent.hpp"


namespace VRender {

	namespace prime {

		class RegisterComponent(VMeshRenderer) {
			friend class VComponent;
		protected:

			optix::GeometryInstance instance;

			VMaterial material = nullptr;


			VMeshRenderer();

		public:

			void Rebind(optix::GeometryInstance instance) { 
				instance->setMaterialCount(1);
				instance->setMaterial(0, material->material);
				this->instance = instance;
			};

			void ApplyPropertiesChanged() {
				instance->setMaterialCount(1);
				instance->setMaterial(0, material->material);
			};

			void SetMaterial(const VMaterial& material) {
				if (this->material != material) {
					this->material = material;
					MarkDirty();
				}
			}

			VMaterial GetMaterial() {
				return material;
			}

			~VMeshRenderer() {

			}
		};

	}

	PublicClass(VMeshRenderer);
}