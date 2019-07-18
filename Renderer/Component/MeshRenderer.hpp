#pragma once

#include "BasicComponent.hpp"


namespace VRender {

	namespace prime {

		class RegisterComponent(VMeshRenderer) {
			friend class VComponent;
		protected:

			optix::Material optiXmaterial;

			VMaterial material = nullptr;


			VMeshRenderer();

		public:

			void Rebind(optix::GeometryInstance instance) { 
				instance->setMaterialCount(1);
				instance->setMaterial(0, optiXmaterial);
			};

			void ApplyPropertiesChanged() {
				if (!dirty) return;

				optiXmaterial->setAnyHitProgram(0, nullptr);
				optiXmaterial->setAnyHitProgram(1, nullptr);
				optiXmaterial->setClosestHitProgram(0, nullptr);
				optiXmaterial->setClosestHitProgram(1, nullptr);

				material->shader->ApplyShader(optiXmaterial);
				
				int v_c = optiXmaterial->getVariableCount();
				for (int i = 0; i < v_c; i++)
				{
					optiXmaterial->removeVariable(optiXmaterial->getVariable(i));
				}

				for each (auto pair in material->properties)
				{
					if (pair.second.Type() == "string") {
						int k1 = pair.first.find('|');
						if (k1 != -1) {
							string special_type = pair.first.substr(0, k1);
							string name = pair.first.substr(k1 + 1, pair.first.length() - k1 - 1);
							if (special_type == "Texture") {
								int id = material->textures[*pair.second.GetData<string>()]->ID();
								optiXmaterial[name]->setUserData(sizeof(int), (void*)&id);
							}
						}
					}
					else {
						pair.second.SetProperty(optiXmaterial, pair.first);
					}
				}

				dirty = false;
			};

			void SetMaterial(const VMaterial& material) {
				this->material = material;
				dirty = true;
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