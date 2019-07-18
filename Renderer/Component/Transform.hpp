#pragma once

#include "BasicComponent.hpp"


namespace VRender {

	namespace prime {

		class RegisterComponent(VTransform) {
			friend class VComponent;
		protected:

			optix::Transform transform;

			float3 pos, rotate, scale;

			VTransform();

			void Rebind(optix::GeometryInstance instance) abandon;
		public:

			void Rebind(optix::GeometryGroup group) { transform->setChild(group); };

			optix::Transform GetPrimeObj() { return transform; }

			template<typename T>
			T* Position() {
				return (T*)& pos;
			}
			template<typename T>
			T* Rotation() {
				return (T*)& rotate;
			}
			template<typename T>
			T* Scale() {
				return (T*)& scale;
			}

			void ApplyPropertiesChanged() {
				Matrix4x4 mat;
				mat = Matrix4x4::scale(scale);

				mat = Matrix4x4::rotate(rotate.x / 180 * M_PI, make_float3(1, 0, 0)) * mat;
				mat = Matrix4x4::rotate(rotate.y / 180 * M_PI, make_float3(0, 1, 0)) * mat;
				mat = Matrix4x4::rotate(rotate.z / 180 * M_PI, make_float3(0, 0, 1)) * mat;

				mat = Matrix4x4::translate(pos) * mat;

				transform->setMatrix(false, mat.getData(), NULL);
			};

			~VTransform() {

			}

		};

	}

	PublicClass(VTransform);
}