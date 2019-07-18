#pragma once

#include "BasicComponent.hpp"


namespace VRender {

	namespace prime {

		class RegisterComponent(VMeshFilter) {
			friend class VComponent;
		protected:

			optix::GeometryTriangles geo_triangle;

			VMesh mesh;

			VMeshFilter();


		public:
			void SetMesh(const VMesh & mesh);

			void Rebind(optix::GeometryInstance instance) { instance->setGeometryTriangles(geo_triangle); };

			void ApplyPropertiesChanged() {};

			~VMeshFilter() {
				SAFE_RELEASE_OPTIX_OBJ(geo_triangle);
			}
		};

	}

	PublicClass(VMeshFilter);
}