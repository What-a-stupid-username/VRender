#include "MeshFilter.hpp"

namespace VRender {
	namespace prime {

		VMeshFilter::VMeshFilter() : VComponent() {
			auto& instance = OptixInstance::Instance();
			auto& context = instance.Context();

			geo_triangle = context->createGeometryTriangles();
			geo_triangle->setAttributeProgram(instance.AttributeProgram());
		}

		void VMeshFilter::SetMesh(const VMesh& mesh) {
			dirty = true;

			this->mesh = mesh;

			RTsize size = -1; mesh->v_index_buffer->getSize(size);
			if (size == -1) throw Exception("Error mesh!");
			geo_triangle->setPrimitiveCount(size);
			mesh->vert_buffer->getSize(size);
			geo_triangle->setVertices(size, mesh->vert_buffer, RT_FORMAT_FLOAT3);
			geo_triangle->setTriangleIndices(mesh->v_index_buffer, RT_FORMAT_UNSIGNED_INT3);
			geo_triangle->setBuildFlags(RTgeometrybuildflags::RT_GEOMETRY_BUILD_FLAG_NONE);

			geo_triangle["vertex_buffer"]->setBuffer(mesh->vert_buffer);
			geo_triangle["v_index_buffer"]->setBuffer(mesh->v_index_buffer);

			auto & instance = OptixInstance::Instance();

			if (mesh->normal_buffer != NULL) {
				geo_triangle["normal_buffer"]->setBuffer(mesh->normal_buffer);
				geo_triangle["n_index_buffer"]->setBuffer(mesh->n_index_buffer);
			}
			else {
				geo_triangle["normal_buffer"]->setBuffer(instance.float3_default);
				geo_triangle["n_index_buffer"]->setBuffer(instance.int3_default);
			}
			if (mesh->tex_buffer != NULL) {
				geo_triangle["texcoord_buffer"]->setBuffer(mesh->tex_buffer);
				geo_triangle["t_index_buffer"]->setBuffer(mesh->t_index_buffer);
			}
			else {
				geo_triangle["texcoord_buffer"]->setBuffer(instance.float2_default);
				geo_triangle["t_index_buffer"]->setBuffer(instance.int3_default);
			}
		}
	}
}