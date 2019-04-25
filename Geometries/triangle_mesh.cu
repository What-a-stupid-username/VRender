#pragma bound mesh_bounds
#pragma intersect mesh_intersect_refine


#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include "intersection_refinement.h"

using namespace optix;

// This is to be plugged into an RTgeometry object to represent
// a triangle mesh with a vertex buffer of triangle soup (triangle list)
// with an interleaved position, normal, texturecoordinate layout.

rtBuffer<float3> vertex_buffer;
rtBuffer<float3> normal_buffer;
rtBuffer<float2> texcoord_buffer;
rtBuffer<int3>   v_index_buffer;
rtBuffer<int3>   n_index_buffer;
rtBuffer<int3>   t_index_buffer;
rtBuffer<int>    material_buffer;

rtDeclareVariable(float3, texcoord, attribute texcoord, );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );

rtDeclareVariable(float3, back_hit_point, attribute back_hit_point, );
rtDeclareVariable(float3, front_hit_point, attribute front_hit_point, );

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );


template<bool DO_REFINE>
static __device__
void meshIntersect(int primIdx)
{
	const int3 v_idx = v_index_buffer[primIdx];

	const float3 p0 = vertex_buffer[v_idx.x];
	const float3 p1 = vertex_buffer[v_idx.y];
	const float3 p2 = vertex_buffer[v_idx.z];

	// Intersect ray with triangle
	float3 n;
	float  t, beta, gamma;
	if (intersect_triangle(ray, p0, p1, p2, n, t, beta, gamma)) {

		if (rtPotentialIntersection(t)) {

			geometric_normal = normalize(n);
			if (normal_buffer.size() == 0) {
				shading_normal = geometric_normal;
			}
			else {
				const int3 n_idx = n_index_buffer[primIdx];
				float3 n0 = normal_buffer[n_idx.x];
				float3 n1 = normal_buffer[n_idx.y];
				float3 n2 = normal_buffer[n_idx.z];
				shading_normal = normalize(n1*beta + n2 * gamma + n0 * (1.0f - beta - gamma));
			}

			if (texcoord_buffer.size() == 0) {
				texcoord = make_float3(0.0f, 0.0f, 0.0f);
			}
			else {
				const int3 t_idx = t_index_buffer[primIdx];
				float2 t0 = texcoord_buffer[t_idx.x];
				float2 t1 = texcoord_buffer[t_idx.y];
				float2 t2 = texcoord_buffer[t_idx.z];
				texcoord = make_float3(t1*beta + t2 * gamma + t0 * (1.0f - beta - gamma));
			}

			if (DO_REFINE) {
				refine_and_offset_hitpoint(
					ray.origin + t * ray.direction,
					ray.direction,
					geometric_normal,
					p0,
					back_hit_point,
					front_hit_point);
			}
			int mat_index = 0;
			if (material_buffer.size() != 0)
				mat_index = material_buffer[primIdx];

			rtReportIntersection(mat_index);
		}
	}
}


RT_PROGRAM void mesh_intersect(int primIdx)
{
	meshIntersect<false>(primIdx);
}


RT_PROGRAM void mesh_intersect_refine(int primIdx)
{
	meshIntersect<true>(primIdx);
}


RT_PROGRAM void mesh_bounds(int primIdx, float result[6])
{
	const int3 v_idx = v_index_buffer[primIdx];

	const float3 v0 = vertex_buffer[v_idx.x];
	const float3 v1 = vertex_buffer[v_idx.y];
	const float3 v2 = vertex_buffer[v_idx.z];
	const float  area = length(cross(v1 - v0, v2 - v0));

	optix::Aabb* aabb = (optix::Aabb*)result;

	if (area > 0.0f && !isinf(area)) {
		aabb->m_min = fminf(fminf(v0, v1), v2);
		aabb->m_max = fmaxf(fmaxf(v0, v1), v2);
	}
	else {
		aabb->invalidate();
	}
}

