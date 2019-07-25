#pragma 0 ClosestHit
#pragma 1 AnyHit

#include "path_tracer.cuh"



//-----------------------------------------------------------------------------
//
//  test
//
//-----------------------------------------------------------------------------

rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3,		texcoord,			attribute texcoord, );
rtDeclareVariable(optix::Ray,	ray,				rtCurrentRay, );
rtDeclareVariable(float,		t_hit,				rtIntersectionDistance, );

RT_PROGRAM void test_ClosestHit() //ray-type = 0(common_ray)
{
	float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
	float l = dot(world_geometric_normal, -ray.direction);
	if (texcoord.x < 0 && texcoord.y < 0) {
		current_prd.radiance = make_float3(l, l, l);
		return;
	}
	float3 color = texcoord.x == 0 ? make_float3(1, 0, 0) : (texcoord.x == 1 ? make_float3(0, 0, 1) : make_float3(0, 1, 0));
	current_prd.radiance = l * color;

	if (current_prd.depth == 0) current_prd.id = -10 * (1 + texcoord.x * 2) - (1 + texcoord.y * 2);
}


//-----------------------------------------------------------------------------
//
//  Shadow any-hit
//
//-----------------------------------------------------------------------------

rtDeclareVariable(PerRayData_pathtrace_shadow, current_prd_shadow, rtPayload, );

RT_PROGRAM void test_AnyHit() //ray-type = 1(shadow_ray)
{
	current_prd_shadow.inShadow = 0;
	rtTerminateRay();
}