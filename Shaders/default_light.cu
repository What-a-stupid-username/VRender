#pragma 0 ClosestHit
#pragma 1 AnyHit

#include "path_tracer.cuh"



//-----------------------------------------------------------------------------
//
//  default_light closest-hit
//
//-----------------------------------------------------------------------------


rtDeclareVariable(float3, emission_color, , );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
RT_PROGRAM void default_light_ClosestHit() //ray-type = 0(normal_ray)
{
	if (dot(ray.direction, geometric_normal) > 0)
		current_prd.radiance = current_prd.countEmitted ? emission_color : make_float3(0.f);
}

RT_PROGRAM void default_light_AnyHit() //ray-type = 1(shaodw_ray)
{
	rtIgnoreIntersection();
}