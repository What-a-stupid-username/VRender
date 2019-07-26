#pragma 0 ClosestHit
#pragma 1 AnyHit

#include "path_tracer.cuh"



//-----------------------------------------------------------------------------
//
//  default_light closest-hit
//
//-----------------------------------------------------------------------------


rtDeclareVariable(int, light_id, , );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
RT_PROGRAM void default_light_ClosestHit() //ray-type = 0(normal_ray)
{
	float3 normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
	if (current_prd.depth == 0) current_prd.id = object_id;
	float3 emission_color = lights[light_id].emission;
	float k = dot(-ray.direction, normal);
	if (k  > 0)
		current_prd.radiance = current_prd.countEmitted ? emission_color * k : make_float3(0.f);
}

RT_PROGRAM void default_light_AnyHit() //ray-type = 1(shaodw_ray)
{
	rtIgnoreIntersection();
}