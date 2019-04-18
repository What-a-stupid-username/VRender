#pragma 0 ClosestHit
#pragma 1 AnyHit

#include "path_tracer.cuh"



//-----------------------------------------------------------------------------
//
//  default_light closest-hit
//
//-----------------------------------------------------------------------------


rtDeclareVariable(float3, emission_color, , );

RT_PROGRAM void default_light_ClosestHit() //ray-type = 0(normal_ray)
{
	current_prd.radiance = current_prd.countEmitted ? emission_color : make_float3(0.f);
}

RT_PROGRAM void default_light_AnyHit() //ray-type = 1(shaodw_ray)
{
	rtIgnoreIntersection();
}