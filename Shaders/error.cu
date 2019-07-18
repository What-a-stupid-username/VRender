#pragma 0 ClosestHit
#pragma 1 AnyHit

#include "path_tracer.cuh"

//-----------------------------------------------------------------------------
//
//  error closest-hit
//
//-----------------------------------------------------------------------------

RT_PROGRAM void error_ClosestHit() //ray-type = 0(common_ray)
{
	current_prd.radiance = make_float3(1.0f, 0.0f, 1.0f);
}

//-----------------------------------------------------------------------------
//
//  Shadow any-hit
//
//-----------------------------------------------------------------------------

RT_PROGRAM void error_AnyHit() //ray-type = 1(shadow_ray)
{
	rtIgnoreIntersection();
}