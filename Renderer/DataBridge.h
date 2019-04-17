#pragma once

#include <optixu/optixu_math_namespace.h>


struct ParallelogramLight                                                        
{                                                                                
    optix::float3 corner;                                                          
    optix::float3 v1, v2;                                                          
    optix::float3 normal;                                                          
    optix::float3 emission;                                                        
};                                                                               

struct Photon
{
	optix::float3 pos;
	optix::float3 color;
	optix::float3 dir;
	Photon(optix::float3 p, optix::float3 c, optix::float3 d) {
		pos = p;
		color = c;
		dir = d;
	}
};