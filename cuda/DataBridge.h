#pragma once

#include <optixu/optixu_math_namespace.h>


struct Light
{                  
	int type;
    optix::float3 a;                                                          
    optix::float3 b, c;                                                          
    optix::float3 d;                                                          
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