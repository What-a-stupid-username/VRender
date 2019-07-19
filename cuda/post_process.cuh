#include <optixu/optixu_math_namespace.h>

using namespace optix;

rtDeclareVariable(uint2,			launch_index, rtLaunchIndex, );

rtDeclareVariable(rtTextureId,		mainTex, , ) = NULL;
rtBuffer<float4, 2>					V_TARGET0;


//-----------------------------------------------------------------------------
//
//  Exception program
//
//-----------------------------------------------------------------------------

RT_PROGRAM void exception()
{
	make_float3(1, 0, 1);
}

//-----------------------------------------------------------------------------
//
//  Miss program
//
//-----------------------------------------------------------------------------
RT_PROGRAM void miss() { }