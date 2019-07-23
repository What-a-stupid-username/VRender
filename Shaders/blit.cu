#include "post_process.cuh"

//-----------------------------------------------------------------------------
//
//  dispatch post
//
//-----------------------------------------------------------------------------

rtBuffer<float4, 2>					V_TARGET1;

RT_PROGRAM void dispatch()
{
	//size_t2 bsize = V_TARGET0.size();
	//float2 uv = make_float2(launch_index) / make_float2(bsize);
	//
	//if (launch_index.y < 256) return;

	//float4 color = mainTex ? tex2D<float4>(mainTex, uv.x, uv.y) : make_float4(0);
	//V_TARGET0[launch_index] = color;
	V_TARGET0[launch_index] = V_TARGET1[launch_index];
}