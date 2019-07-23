#include "post_process.cuh"

//-----------------------------------------------------------------------------
//
//  dispatch post
//
//-----------------------------------------------------------------------------


rtDeclareVariable(int, selected_object_id, , ) = -2;
rtBuffer<int, 2>					ID_MASK;
rtBuffer<float4, 2>					TARGET;

inline float4 saturate(float4 c) {
	float4 res;
	res.x = c.x < 1 ? c.x : 1;
	res.y = c.y < 1 ? c.y : 1;
	res.z = c.z < 1 ? c.z : 1;
	res.w = c.w < 1 ? c.w : 1;
	return res;
}

RT_PROGRAM void dispatch()
{
	float4 res = V_TARGET0[launch_index];
	if (selected_object_id != -1 && ID_MASK[launch_index] == selected_object_id) {
		res = saturate(res) * 0.5 + make_float4(0.5, 0.5, 0, 0);
		res.w = 1;
	}
	TARGET[launch_index] = res;
}