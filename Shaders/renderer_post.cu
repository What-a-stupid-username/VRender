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
	if (selected_object_id != -1) {
		size_t2 bsize = ID_MASK.size();
		if (ID_MASK[launch_index] == selected_object_id) {
			if (launch_index.x + 2 >= bsize.y || launch_index.y + 2 >= bsize.y || launch_index.x <= 1 || launch_index.y <= 1) {
				res = make_float4(1, 1, 0, 1);
			}
			else if (ID_MASK[make_uint2(launch_index.x + 2, launch_index.y)] != selected_object_id ||
				ID_MASK[make_uint2(launch_index.x, launch_index.y + 2)] != selected_object_id ||
				ID_MASK[make_uint2(launch_index.x - 2, launch_index.y)] != selected_object_id ||
				ID_MASK[make_uint2(launch_index.x, launch_index.y - 2)] != selected_object_id) {
				res = make_float4(1, 1, 0, 1);
			}
			else {
				res = res * 0.9 +  make_float4(0.1, 0.1, 0, 0);
				res.w = 1;
			}
		}
	}
	TARGET[launch_index] = res;
}