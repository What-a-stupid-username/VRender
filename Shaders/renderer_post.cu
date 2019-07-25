#include "post_process.cuh"



struct PerRayData_pathtrace
{
	float3 radiance;
	unsigned int seed;
	int depth;
	int id;
	int countEmitted;
	float importance;
};
rtDeclareVariable(PerRayData_pathtrace, current_prd, rtPayload, );


//-----------------------------------------------------------------------------
//
//  dispatch post
//
//-----------------------------------------------------------------------------

rtDeclareVariable(rtObject, handle_object, , );

rtDeclareVariable(float3, camera_position, , );
rtDeclareVariable(float3, camera_up, , );
rtDeclareVariable(float3, camera_forward, , );
rtDeclareVariable(float3, camera_right, , );
rtDeclareVariable(float2, camera_fov, , );

rtDeclareVariable(int, selected_object_id, , ) = -2;
rtDeclareVariable(int, draw_handle, , ) = 0;
rtBuffer<int, 2>					ID_MASK;
rtBuffer<float4, 2>					TARGET;

inline float4 saturate(float4 c) {
	return make_float4(saturate(c.x), saturate(c.y), saturate(c.z), saturate(c.w));
}

RT_PROGRAM void dispatch()
{
	int id = -1;
	float3 color;
	if (draw_handle && selected_object_id != -1) {
		uint2 screen = make_uint2(V_TARGET0.size().x, V_TARGET0.size().y);
		float2 inv_screen = 1.0f / make_float2(screen) * 2.f;
		float2 pixel = (make_float2(launch_index)) * inv_screen - 1.f;

		float2 d = pixel;

		float3 ray_origin = camera_position;
		float3 ray_direction = normalize(d.x * camera_right * camera_fov.y + d.y * camera_up * camera_fov.x + camera_forward);

		// Initialze per-ray data
		PerRayData_pathtrace prd;
		prd.id = -1;
		prd.radiance = make_float3(0);

		Ray ray = make_Ray(ray_origin, ray_direction, 0, 0.1, RT_DEFAULT_MAX);
		rtTrace(handle_object, ray, prd);

		id = prd.id;
		color = prd.radiance;
	}

	float4 res = V_TARGET0[launch_index];
	if (selected_object_id != -1) {
		if (ID_MASK[launch_index] == selected_object_id) {
			if (launch_index.x <= 1 || launch_index.y <= 1 || launch_index.x >= ID_MASK.size().x - 2 || launch_index.y >= ID_MASK.size().y - 2) {
				res = make_float4(1, 1, 0, 1);
			}
			else {
				int id0 = ID_MASK[make_uint2(launch_index.x + 2, launch_index.y)];
				int id1 = ID_MASK[make_uint2(launch_index.x - 2, launch_index.y)];
				int id2 = ID_MASK[make_uint2(launch_index.x , launch_index.y + 2)];
				int id3 = ID_MASK[make_uint2(launch_index.x, launch_index.y - 2)];
				if ((id0 != selected_object_id && id0 >= -1) ||
					(id1 != selected_object_id && id1 >= -1) ||
					(id2 != selected_object_id && id2 >= -1) ||
					(id3 != selected_object_id && id3 >= -1)) {
					res = make_float4(1, 1, 0, 1);
				}
			}
		}
	}

	if (id != -1) {
		TARGET[launch_index] = make_float4(color, 1) * 0.8 + saturate(res) * 0.2;
		ID_MASK[launch_index] = id;
	}
	else {
		TARGET[launch_index] = res;
	}
}