#include "post_process.cuh"

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
rtBuffer<int, 2>					ID_MASK;
rtBuffer<float4, 2>					TARGET;


RT_PROGRAM void dispatch()
{
	int id = -1;
	float3 color;
	if (selected_object_id != -1) {
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

		Ray ray = make_Ray(ray_origin, ray_direction, 0, scene_epsilon, RT_DEFAULT_MAX);
		rtTrace(handle_object, ray, prd);

		id = prd.id;
		color = prd.radiance;
	}

	float4 res = V_TARGET0[launch_index];
	if (ID_MASK[launch_index] == selected_object_id) {
		res = saturate(res) * 0.5 + make_float4(0.5, 0.5, 0, 0);
		res.w = 1;
	}
	TARGET[launch_index] = res;
}