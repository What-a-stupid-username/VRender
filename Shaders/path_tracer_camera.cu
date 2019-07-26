#include "path_tracer.cuh"

//-----------------------------------------------------------------------------
//
//  Camera program -- main ray tracing loop
//
//-----------------------------------------------------------------------------

rtDeclareVariable(float3,			camera_position, , );
rtDeclareVariable(float3,			camera_up, , );
rtDeclareVariable(float3,			camera_forward, , );
rtDeclareVariable(float3,			camera_right, , );
rtDeclareVariable(float2,			camera_fov, , );
rtDeclareVariable(unsigned int,		camera_staticFrameNum, , );
rtDeclareVariable(unsigned int,		sqrt_num_samples, , ) = 1;
rtDeclareVariable(unsigned int,		cut_off_high_variance_result, , ) = 1;


rtBuffer<float4, 2>              V_TARGET0;
rtBuffer<float4, 2>              V_TARGET1;
rtBuffer<int, 2>				 ID_MASK;

inline float distance2(const float3& a, const float3& b) {
	float3 d = a - b;
	return dot(d, d);
}




RT_PROGRAM void dispatch()
{
	float3 last_color_result = make_float3(V_TARGET0[launch_index]);

    uint2 screen = make_uint2(V_TARGET0.size().x, V_TARGET0.size().y);
    float2 inv_screen = 1.0f/make_float2(screen) * 2.f;
    float2 pixel = (make_float2(launch_index)) * inv_screen - 1.f;

	int actrual_sqrt_sample_num = sqrt_num_samples;

    float2 jitter_scale = inv_screen / actrual_sqrt_sample_num;
    unsigned int samples_per_pixel = actrual_sqrt_sample_num*actrual_sqrt_sample_num;
	unsigned int samples_index = samples_per_pixel;

	unsigned int pixel_id = (screen.x * launch_index.y + launch_index.x) * (samples_per_pixel + 1);
	unsigned int seed = tea<16>(pixel_id, camera_staticFrameNum);
	float z = rnd(seed);
	float3 color_result = make_float3(0.0f);
	int id = -1;
    do 
    {
        //
        // Sample pixel using jittering
        //
        unsigned int x = samples_index % actrual_sqrt_sample_num;
        unsigned int y = samples_index / actrual_sqrt_sample_num;
        float2 jitter = make_float2(x - z, y - z);
        float2 d = pixel + jitter*jitter_scale;
		
		float3 ray_origin = camera_position;
        float3 ray_direction = normalize(d.x * camera_right * camera_fov.y + d.y * camera_up * camera_fov.x + camera_forward);

        // Initialze per-ray data
        PerRayData_pathtrace prd;
        prd.countEmitted = true;
        prd.seed = tea<16>(pixel_id + samples_index, camera_staticFrameNum);
		prd.depth = 0;
		prd.id = -1;
		prd.radiance = make_float3(0);

        Ray ray = make_Ray(ray_origin, ray_direction, pathtrace_common_ray_type, scene_epsilon, RT_DEFAULT_MAX);
        rtTrace(top_object, ray, prd);

		id = prd.id;
		float3 sample_result;
		if (cut_off_high_variance_result) {
			float sat = 50;
			sample_result = make_float3(min(prd.radiance.x, sat), min(prd.radiance.y, sat), min(prd.radiance.z, sat));
		}
		else {
			sample_result = prd.radiance;
		}
		color_result += sample_result;

    } while (--samples_index);

    //
    // Update the output buffer
    //
    color_result /= samples_per_pixel;

    if (camera_staticFrameNum > 1)
    {
		float a = 1.0f / (float)camera_staticFrameNum;
        V_TARGET0[launch_index] = make_float4( lerp(last_color_result, color_result, a), 1.0f );
    }
    else
    {
        V_TARGET0[launch_index] = make_float4(color_result, 1.0f);
    }
	ID_MASK[launch_index] = id;
}

//-----------------------------------------------------------------------------
//
//  Exception program
//
//-----------------------------------------------------------------------------

RT_PROGRAM void exception()
{
    V_TARGET0[launch_index] = make_float4(bad_color, 1.0f);
}

//-----------------------------------------------------------------------------
//
//  Miss program
//
//-----------------------------------------------------------------------------

RT_PROGRAM void miss()
{
    current_prd.radiance = bg_color;
}


