#include "path_tracer.cuh"

//-----------------------------------------------------------------------------
//
//  Camera program -- main ray tracing loop
//
//-----------------------------------------------------------------------------

rtDeclareVariable(float3,        eye, , );
rtDeclareVariable(float3,        U, , );
rtDeclareVariable(float3,        V, , );
rtDeclareVariable(float3,        W, , );
rtDeclareVariable(float3,        bad_color, , );
rtDeclareVariable(unsigned int,  frame_number, , );
rtDeclareVariable(unsigned int, sqrt_num_samples, , );
rtDeclareVariable(unsigned int, cut_off_high_variance_result, , );


rtBuffer<float4, 2>              output_buffer;
rtBuffer<float4, 2>              helper_buffer;


inline float distance2(const float3& a, const float3& b) {
	float3 d = a - b;
	return dot(d, d);
}




RT_PROGRAM void path_tracer_camera()
{
	float3 last_color_result = make_float3(output_buffer[launch_index]);
	float last_varient = frame_number > 100 ? helper_buffer[launch_index].x : 1;

    uint2 screen = make_uint2(output_buffer.size().x, output_buffer.size().y);
    float2 inv_screen = 1.0f/make_float2(screen) * 2.f;
    float2 pixel = (make_float2(launch_index)) * inv_screen - 1.f;

	unsigned int seed = tea<16>(screen.x*launch_index.y + launch_index.x + rnd_seed, frame_number);
	float z = rnd(seed);

	int actrual_sqrt_sample_num = sqrt_num_samples;

    float2 jitter_scale = inv_screen / actrual_sqrt_sample_num;
    unsigned int samples_per_pixel = actrual_sqrt_sample_num*actrual_sqrt_sample_num;
	unsigned int samples_index = samples_per_pixel;

	float3 color_result = make_float3(0.0f);
	float varient_result = 0;
    do 
    {
        //
        // Sample pixel using jittering
        //
        unsigned int x = samples_index % actrual_sqrt_sample_num;
        unsigned int y = samples_index / actrual_sqrt_sample_num;
        float2 jitter = make_float2(x-rnd(seed), y-rnd(seed));
        float2 d = pixel + jitter*jitter_scale;
		
		float3 ray_origin = eye/* + eye_jit*/;
        float3 ray_direction = normalize(d.x*U + d.y*V + W);

        // Initialze per-ray data
        PerRayData_pathtrace prd;
        prd.countEmitted = true;
        prd.seed = seed;
		prd.depth = 0;
		prd.radiance = make_float3(0);
		prd.importance = last_varient;

        Ray ray = make_Ray(ray_origin, ray_direction, common_ray_type, scene_epsilon, RT_DEFAULT_MAX);
        rtTrace(top_object, ray, prd);


		float3 sample_result;
		if (cut_off_high_variance_result) {
			float sat = 50;
			sample_result = make_float3(min(prd.radiance.x, sat), min(prd.radiance.y, sat), min(prd.radiance.z, sat));
		}
		else {
			sample_result = prd.radiance;
		}
		color_result += sample_result;
		varient_result += distance2(sample_result, last_color_result);

    } while (--samples_index);
    //
    // Update the output buffer
    //
	varient_result /= samples_per_pixel * max(dot(last_color_result, last_color_result),0.01);
	varient_result = min(max(varient_result, 0.4f), 1.f);
    color_result /= samples_per_pixel;

	//output_buffer[launch_index] = make_float4(0,0,0,1);
    if (frame_number > 1)
    {
		float a = 1.0f / (float)frame_number;
        output_buffer[launch_index] = make_float4( lerp(last_color_result, color_result, a), 1.0f );
    }
    else
    {
        output_buffer[launch_index] = make_float4(color_result, 1.0f);
    }

	if (frame_number < 100) {
		if (frame_number > 1)
		{
			float a = 1.0f / (float)frame_number;
			helper_buffer[launch_index] = make_float4(lerp(helper_buffer[launch_index].x, varient_result, a), 0, 0, 0);
		}
		else
		{
			helper_buffer[launch_index] = make_float4(varient_result, 0, 0, 0);
		}
		//float4 vv = helper_buffer[min(max(launch_index + make_uint2(0, 1), make_uint2(0)), screen - make_uint2(1))] * 0.025 +
		//	helper_buffer[min(max(launch_index + make_uint2(0, -1), make_uint2(0)), screen - make_uint2(1))] * 0.025 +
		//	helper_buffer[min(max(launch_index + make_uint2(1, 0), make_uint2(0)), screen - make_uint2(1))] * 0.025 +
		//	helper_buffer[min(max(launch_index + make_uint2(-1, 0), make_uint2(0)), screen - make_uint2(1))] * 0.025 +
		//	helper_buffer[launch_index] * 0.9;
		//helper_buffer[launch_index] = vv;
	}
}

//-----------------------------------------------------------------------------
//
//  Exception program
//
//-----------------------------------------------------------------------------

RT_PROGRAM void exception()
{
    output_buffer[launch_index] = make_float4(bad_color, 1.0f);
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


