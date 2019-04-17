#include <optixu/optixu_math_namespace.h>
#include "DataBridge.h"
#include "random.h"

#include "PBS.h"

using namespace optix;

struct PerRayData_pathtrace
{
    float3 radiance;
    unsigned int seed;
    int depth;
    int countEmitted;
};

struct PerRayData_pathtrace_shadow
{
    float inShadow;
};

// Scene wide variables
rtDeclareVariable(float,         scene_epsilon, , );
rtDeclareVariable(rtObject,      top_object, , );
rtDeclareVariable(uint2,         launch_index, rtLaunchIndex, );

rtDeclareVariable(PerRayData_pathtrace, current_prd, rtPayload, );


rtDeclareVariable(unsigned int, common_ray_type, , );
rtDeclareVariable(unsigned int, pathtrace_shadow_ray_type, , );


rtDeclareVariable(unsigned int, rnd_seed, , );
rtDeclareVariable(float, diffuse_strength, , ) = 1;


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
rtDeclareVariable(unsigned int,  sqrt_num_samples, , );

rtBuffer<float4, 2>              output_buffer;
rtBuffer<ParallelogramLight>     lights;


RT_PROGRAM void pathtrace_camera()
{
    size_t2 screen = output_buffer.size();

    float2 inv_screen = 1.0f/make_float2(screen) * 2.f;
    float2 pixel = (make_float2(launch_index)) * inv_screen - 1.f;

    float2 jitter_scale = inv_screen / sqrt_num_samples;
    unsigned int samples_per_pixel = sqrt_num_samples*sqrt_num_samples;
    float3 result = make_float3(0.0f);

    unsigned int seed = tea<16>(screen.x*launch_index.y+launch_index.x + rnd_seed, frame_number);
    do 
    {
        //
        // Sample pixel using jittering
        //
        unsigned int x = samples_per_pixel%sqrt_num_samples;
        unsigned int y = samples_per_pixel/sqrt_num_samples;
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

        Ray ray = make_Ray(ray_origin, ray_direction, common_ray_type, scene_epsilon, RT_DEFAULT_MAX);
        rtTrace(top_object, ray, prd);

		float sat = 50;
		result += make_float3(min(prd.radiance.x,sat), min(prd.radiance.y, sat), min(prd.radiance.z, sat));

    } while (--samples_per_pixel);

    //
    // Update the output buffer
    //
    float3 pixel_color = result/(sqrt_num_samples*sqrt_num_samples);

	//output_buffer[launch_index] = make_float4(0,0,0,1);
    if (frame_number > 1)
    {
        float a = 1.0f / (float)frame_number;
        float3 old_color = make_float3(output_buffer[launch_index]);
        output_buffer[launch_index] = make_float4( lerp( old_color, pixel_color, a ), 1.0f );
    }
    else
    {
        output_buffer[launch_index] = make_float4(pixel_color, 1.0f);
    }
}


//-----------------------------------------------------------------------------
//
//  default_light closest-hit
//
//-----------------------------------------------------------------------------

rtDeclareVariable(float3,        emission_color, , );

RT_PROGRAM void default_light_closest_hit() //ray-type = 0(normal_ray)
{
    current_prd.radiance = current_prd.countEmitted ? emission_color : make_float3(0.f);
}


//-----------------------------------------------------------------------------
//
//  default_lit_ closest-hit
//
//-----------------------------------------------------------------------------

rtDeclareVariable(float3,		albedo, , );
rtDeclareVariable(float,		transparent, , ) = 0.f;
rtDeclareVariable(float,		metallic, , ) = 0.f;
rtDeclareVariable(float,		smoothness, , ) = 0.f;
rtDeclareVariable(float,		refraction_index, , ) = 1.5f;


rtDeclareVariable(float3,		geometric_normal,	attribute geometric_normal, );
rtDeclareVariable(float3,		shading_normal,		attribute shading_normal, );
rtDeclareVariable(float3,		texcoord,			attribute texcoord, );
rtDeclareVariable(optix::Ray,	ray,				rtCurrentRay, );
rtDeclareVariable(float,		t_hit,				rtIntersectionDistance, );

RT_PROGRAM void default_lit_closest_hit() //ray-type = 0(common_ray)
{
    float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
    float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
    float3 ffnormal = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );

	float3 hitpoint = ray.origin + t_hit * ray.direction;

	current_prd.seed += 197;
	float z1 = rnd(current_prd.seed);
	current_prd.seed += 197;
	float z2 = rnd(current_prd.seed);
	float3 baseColor;

	// initialize surface info
	SurfaceInfo IN;
	IN.baseColor = albedo;
	IN.transparent = transparent;
	IN.metallic = metallic;
	IN.smoothness = smoothness;
	IN.normal = ffnormal;

	current_prd.radiance = make_float3(0);

	{
		int in_to_out = dot(ray.direction, world_geometric_normal) > 0;

		float3 p;
		PerRayData_pathtrace prd;
		Ray next_ray;
		prd.depth = current_prd.depth + 1;
		prd.seed = current_prd.seed;
		prd.radiance = make_float3(0);

		float3 a;
		float b;
		baseColor = DiffuseAndSpecularFromMetallic(IN.baseColor, IN.metallic, a, b);

		if (current_prd.depth < 6) 
		{
			optix::Onb onb(ffnormal);
			{
				float max_diffuse = max(max(baseColor.x, baseColor.y), baseColor.z);
				if (z1 < max_diffuse * transparent / (current_prd.depth + 2)) //透射部分
				{
					float pd;
					float3 n;
					ImportanceSampleGGX(make_float2(z1, z2), IN.smoothness, n, pd); //修正毛玻璃
					onb.inverse_transform(n);

					if (refract(p, ray.direction, n, in_to_out ? 1.0f / refraction_index : refraction_index)) {
						prd.countEmitted = false;

						next_ray = make_Ray(hitpoint, p, common_ray_type, scene_epsilon, RT_DEFAULT_MAX);

						rtTrace(top_object, next_ray, prd);

						current_prd.radiance += prd.radiance * baseColor / max_diffuse * (current_prd.depth + 2);
					}
				}
				if (z2 < max_diffuse * (2 * in_to_out * transparent + 1 - in_to_out - transparent)) { //漫射部分
					cosine_sample_hemisphere(z1, z2, p);
					onb.inverse_transform(p);

					prd.countEmitted = false;

					next_ray = make_Ray(hitpoint, p, common_ray_type, scene_epsilon, RT_DEFAULT_MAX);

					rtTrace(top_object, next_ray, prd);
					current_prd.radiance += prd.radiance * baseColor / max_diffuse * diffuse_strength;
				}
			}
			if (z1 < 1.f / (current_prd.depth+3))
			{// 反射部分
				float pd;
				float3 n;
				//uniform_sample_hemisphere(z1, z2, n);
				if (z1 > 1 || z2 > 1 || z1 < 0 || z2 < 0) {
					current_prd.radiance = make_float3(10000, 0, 10000); return;
				}
				ImportanceSampleGGX(make_float2(z1, z2), IN.smoothness, n, pd);

				onb.inverse_transform(n);
				p = reflect(ray.direction, n);

				if (dot(p, ffnormal) > 0) {
					prd.countEmitted = false;

					next_ray = make_Ray(hitpoint, p, common_ray_type, scene_epsilon, RT_DEFAULT_MAX);

					rtTrace(top_object, next_ray, prd);

					current_prd.radiance += PBS(IN, p, prd.radiance, -ray.direction) * (current_prd.depth + 3) / pd;
				}
			}
		}
	}

	if (z1 > 1.f / (current_prd.depth + 1)) return;

	unsigned int num_lights = lights.size();
	float3 result = make_float3(0.0f);

	for (int i = 0; i < num_lights; ++i)
	{
		// Choose random point on light
		ParallelogramLight light = lights[i];
		const float3 light_pos = light.corner + light.v1 * z1 + light.v2 * z2;

		// Calculate properties of light sample (for area based pdf)
		const float  Ldist = length(light_pos - hitpoint);
		const float3 L = normalize(light_pos - hitpoint);
		const float  nDl = dot(ffnormal, L);
		const float  LnDl = -dot(light.normal, L);

		// cast shadow ray
		if (nDl > 0.0f && LnDl > 0.0f)
		{
			PerRayData_pathtrace_shadow shadow_prd;
			shadow_prd.inShadow = 1;
			// Note: bias both ends of the shadow ray, in case the light is also present as geometry in the scene.
			Ray shadow_ray = make_Ray(hitpoint, L, pathtrace_shadow_ray_type, scene_epsilon, Ldist - scene_epsilon);
			rtTrace(top_object, shadow_ray, shadow_prd);

			if (shadow_prd.inShadow != 0)
			{
				const float A = length(cross(light.v1, light.v2));
				// convert area based pdf to solid angle
				const float weight = LnDl * A / (M_PIf * Ldist * Ldist);
				float3 light_satu = light.emission * weight * shadow_prd.inShadow;
				current_prd.radiance += (PBS(IN, L, light_satu, -ray.direction) + nDl * LnDl * light_satu * baseColor) * (current_prd.depth + 1);
			}
		}
	}
}


//-----------------------------------------------------------------------------
//
//  Shadow any-hit
//
//-----------------------------------------------------------------------------

rtDeclareVariable(PerRayData_pathtrace_shadow, current_prd_shadow, rtPayload, );

RT_PROGRAM void default_lit_any_hit() //ray-type = 1(shadow_ray)
{
	if (transparent == 0) {
		current_prd_shadow.inShadow = 0;
		rtTerminateRay();
	}
	else {
		current_prd_shadow.inShadow *= transparent * 0.8;
		rtIgnoreIntersection();
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

rtDeclareVariable(float3, bg_color, , );

RT_PROGRAM void miss()
{
    current_prd.radiance = bg_color;
}


