#pragma 0 ClosestHit
#pragma 1 AnyHit

#include "path_tracer.cuh"



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

RT_PROGRAM void default_lit_ClosestHit() //ray-type = 0(common_ray)
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
				if (z1 < max_diffuse * transparent / (current_prd.depth + 2)) //͸�䲿��
				{
					float pd;
					float3 n;
					ImportanceSampleGGX(make_float2(z1, z2), IN.smoothness, n, pd); //����ë����
					onb.inverse_transform(n);

					if (refract(p, ray.direction, n, in_to_out ? 1.0f / refraction_index : refraction_index)) {
						prd.countEmitted = false;

						next_ray = make_Ray(hitpoint, p, common_ray_type, scene_epsilon, RT_DEFAULT_MAX);

						rtTrace(top_object, next_ray, prd);

						current_prd.radiance += prd.radiance * baseColor / max_diffuse * (current_prd.depth + 2);
					}
				}
				if (z2 < max_diffuse * (2 * in_to_out * transparent + 1 - in_to_out - transparent)) { //���䲿��
					cosine_sample_hemisphere(z1, z2, p);
					onb.inverse_transform(p);

					prd.countEmitted = false;

					next_ray = make_Ray(hitpoint, p, common_ray_type, scene_epsilon, RT_DEFAULT_MAX);

					rtTrace(top_object, next_ray, prd);
					current_prd.radiance += prd.radiance * baseColor / max_diffuse * diffuse_strength;
				}
			}
			if (z1 < 1.f / (current_prd.depth+3))
			{// ���䲿��
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

RT_PROGRAM void default_lit_AnyHit() //ray-type = 1(shadow_ray)
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