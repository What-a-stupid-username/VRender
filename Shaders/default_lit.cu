#pragma 0 ClosestHit
#pragma 1 AnyHit

#include "path_tracer.cuh"



//-----------------------------------------------------------------------------
//
//  default_lit closest-hit
//
//-----------------------------------------------------------------------------

rtDeclareVariable(float3,		albedo, , );
rtDeclareVariable(float,		transparent, , ) = 0.f;
rtDeclareVariable(float,		metallic, , ) = 0.f;
rtDeclareVariable(float,		smoothness, , ) = 0.f;
rtDeclareVariable(float,		refraction_index, , ) = 1.5f;


rtDeclareVariable(rtTextureId, baseColorTex, , ) = NULL;
rtDeclareVariable(rtTextureId, metallicTex, , ) = NULL;
rtDeclareVariable(rtTextureId, normalTex, , ) = NULL;
rtDeclareVariable(rtTextureId, roughnessTex, , ) = NULL;

rtDeclareVariable(float3,		geometric_normal,	attribute geometric_normal, );
rtDeclareVariable(float3,		shading_normal,		attribute shading_normal, );
rtDeclareVariable(float3,		texcoord,			attribute texcoord, );
rtDeclareVariable(optix::Ray,	ray,				rtCurrentRay, );
rtDeclareVariable(float,		t_hit,				rtIntersectionDistance, );

RT_PROGRAM void default_lit_ClosestHit() //ray-type = 0(common_ray)
{
    float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
    float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
    //float shading_normal = normalTex == 0 ? world_shading_normal : world_shading_normal * ;
    float3 ffnormal = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );

	float3 hitpoint = ray.origin + t_hit * ray.direction;

	current_prd.seed += 197;
	float z1 = rnd(current_prd.seed);
	current_prd.seed += 197;
	float z2 = rnd(current_prd.seed);
	float3 baseColor;

	// initialize surface info
	SurfaceInfo IN;
	IN.baseColor = baseColorTex == 0 ? albedo : make_float3(tex2D<float4>(baseColorTex, texcoord.x, texcoord.y));
	IN.transparent = transparent;
	IN.metallic = metallicTex == 0 ? metallic : tex2D<float4>(metallicTex, texcoord.x, texcoord.y).x;
	IN.smoothness = roughnessTex == 0 ? smoothness : 1 - tex2D<float4>(roughnessTex, texcoord.x, texcoord.y).x;
	IN.normal = ffnormal;
	//current_prd.radiance = make_float3(IN.metallic); return;
	
	int in_to_out = dot(ray.direction, world_geometric_normal) > 0;

	float3 a;
	float b;
	baseColor = DiffuseAndSpecularFromMetallic(IN.baseColor, IN.metallic, a, b);
	b = current_prd.depth + 1;
	float cut_off = 1 / b;

	
	if (current_prd.depth < max_depth)
	{
		if (z2 < cut_off)
		{
			optix::Onb onb(ffnormal);

			float3 p;
			PerRayData_pathtrace prd;
			Ray next_ray;
			prd.depth = current_prd.depth + 1;
			prd.seed = current_prd.seed;
			prd.importance = current_prd.importance;
			prd.radiance = make_float3(0);

			float max_diffuse = max(max(baseColor.x, baseColor.y), baseColor.z);

			float3 refr_diff_refl;
			refr_diff_refl.x = max_diffuse * transparent;
			refr_diff_refl.y = max_diffuse * (2 * in_to_out * transparent + 1 - in_to_out - transparent) * diffuse_strength;
			refr_diff_refl.z = (1 - max_diffuse);
			float sum_w = refr_diff_refl.x + refr_diff_refl.y + refr_diff_refl.z;
			refr_diff_refl /= sum_w;
			refr_diff_refl.y += refr_diff_refl.x;

			if (z1 < refr_diff_refl.x) { //透射部分
				float pd;
				float3 n;
				ImportanceSampleGGX(make_float2(z1, z2), IN.smoothness, n, pd); //修正毛玻璃
				onb.inverse_transform(n);

				if (refract(p, ray.direction, n, in_to_out ? 1.0f / refraction_index : refraction_index)) {
					prd.countEmitted = false;

					next_ray = make_Ray(hitpoint, p, pathtrace_common_ray_type, scene_epsilon, RT_DEFAULT_MAX);

					rtTrace(top_object, next_ray, prd);

					current_prd.radiance = prd.radiance * baseColor / max_diffuse;
				}
			}
			else if (z1 < refr_diff_refl.y) { //漫射部分
				cosine_sample_hemisphere(z1, z2, p);
				onb.inverse_transform(p);

				prd.countEmitted = false;

				next_ray = make_Ray(hitpoint, p, pathtrace_common_ray_type, scene_epsilon, RT_DEFAULT_MAX);

				rtTrace(top_object, next_ray, prd);
				current_prd.radiance = PBS<0>(IN, p, prd.radiance, -ray.direction) / max_diffuse;
			}
			else { // 反射部分
				float pd;
				float3 n;
				ImportanceSampleGGX(make_float2(z1, z2), IN.smoothness, n, pd);

				onb.inverse_transform(n);
				p = reflect(ray.direction, n);

				if (dot(p, ffnormal) > 0) {
					prd.countEmitted = false;

					next_ray = make_Ray(hitpoint, p, pathtrace_common_ray_type, scene_epsilon, RT_DEFAULT_MAX);

					rtTrace(top_object, next_ray, prd);

					current_prd.radiance = PBS<1>(IN, p, prd.radiance, -ray.direction) / pd / (1 - max_diffuse);
				}
			}
			current_prd.radiance *= sum_w * b;
		}
	}
	
	
	unsigned int num_lights = lights.size();
	float3 result = make_float3(0.0f);

	for (int i = 0; i < num_lights; ++i) //当前把所有类型的光都当作片光源
	{
		// Choose random point on light
		Light light = lights[i];
		const float3 light_pos = light.a + light.b * z1 + light.c * z2;

		// Calculate properties of light sample (for area based pdf)
		const float  Ldist = length(light_pos - hitpoint);
		const float3 L = normalize(light_pos - hitpoint);
		const float  nDl = dot(ffnormal, L);
		const float  LnDl = -dot(light.d, L);

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
				const float A = length(cross(light.b, light.c));
				// convert area based pdf to solid angle
				const float weight = LnDl * A / (M_PIf * Ldist * Ldist);
				float3 light_satu = light.emission * weight * shadow_prd.inShadow;
				current_prd.radiance += ((PBS<2>(IN, L, light_satu, -ray.direction) + nDl * LnDl * light_satu * baseColor));
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