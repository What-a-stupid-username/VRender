#include <optixu/optixu_math_namespace.h>
#include "DataBridge.h"
#include "random.h"

using namespace optix;

#define DEBUG
#ifdef DEBUG



rtBuffer<float4, 2>              output_buffer;
rtDeclareVariable(float3, eye, , );
rtDeclareVariable(float3, U, , );
rtDeclareVariable(float3, V, , );
rtDeclareVariable(float3, W, , );
void AddColor(float3 pos, float3 color) {
	float3 dir = pos - eye;

	float dw = dot(dir, W) / length(W);
	if (dw < 0) return;
	dir = dir / dw * length(W);
	float3 duv = dir - W;
	float2 uv = make_float2(dot(duv, U) / length(U) / length(U), dot(duv, V) / length(V) / length(V)) / 2 + make_float2(0.5);
	size_t2 bsize = output_buffer.size();
	int2 index = make_int2(uv * make_float2(bsize));

	if (index.x < 0 || index.y < 0 || index.x >= bsize.x || index.y >= bsize.y) return;

	atomicAdd(&output_buffer[make_uint2(index.x, index.y)].x, color.x);
	atomicAdd(&output_buffer[make_uint2(index.x, index.y)].y, color.y);
	atomicAdd(&output_buffer[make_uint2(index.x, index.y)].z, color.z);
}
void SetColor(float3 pos, float3 color) {
	float3 dir = pos - eye;

	float dw = dot(dir, W) / length(W);
	if (dw < 0) return;
	dir = dir / dw * length(W);
	float3 duv = dir - W;
	float2 uv = make_float2(dot(duv, U) / length(U) / length(U), dot(duv, V) / length(V) / length(V)) / 2 + make_float2(0.5);
	size_t2 bsize = output_buffer.size();
	int2 index = make_int2(uv * make_float2(bsize));

	if (index.x < 0 || index.y < 0 || index.x >= bsize.x || index.y >= bsize.y) return;

	output_buffer[make_uint2(index.x, index.y)] = make_float4(color,1);
}
#else

void SetColor(float3 pos, float3 color) {};
void AddColor(float3 pos, float3 color) {};

#endif // DEBUG


rtDeclareVariable(unsigned int, rnd_seed, , );

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );

rtDeclareVariable(unsigned int, photon_ray_type, , ); // type = 2


//-----------------------------
// scene data
//-----------------------------
rtDeclareVariable(float, scene_epsilon, , );
rtDeclareVariable(rtObject, top_object, , );

//-----------------------------
// light list
//-----------------------------
rtBuffer<ParallelogramLight>     lights;



//-----------------------------
// Photon list
//-----------------------------
rtBuffer<int, 1>              length_buffer;



rtBuffer<Photon, 1>              photon_buffer;

inline void append(Photon value) {
	int index = atomicAdd(&length_buffer[0], 1);
	if (index >= photon_buffer.size()) return;//Todo: delete it, it should be ensured by host code
	photon_buffer[index] = value;
}

inline int maxPhotonNum() {
	return photon_buffer.size();
}


//-----------------------------
// Per ray data
//-----------------------------
struct PerRayData_photon
{
	float3 emmit;
	unsigned int seed;
	unsigned int deep;
};

rtDeclareVariable(PerRayData_photon, current_prd, rtPayload, );


//-----------------------------
// entrypoint
//-----------------------------
RT_PROGRAM void emmit() {
	
	int lightNum = lights.size();
	if (lightNum == 0) return;
	int l_index = launch_index.x;
	float emmit_sum = 0;

	for (int light_index = 0; light_index < lightNum; light_index++)
	{
		ParallelogramLight light = lights[light_index];

		emmit_sum += max(max(light.emission.x, light.emission.y), light.emission.z);
	}

	int start_index = 0;
	for (int light_index = 0; light_index < lightNum; light_index++)
	{		
		//----------------
		// get task light
		//----------------
		ParallelogramLight light = lights[light_index];
		
		float emmit = max(max(light.emission.x, light.emission.y), light.emission.z);

		int end_index = start_index + emmit / emmit_sum * 1000000;

		if (l_index > end_index) {
			start_index = end_index;
			continue;
		}

		float photon_num_per_light = (end_index - start_index) / 1000;
		//----------------
		// emmit photon
		//----------------
		unsigned int seed = tea<16>(l_index + rnd_seed, rnd_seed);
		float z1 = rnd(seed);
		seed += rnd_seed;
		float z2 = rnd(seed);

		const float3 light_pos = light.corner + light.v1 * z1 + light.v2 * z2;

		z1 = rnd(seed);
		seed += rnd_seed;
		z2 = rnd(seed);

		float3 light_dir;
		cosine_sample_hemisphere(z1, z2, light_dir);
		optix::Onb onb(light.normal);
		onb.inverse_transform(light_dir);
		//light_dir = make_float3(0, -1, 0);

		Ray photon_ray = make_Ray(light_pos, light_dir, photon_ray_type, scene_epsilon, RT_DEFAULT_MAX);
		
		PerRayData_photon prd;
		prd.emmit = light.emission / photon_num_per_light;
		prd.seed = seed;
		prd.deep = 0;

		rtTrace(top_object, photon_ray, prd);
		//SetColor(light_pos, light_dir*-1);
		break;
	}
}





//-----------------------------
//  photon-surface intersection
//-----------------------------
rtDeclareVariable(float3, diffuse_color, , );
rtDeclareVariable(float, spec, , ) = 0;
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );


RT_PROGRAM void default_lit_photon_closest_hit()
{
	float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
	float3 ffnormal = faceforward(world_shading_normal, -ray.direction, world_geometric_normal);

	float3 hitpoint = ray.origin + t_hit * ray.direction;

	float russian_value = rnd(current_prd.seed);
	current_prd.seed += rnd_seed;

	float max_diffuse = max(max(diffuse_color.x, diffuse_color.y), diffuse_color.z);

	if (russian_value < max_diffuse) {
		float3 change_emmit = diffuse_color / max_diffuse;
		PerRayData_photon prd;
		prd.seed = current_prd.seed;
		prd.emmit = current_prd.emmit * change_emmit;
		prd.deep = current_prd.deep + 1;

		float3 p;
		cosine_sample_hemisphere(russian_value, rnd(current_prd.seed), p);
		optix::Onb onb(ffnormal);
		onb.inverse_transform(p);

		Ray next_ray = make_Ray(hitpoint, p, photon_ray_type, scene_epsilon, RT_DEFAULT_MAX);
		
		rtTrace(top_object, next_ray, prd);
	}

	//-----------------------------
	//  append photon to list
	//-----------------------------
	append(Photon(hitpoint, current_prd.emmit, ray.direction));
	AddColor(hitpoint, current_prd.emmit);
	//const float3 cs[] = { make_float3(100,0,0), make_float3(0,100,0), make_float3(0,0,100) };
	//SetColor(hitpoint, cs[min(current_prd.deep,2)]);
}

RT_PROGRAM void light_ignore_photon_hit() {
	rtIgnoreIntersection();
}


RT_PROGRAM void exception()
{

}

RT_PROGRAM void miss()
{
	
}