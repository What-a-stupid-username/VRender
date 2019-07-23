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
	int id;
	int countEmitted;
	float importance;
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


rtDeclareVariable(unsigned int, pathtrace_common_ray_type, , ) = 0;
rtDeclareVariable(unsigned int, pathtrace_shadow_ray_type, , ) = 1;


rtDeclareVariable(unsigned int, rnd_seed, , );
rtDeclareVariable(float, diffuse_strength, , ) = 1;
rtDeclareVariable(int, max_depth, , ) = 3;

rtDeclareVariable(int, object_id, , ) = -1;

rtBuffer<Light>     lights;

rtDeclareVariable(float3, bg_color, , );
rtDeclareVariable(float3, bad_color, , );