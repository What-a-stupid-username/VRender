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
rtDeclareVariable(float, diffuse_strength, , );
rtDeclareVariable(int, max_depth, , );


rtBuffer<float4, 2>              output_buffer;
rtBuffer<ParallelogramLight>     lights;

rtDeclareVariable(float3, bg_color, , );