#include <optixu/optixu_math_namespace.h>

using namespace optix;

rtDeclareVariable(uint2,			launch_index, rtLaunchIndex, );

rtDeclareVariable(rtTextureId,		mainTex, , ) = NULL;
rtBuffer<float4, 2>					V_TARGET0;