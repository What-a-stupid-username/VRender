#include <optix.h>
#include <optixu/optixu_math_namespace.h>

rtDeclareVariable( float3, shading_normal,   attribute shading_normal, );
rtDeclareVariable( float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable( float3, texcoord,         attribute texcoord, );
rtDeclareVariable( float2, barycentrics,     attribute barycentrics, );

rtBuffer<float3> vertex_buffer;
rtBuffer<float3> normal_buffer;
rtBuffer<float2> texcoord_buffer;
rtBuffer<int3>   v_index_buffer;
rtBuffer<int3>   n_index_buffer;
rtBuffer<int3>   t_index_buffer;
//rtBuffer<int>    material_buffer;

RT_PROGRAM void triangle_attributes()
{
    const int primID = rtGetPrimitiveIndex();
    const int3   v_idx = v_index_buffer[primID];
    int3   n_idx = v_idx, t_idx = v_idx;

    if (n_index_buffer.size() != 0){
        n_idx = n_index_buffer[primID];
    }

    if (t_index_buffer.size() != 0){
        t_idx = t_index_buffer[primID];
    }

    const float3 v0    = vertex_buffer[v_idx.x];
    const float3 v1    = vertex_buffer[v_idx.y];
    const float3 v2    = vertex_buffer[v_idx.z];
    const float3 Ng    = optix::cross( v1 - v0, v2 - v0 );

    geometric_normal = optix::normalize( Ng );

    barycentrics = rtGetTriangleBarycentrics();
    texcoord = make_float3( barycentrics.x, barycentrics.y, 0.0f );

    if( normal_buffer.size() == 0 )
    {
        shading_normal = geometric_normal;
    }
    else
    {
        shading_normal = normal_buffer[n_idx.y] * barycentrics.x + normal_buffer[n_idx.z] * barycentrics.y
            + normal_buffer[n_idx.x] * ( 1.0f-barycentrics.x-barycentrics.y );
    }

    if( texcoord_buffer.size() == 0 )
    {
        texcoord = make_float3( 0.0f, 0.0f, 0.0f );
    }
    else
    {
        const float2 t0 = texcoord_buffer[t_idx.x];
        const float2 t1 = texcoord_buffer[t_idx.y];
        const float2 t2 = texcoord_buffer[t_idx.z];
        texcoord = make_float3( t1*barycentrics.x + t2*barycentrics.y + t0*(1.0f-barycentrics.x-barycentrics.y) );
    }
}