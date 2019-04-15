#ifndef PBS_H_
#define PBS_H_



#include <optixu/optixu_math_namespace.h>
#include "optixPathTracer.h"
#include "random.h"   

using namespace optix;

//-----------------------------------------------------------------------------
//
//  Surface Info
//
//-----------------------------------------------------------------------------

struct SurfaceInfo {
	float3 baseColor;
	float transparent;
	float metallic;
	float smoothness;
	float3 normal;
};

//-----------------------------------------------------------------------------
//
//  Helper funcs
//
//-----------------------------------------------------------------------------

#define M_E        2.71828182845904523536   // e
#define M_LOG2E    1.44269504088896340736   // log2(e)
#define M_LOG10E   0.434294481903251827651  // log10(e)
#define M_LN2      0.693147180559945309417  // ln(2)
#define M_LN10     2.30258509299404568402   // ln(10)
#define M_PI       3.14159265358979323846   // pi
#define M_PI_2     1.57079632679489661923   // pi/2
#define M_PI_4     0.785398163397448309616  // pi/4
#define M_1_PI     0.318309886183790671538  // 1/pi
#define M_2_PI     0.636619772367581343076  // 2/pi
#define M_2_SQRTPI 1.12837916709551257390   // 2/sqrt(pi)
#define M_SQRT2    1.41421356237309504880   // sqrt(2)
#define M_SQRT1_2  0.707106781186547524401  // 1/sqrt(2)

inline float Pow4(float x)
{
	return x * x*x*x;
}

inline float2 Pow4(float2 x)
{
	return x * x*x*x;
}

inline float3 Pow4(float3 x)
{
	return x * x*x*x;
}

inline float4 Pow4(float4 x)
{
	return x * x*x*x;
}

inline float Pow5(float x)
{
	return x * x * x*x * x;
}

inline float2 Pow5(float2 x)
{
	return x * x * x*x * x;
}

inline float3 Pow5(float3 x)
{
	return x * x * x*x * x;
}

inline float4 Pow5(float4 x)
{
	return x * x * x*x * x;
}


inline float saturate(float in) {
	return min(max(in, 0.0f), 1.0f);
}

inline bool any(float3 in) {
	return (in.x * in.y * in.z) != 0;
}

inline float OneMinusReflectivityFromMetallic(float metallic)
{
	float oneMinusDielectricSpec = 1.0 - 0.220916301;
	return oneMinusDielectricSpec - metallic * oneMinusDielectricSpec;
}


inline float3 DiffuseAndSpecularFromMetallic(float3 albedo, float metallic, float3& specColor, float& oneMinusReflectivity)
{
	specColor = lerp(make_float3(0.220916301, 0.220916301, 0.220916301), albedo, metallic);
	oneMinusReflectivity = OneMinusReflectivityFromMetallic(metallic);
	return albedo * oneMinusReflectivity;
}

inline float SmoothnessToPerceptualRoughness(float smoothness)
{
	return (1 - smoothness);
}

float DisneyDiffuse(float NdotV, float NdotL, float LdotH, float perceptualRoughness)
{
	float fd90 = 0.5 + 2 * LdotH * LdotH * perceptualRoughness;
	// Two schlick fresnel term
	float lightScatter = (1 + (fd90 - 1) * Pow5(1 - NdotL));
	float viewScatter = (1 + (fd90 - 1) * Pow5(1 - NdotV));

	return lightScatter * viewScatter;
}

inline float PerceptualRoughnessToRoughness(float perceptualRoughness)
{
	return perceptualRoughness * perceptualRoughness;
}

inline float SmithJointGGXVisibilityTerm(float NdotL, float NdotV, float roughness)
{
	// Approximation of the above formulation (simplify the sqrt, not mathematically correct but close enough)
	float a = roughness;
	float lambdaV = NdotL * (NdotV * (1 - a) + a);
	float lambdaL = NdotV * (NdotL * (1 - a) + a);

#if defined(SHADER_API_SWITCH)
	return 0.5f / (lambdaV + lambdaL + 1e-4f); // work-around against hlslcc rounding error
#else
	return 0.5f / (lambdaV + lambdaL + 1e-5f);
#endif
}

inline float GGXTerm(float NdotH, float roughness)
{
	float a2 = roughness * roughness;
	float d = (NdotH * a2 - NdotH) * NdotH + 1.0f; // 2 mad
	return M_1_PI * a2 / (d * d + 1e-7f); // This function is not intended to be running on Mobile,
										  // therefore epsilon is smaller than what can be represented by float
}

inline float3 FresnelTerm(float3 F0, float cosA)
{
	float t = Pow5(1 - cosA);   // ala Schlick interpoliation
	return F0 + (1 - F0) * t;
}

inline float3 FresnelLerp(float3 F0, float3 F90, float cosA)
{
	float t = Pow5(1 - cosA);   // ala Schlick interpoliation
	return lerp(F0, F90, t);
}


float3 BRDF(const float3 diffColor, const float3 specColor, const float smoothness,
	float3 normal, const float3 viewDir, const float3 lightDir,
	const float3 lightSatu) {
	float perceptualRoughness = SmoothnessToPerceptualRoughness(smoothness);
	float3 floatDir = normalize(lightDir + viewDir);

	float shiftAmount = dot(normal, viewDir);
	normal = shiftAmount < 0.0f ? normal + viewDir * (-shiftAmount + 1e-5f) : normal;

	float nv = saturate(dot(normal, viewDir));

	float nl = saturate(dot(normal, lightDir));
	float nh = saturate(dot(normal, floatDir));

	float lv = saturate(dot(lightDir, viewDir));
	float lh = saturate(dot(lightDir, floatDir));

	float diffuseTerm = DisneyDiffuse(nv, nl, lh, perceptualRoughness) * nl;

	float roughness = PerceptualRoughnessToRoughness(perceptualRoughness);

	roughness = max(roughness, 0.002);
	float V = SmithJointGGXVisibilityTerm(nl, nv, roughness);
	float D = GGXTerm(nh, roughness);

	float specularTerm = V * D * M_PI;

	specularTerm = max(0.0f, specularTerm * nl);

	specularTerm *= any(specColor) ? 1.0 : 0.0;


	float3 color = specularTerm * lightSatu * FresnelTerm(specColor, lh);

	return color;
}


float3 PBS(const SurfaceInfo IN, const float3 lightDir, const float3 lightSatu, const float3 viewDir) {
	float3 color;

	float oneMinusReflectivity;
	float3 baseColor, specColor;
	baseColor = DiffuseAndSpecularFromMetallic(IN.baseColor, IN.metallic, /*ref*/ specColor, /*ref*/ oneMinusReflectivity);

	float3 normal = IN.normal;

	color = BRDF(baseColor, specColor, IN.smoothness, normal, viewDir, lightDir, lightSatu);
	return color;
}


inline void uniform_sample_hemisphere(float x, float y, float3& p) {
	float Phi = 2 * M_PI * x;
	float CosTheta = y;
	float SinTheta = sqrt(1 - CosTheta * CosTheta);

	p.x = SinTheta * cos(Phi);
	p.y = SinTheta * sin(Phi);
	p.z = CosTheta;
}

inline float NDF(const float3& h, float roughness) {
	float alpha = roughness * roughness;
	float alpha2 = alpha * alpha;
	float NoH = h.z;
	float k = NoH * NoH*(alpha2 - 1) + 1;
	return alpha2 / (M_PI * k * k);
}

void sample_GGX(float2 E, const float smoothness, float3 & n, float & pd) {

	float perceptualRoughness = SmoothnessToPerceptualRoughness(smoothness);
	float roughness = PerceptualRoughnessToRoughness(perceptualRoughness);
	// 根据 NDF 采样 h
	float Xi1 = E.x;
	float Xi2 = E.y;
	float alpha = roughness * roughness;
	float cosTheta2 = max((1 - Xi1) / (Xi1*(alpha*alpha - 1) + 1),0.f);
	float cosTheta = sqrt(cosTheta2);
	float sinTheta = sqrt(max(1 - cosTheta2, 0.f));
	float phi = 2 * M_PI * Xi2;

	n = make_float3(sinTheta*cos(phi), sinTheta*sin(phi), cosTheta);
	pd = NDF(n, roughness) / 4.0f;
}


#endif // !PBS_H_