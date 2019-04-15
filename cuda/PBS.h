#pragma once

#include <optixu/optixu_math_namespace.h>
#include "optixPathTracer.h"
#include "random.h"                                    

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

	specularTerm = max(0, specularTerm * nl);
	
	specularTerm *= any(specColor) ? 1.0 : 0.0;


	float3 color = specularTerm * lightSatu * FresnelTerm(specColor, lh);

	return color;
}


float3 PBS(const float2 seed, const SurfaceInfo IN, const float3 lightDir, const float3 lightSatu, const float3 viewDir) {
	float3 color;

	float oneMinusReflectivity;
	float3 baseColor, specColor;
	baseColor = DiffuseAndSpecularFromMetallic(IN.baseColor, IN.metallic, /*ref*/ specColor, /*ref*/ oneMinusReflectivity);

	float3 normal = IN.normal;
	float4 worldPos = IN.worldPos;

	color = BRDF(baseColor, specColor, IN.smoothness, normal, viewDir, lightDir, lightSatu);
	return color;
}


inline void uniform_sample_hemisphere(float2 E, float3& p) {
	float Phi = 2 * M_PI * E.x;
	float CosTheta = E.y;
	float SinTheta = sqrt(1 - CosTheta * CosTheta);

	p.x = SinTheta * cos(Phi);
	p.y = SinTheta * sin(Phi);
	p.z = CosTheta;
}