#ifndef TEA_H_
#define TEA_H_



#include <optixu/optixu_math_namespace.h>

template<unsigned int N>
static __host__ __device__ __inline__ unsigned int tea(unsigned int val0, unsigned int val1)
{
	unsigned int v0 = val0;
	unsigned int v1 = val1;
	unsigned int s0 = 0;

	for (unsigned int n = 0; n < N; n++)
	{
		s0 += 0x9e3779b9;
		v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
		v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
	}

	return v0;
}

// Generate random unsigned int in [0, 2^24)
static __host__ __device__ __inline__ unsigned int lcg(unsigned int &prev)
{
	const unsigned int LCG_A = 1664525u;
	const unsigned int LCG_C = 1013904223u;
	prev = (LCG_A * prev + LCG_C);
	return prev & 0x00FFFFFF;
}

static __host__ __device__ __inline__ unsigned int lcg2(unsigned int &prev)
{
	prev = (prev * 8121 + 28411) % 134456;
	return prev;
}

// Generate random float in [0, 1)
static __host__ __device__ __inline__ float rnd(unsigned int &prev)
{
	return ((float)lcg(prev) / (float)0x01000000);
}

static __host__ __device__ __inline__ unsigned int rot_seed(unsigned int seed, unsigned int frame)
{
	return seed ^ frame;
}




#ifndef PI
#define PI 3.1415926536
#endif

#ifndef Inv_PI
#define Inv_PI 1.0f / 3.1415926536
#endif

#ifndef Two_PI
#define Two_PI 2 * 3.1415926536
#endif

unsigned int ReverseBits32(unsigned int bits)
{
	bits = (bits << 16) | (bits >> 16);
	bits = ((bits & 0x00ff00ff) << 8) | ((bits & 0xff00ff00) >> 8);
	bits = ((bits & 0x0f0f0f0f) << 4) | ((bits & 0xf0f0f0f0) >> 4);
	bits = ((bits & 0x33333333) << 2) | ((bits & 0xcccccccc) >> 2);
	bits = ((bits & 0x55555555) << 1) | ((bits & 0xaaaaaaaa) >> 1);
	return bits;
}

uint2 SobolIndex(uint2 Base, int Index, int Bits = 10) {
	unsigned int SobolNumbers[20] = {
		0x8680u, 0x4c80u, 0xf240u, 0x9240u, 0x8220u, 0x0e20u, 0x4110u, 0x1610u, 0xa608u, 0x7608u,
		0x8a02u, 0x280au, 0xe204u, 0x9e04u, 0xa400u, 0x4682u, 0xe300u, 0xa74du, 0xb700u, 0x9817u
	};

	uint2 Result = Base;

	for (int b = 0; b < 10 && b < Bits; ++b) {
		Result.x ^= (Index & (1 << b)) ? SobolNumbers[2 * b] : 0;
		Result.y ^= (Index & (1 << b)) ? SobolNumbers[2 * b + 1] : 0;
	}
	return Result;
}

float HaltonSequence(unsigned int Index, float base = 3)
{
	float result = 0;
	float f = 1 / base;
	int i = Index;

	
	while (i > 0) {
		result += f * (i % (int)base);
		i = floor(i / base);
		f /= base;
	}
	return result;
}

float frac(float c) {
	return c - floor(c);
}

float2 Hammersley(unsigned int Index, unsigned int NumSamples)
{
	return make_float2((float)Index / (float)NumSamples, ReverseBits32(Index));
}

float2 Hammersley(unsigned int Index, unsigned int NumSamples, uint2 Random)
{
	float E1 = frac((float)Index / NumSamples + float(Random.x & 0xffff) / (1 << 16));
	float E2 = float(ReverseBits32(Index) ^ Random.y) * 2.3283064365386963e-10;
	return make_float2(E1, E2);
}



#endif // !TEA_H_