#ifndef GPU_KDTREE_HELPER_H

#include "gpu_kdtree_basic.cuh";

namespace gpu_kdtree {

	struct maximumFloat3
	{
		__host__ __device__ float3 operator()(const float3 &lhs, const float3 &rhs) const {
			float3 r;
			r.x = lhs.x < rhs.x ? rhs.x : lhs.x;
			r.y = lhs.y < rhs.y ? rhs.y : lhs.y;
			r.z = lhs.z < rhs.z ? rhs.z : lhs.z;
			return r;
		}
	};
	struct minimumFloat3 {
		__host__ __device__ float3 operator()(const float3 &lhs, const float3 &rhs) const {
			float3 r;
			r.x = lhs.x > rhs.x ? rhs.x : lhs.x;
			r.y = lhs.y > rhs.y ? rhs.y : lhs.y;
			r.z = lhs.z > rhs.z ? rhs.z : lhs.z;
			return r;
		}
	};
	struct lessFloat3X {
		__host__ __device__ bool operator()(const float3 &lhs, const float3 &rhs) const {
			return lhs.x < rhs.x;
		}
	};
	struct lessFloat3Y {
		__host__ __device__ bool operator()(const float3 &lhs, const float3 &rhs) const {
			return lhs.y < rhs.y;
		}
	};
	struct lessFloat3Z {
		__host__ __device__ bool operator()(const float3 &lhs, const float3 &rhs) const {
			return lhs.z < rhs.z;
		}
	};

	struct BBox
	{
		float min[3];
		float max[3];
		BBox() {};
		BBox(const InputBuffers& input, int from = 0, int to = -1) {
			if (to < 0)to = input.size;
			if (to > input.size || from < 0 || from >= input.size || from >= to) {
				std::cout << "ERROR!";
			}
			for (int i = 0; i < 3; i++)
			{
				thrust::device_ptr<float> buffer_ptr(input[i]);
				max[i] = thrust::reduce(buffer_ptr + from, buffer_ptr + to, -999999999.f, thrust::maximum<float>());
				min[i] = thrust::reduce(buffer_ptr + from, buffer_ptr + to, 999999999.f, thrust::minimum<float>());
			}
		}

		BBox(thrust::device_ptr<float3> input, int size) {
			float3 f; f.x = -999999999; f.y = -999999999; f.z = -999999999;
			float3 max_p = thrust::reduce(input, input + size, f, maximumFloat3());
			max[0] = max_p.x; max[1] = max_p.y; max[2] = max_p.z;
			f.x = 999999999; f.y = 999999999; f.z = 999999999;
			float3 min_p = thrust::reduce(input, input + size, f, minimumFloat3());
			min[0] = min_p.x; min[1] = min_p.y; min[2] = min_p.z;
		}
	};

}


#endif

