#ifndef GPU_KDTREE_BASIC_H
#define GPU_KDTREE_BASIC_H

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <iostream>
#include <thrust/functional.h>

namespace gpu_kdtree {

	//typedef float T;
	template<typename T>
	class CUDABuffer {
		T* ptr;
		size_t size;
		CUDABuffer(CUDABuffer& c) {}
	public:
		CUDABuffer(size_t size) {
			cudaMalloc((void**)&ptr, size * sizeof(T));
			this->size = size;
		}
		~CUDABuffer() {
			cudaFree(ptr);
		}
		operator T*() {
			return ptr;
		}
		T* Value() {
			return ptr;
		}
		size_t Size() {
			return size;
		}
		void SetValue(T* value) {
			cudaMemcpy(ptr, value, size * sizeof(T), cudaMemcpyKind::cudaMemcpyHostToDevice);
		}
		void GetValue(T* value) {
			cudaMemcpy(value, ptr, size * sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToHost);
		}
	};

	struct InputBuffers
	{
		size_t size;
		float* posX, *posY, *posZ;
		float* operator[](int index) const{
			switch (index)
			{
			case 0:
				return posX;
			case 1:
				return posY;
			case 2:
				return posZ;
			default:
				return nullptr;
			}
		}
		InputBuffers(size_t size, float* posX, float* posY, float* posZ) :size(size), posX(posX), posY(posY), posZ(posZ) {}
	};

}

#endif