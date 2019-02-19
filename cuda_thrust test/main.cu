#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <iostream>

//typedef float T;
template<typename T>
class CUDABuffer {
	T* ptr;
	size_t size;
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
};

__global__ void SetID(int* ids, int n) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < n) {
		ids[index] = index+1;
	}
}

void SetID(CUDABuffer<int> buffer) {
	cudaDeviceProp myCUDA;
	if (cudaGetDeviceProperties(&myCUDA, 0) != cudaSuccess) {
		std::cout << "error";
	}
	int threadsPerBlock = myCUDA.maxThreadsPerBlock;
	int bufferSize = buffer.Size();
	int blocksPerGrid = bufferSize / threadsPerBlock;
	if (blocksPerGrid == 0) {
		blocksPerGrid = 1; threadsPerBlock = bufferSize;
	}
	else {
		blocksPerGrid += bufferSize % threadsPerBlock ? 1 : 0;
	}
	SetID<<<blocksPerGrid, threadsPerBlock>>>(buffer.Value(), bufferSize);
}

void BuildKdTree(CUDABuffer<float> buffer) {

}


int main(void)
{
	int sss = 0;
	std::cin >> sss;
	CUDABuffer<int> buffer(sss);
	SetID(buffer);
	if (cudaDeviceSynchronize()) {
		std::cout << "!*!*!**!*!*";
	}
	SetID(buffer);
	if (cudaDeviceSynchronize()) {
		std::cout << "!*!*!**!*!*";
	}
	//thrust::device_ptr<int> buffer_ptr(buffer.Value());

	//thrust::device_vector<int> k = thrust::device_vector<int>(buffer_ptr, buffer_ptr+sss);
	//

	//// H has storage for 4 integers
	//thrust::host_vector <int> H = k;

	//for each (auto i in H)
	//{
	//	std::cout << i << std::endl;
	//}


	system("PAUSE");
	return 0;
}