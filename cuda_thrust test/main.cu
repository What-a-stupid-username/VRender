#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <iostream>
#include <thrust/functional.h>

typedef float t_t;


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
};

__global__ void SetV(t_t* data, int n) {
	int index = blockDim.x * threadIdx.y + threadIdx.x;
	if (index < n) {
		data[index] = 197 * index % 17 + 1;
	}
}

void SetV(CUDABuffer<t_t>& buffer) {
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
	SetV<<<blocksPerGrid, threadsPerBlock>>>(buffer.Value(), bufferSize);
}

void BuildKdTree(CUDABuffer<float> buffer) {

}


int main(void)
{
	int sss = 0;
	std::cin >> sss;
	CUDABuffer<t_t> buffer(sss);
	t_t* res = new t_t[sss];
	for (int i = 0; i < sss; i++)
	{
		res[i] = i+1;
	}
	cudaMemcpy(buffer.Value(), res, sss * sizeof(t_t), cudaMemcpyKind::cudaMemcpyHostToDevice);
	for (int i = 0; i < sss; i++)
	{
		res[i] = 0;
	}

	SetV(buffer);

	if (cudaDeviceSynchronize()) {
		std::cout << "!*!*!**!*!*";
	}
	cudaMemcpy(res, buffer.Value(), sss * sizeof(t_t), cudaMemcpyKind::cudaMemcpyDeviceToHost);
	for (int i = 0; i < sss; i++)
	{
		std::cout << res[i] << std::endl;
	}

	thrust::device_ptr<t_t> buffer_ptr(buffer.Value());
	
	thrust::sort(buffer_ptr, buffer_ptr + sss);

	cudaMemcpy(res, buffer.Value(), sss * sizeof(t_t), cudaMemcpyKind::cudaMemcpyDeviceToHost);
	for (int i = 0; i < sss; i++)
	{
		std::cout << res[i] << std::endl;
	}

	system("PAUSE");
	return 0;
}