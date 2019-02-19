#include <optixu/optixu_math_namespace.h>
#include "cuda_runtime.h"
#include "optixPathTracer.h"
#include "random.h"

using namespace optix;

rtDeclareVariable(uint, launch_index, rtLaunchIndex, );

rtBuffer<int, 1>              length_buffer;
rtBuffer<int, 1>              intput_buffer;
rtBuffer<int, 1>              output_buffer;

inline void append(int value) {
	int index = atomicAdd(&length_buffer[0], 1);
	output_buffer[index] = value;
}

inline void swap(int &a, int &b) {
	int temp = a;
	a = b;
	b = temp;
}

void bitonic_sort() {
	output_buffer[launch_index] = intput_buffer[launch_index];
	__threadfence();
	int sum = intput_buffer.size();


	for (unsigned int i = 2; i <= sum; i <<= 1) {
		for (unsigned int j = i >> 1; j>0; j >>= 1) {
			unsigned int tid_comp = launch_index ^ j;
			if (tid_comp < sum) {
				if (tid_comp > launch_index) {
					if ((launch_index & i) == 0) { //ascending
						if (output_buffer[launch_index]>output_buffer[tid_comp]) {
							swap(output_buffer[launch_index], output_buffer[tid_comp]);
						}
					}
					else { //desending
						if (output_buffer[launch_index]<output_buffer[tid_comp]) {
							swap(output_buffer[launch_index], output_buffer[tid_comp]);
						}
					}
				}
			}
			__threadfence();
		}
	}
}

RT_PROGRAM void sort()
{
	bitonic_sort();
}


RT_PROGRAM void exception()
{
	output_buffer[launch_index] = -1;
}

RT_PROGRAM void miss()
{
	
}


