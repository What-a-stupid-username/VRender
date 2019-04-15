#ifndef GPU_KDTREE_H
#define GPU_KDTREE_H

#include "gpu_kdtree_basic.cuh";

namespace gpu_kdtree {

	//extern "C" void BuildKdTree(const gpu_kdtree::InputBuffers& input);
	extern "C" void BuildKdTree(float3* input, int size);

}


#endif