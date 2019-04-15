#include <iostream>

#include "gpu_kdtree.cuh"

using namespace std;

using namespace gpu_kdtree;

struct float3_
{
	float x, y, z;
};

void Test() {
	const int size = 50;
	CUDABuffer<float3_> pos(size);

	float3_ pos_host[size];
	{
		for (int i = 0; i < size; i++) {
			pos_host[i].x = 197 * i % 11;
			pos_host[i].y = i;
			pos_host[i].z = size - i;
		}
		pos.SetValue(pos_host);
	}

	BuildKdTree((float3*)pos.Value(),  size);
}

int main(void)
{
	Test();

	system("PAUSE");
	return 0;
}