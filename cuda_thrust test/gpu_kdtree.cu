#include "gpu_kdtree.cuh"
#include "gpu_kdtree_helper.cuh"
#include <vector>
#include <algorithm>

using namespace std;

namespace gpu_kdtree {

	struct TreeNode
	{
		int split_axil_leaf_flag;
		float split_size;
		int id;
	};

	struct KDTree
	{
		TreeNode* root;
		int num;
	};




	int BuildKdTree(KDTree* tree, thrust::device_ptr<float3>& input, thrust::device_ptr<int>& id, int start, int size) {
		int num = tree->num++;
		if (size < 8) {
			TreeNode node;
			node.split_axil_leaf_flag = -1;
			node.split_size = size;
			node.id = start;
			tree->root[num] = node;
			return num;
		}

		BBox bbox(input, size);
		float lengths_of_bbox[3];
		{
			for (int i = 0; i < 3; i++)
			{
				lengths_of_bbox[i] = bbox.max[i] - bbox.min[i];
			}
		}

		int axil = -1;
		int spit = size / 2;
		{
			float max_l = -1;
			for (int i = 0; i < 3; i++)
			{
				if (lengths_of_bbox[i] > max_l) {
					axil = i;
					max_l = lengths_of_bbox[i];
				}
			}
		}

		if (axil == 0)
			thrust::sort_by_key(input + start, input + size, id + start, lessFloat3X());
		else if (axil == 1)
			thrust::sort_by_key(input + start, input + size, id + start, lessFloat3Y());
		else
			thrust::sort_by_key(input + start, input + size, id + start, lessFloat3Z());

		TreeNode node;
		node.split_axil_leaf_flag = axil;
		node.split_size = size;
		node.id = start;
		tree->root[num] = node;
		return num;
	}

	void BuildKdTree(float3* input, int size) {
		thrust::device_ptr<float3> ptr(input);
		KDTree tree;
		tree.root = new TreeNode[size*size];
		tree.num = 0;

		thrust::device_vector<int> id;
		thrust::sequence(id.begin(), id.end());

		BuildKdTree(&tree, ptr, id.data(), 0, size);
	}
}

//int BuildKdTree_host_(KDTree* tree, thrust::host_vector<int>* id, const gpu_kdtree::InputBuffers& input, int s, int e, BBox bbox) {

//	int num;

//	int size = e - s;
//	float lengths_of_bbox[3];
//	{
//		for (int i = 0; i < 3; i++)
//		{
//			lengths_of_bbox[i] = bbox.max[i] - bbox.min[i];
//		}
//	}

//	float min_cost = size - 8/*travel time*/;
//	int split_axil = -1;
//	int split_index = -1;
//	for (int i = 0; i < 3; i++)
//	{
//		float s = lengths_of_bbox[(i + 1) % 3] * lengths_of_bbox[(i + 2) % 3];
//		for (int j = 0; j < size; j++)
//		{
//			int index = id[i][j];
//			int num_of_left_node = j;
//			float possibility_left = (input[i][index] - bbox.min[i]) / lengths_of_bbox[i];
//			float cost = num_of_left_node * possibility_left + (size - num_of_left_node) * (1 - possibility_left);
//			if (cost < min_cost) {
//				split_axil = i;
//				split_index = j;
//				min_cost = cost;
//			}
//		}
//	}
//	if (split_axil == -1) {//leaf
//		TreeNode node;
//		node.split_axil_leaf_flag = -1;
//		node.s = s;
//		node.e = e;
//		node.split_value = 0;
//		num = tree->num++;
//		tree->root[num] = node;
//	}
//	else
//	{
//		TreeNode node;
//		node.split_axil_leaf_flag = split_axil;
//		node.s = BuildKdTree_host_(tree, id, input, s, split_index, bbox);
//		node.e = BuildKdTree_host_(tree, id, input, split_index, e, bbox);
//		node.split_value = 0;
//		num = tree->num++;
//		tree->root[num] = node;
//	}
//	return num;
//}

//void BuildKdTree_host(const gpu_kdtree::InputBuffers& input) {

//	int size = input.size;

//	thrust::host_vector<int> id[3];
//	{
//		// init ids
//		id[0] = thrust::host_vector<int>(size);
//		thrust::sequence(id[0].begin(), id[0].end());
//		id[1] = thrust::host_vector<int>(id[0]);
//		id[2] = thrust::host_vector<int>(id[0]);
//		//sort
//		for (int i = 0; i < 3; i++)
//		{
//			const float* buffer_ptr = input[i];
//			thrust::host_vector<float> copy(buffer_ptr, buffer_ptr + size);//copy
//			thrust::sort_by_key(copy.begin(), copy.end(), id[i].begin(), thrust::less<float>());
//		}
//	}

//	BBox bbox;
//	{
//		for (int i = 0; i < 3; i++)
//		{
//			bbox.min[i] = input[i][id[i].front()];
//			bbox.max[i] = input[i][id[i].back()];
//		}
//	}

//	KDTree tree;
//	tree.root = new TreeNode[size*size];
//	int NodeNum = 0;

//	BuildKdTree_host_(&tree, id, input, 0, size, bbox);
//}


//void BuildKdTree(const InputBuffers& input) {
//	int size = input.size;

//	BBox bbox(input, 0, 1);
//	float lengths_of_bbox[3];
//	{
//		for (int i = 0; i < 3; i++)
//		{
//			lengths_of_bbox[i] = bbox.max[i] - bbox.min[i];
//		}
//	}
//	int axil = -1;
//	{		
//		float max_l = -1;
//		for (int i = 0; i < 3; i++)
//		{
//			if (lengths_of_bbox[i] > max_l) {
//				axil = i;
//				max_l = lengths_of_bbox[i];
//			}
//		}
//	}

//	KDTree tree;
//	tree.root = new TreeNode[size*size];
//	tree.num = 0;

//	thrust::device_vector<int> id;
//	{
//		id = thrust::device_vector<int>(size);
//		thrust::sequence(id.begin(), id.end());
//		{
//			thrust::device_ptr<float> buffer_ptr(input[axil]);
//			thrust::device_vector<float> copy(buffer_ptr, buffer_ptr + size);//copy
//			thrust::sort_by_key(copy.begin(), copy.end(), id.begin(), thrust::less<float>());
//		}
//	}




//	//thrust::device_vector<int> id[3];
//	//{
//	//	// init id
//	//	id[0] = thrust::device_vector<int>(size);
//	//	thrust::sequence(id[0].begin(), id[0].end());
//	//	id[1] = thrust::device_vector<int>(id[0]);
//	//	id[2] = thrust::device_vector<int>(id[0]);
//	//	//sort
//	//	for (int i = 0; i < 3; i++)
//	//	{
//	//		const float* buffer_ptr = input[i];
//	//		thrust::device_vector<float> copy(buffer_ptr, buffer_ptr + size);//copy
//	//		thrust::sort_by_key(copy.begin(), copy.end(), id[i].begin(), thrust::less<float>());
//	//	}
//	//}
//	



//}