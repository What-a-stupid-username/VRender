#include "Basic.h"

namespace VRender {

	static int next_id;
	static unordered_map<string, int> table;

	int StringToID::ID(string str) {
		auto& id = table[str];
		if (id) return id;
		if (next_id == 0) next_id = 1;
		id = next_id++;
		table[str] = id;
		return id;
	}

}