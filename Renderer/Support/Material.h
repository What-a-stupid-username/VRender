#pragma once

#include "CommonInclude.h"
#include "Basic.h"


class VMaterialManager;

class VMaterial {
	friend class VMaterialManager;
private:
	int id;
	string name;
public:

private:
	VMaterial() {}
	~VMaterial() = default;

public:
	inline int GetID() { return id; }
	inline string GetName() { return name; }
	void SaveToFile(string path = "./") {
		PropertyWriter io = PropertyWriter(path, name);
		optix::Material mat;
		//mat["aa"]->set1iv
	}
};

class VMaterialManager {
private:
	int next_id = 1;
	unordered_map<int, VMaterial*> table;
private:
	VMaterialManager() {}
	VMaterialManager(const VMaterialManager&) abandon;
	VMaterialManager& operator=(const VMaterialManager&) abandon;
public:
	static VMaterialManager& VMaterialManager::Instance();

	VMaterial * GetByID(int id) {
		auto mat = table[id];
		return mat;
	}

	VMaterial* LoadByPath(string path) {

	}
};