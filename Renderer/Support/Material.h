#pragma once

#include "CommonInclude.h"
#include "Basic.h"
#include "OptiXLayer.h"
#include <regex>
#include <sstream> 
#include <set>

class VMaterial;
class VShader {
	friend class VMaterial;
	unordered_map<VMaterial*, function<void()>> reference;
	unordered_map<int, Program> closestHitProgram;
	unordered_map<int, Program> anyHitProgram;
public:
	VShader() {};

	VShader(string shader_name);

	void LoadFromFile(string shader_name);

	void Release();
private:
public:
	static VShader* Find(string name);
	static void Reload(string name);
};

class VObject;
class VMaterial {
	friend class VObject;
private:
	string name;
	string shader_name;
	unordered_map<string, VProperty> properties;
	unordered_map<VObject*, function<void()>> reference;
public:

	Material mat;
private:
	VMaterial() { name = shader_name = ""; }
	~VMaterial() {}
	VMaterial(string name);
	void ReloadShader();
	void Release();
public:
	inline string GetName() { return name; }
	void SaveToFile();

	void ApplyPropertiesChange();

	template<typename T>
	void ChangeProperty(string name, T value);

	void MarkDirty();
private:
public:
	static VMaterial* Find(string name);
	static unordered_map<string, VMaterial*> GetAllMaterials();
	static void ApllyAllChanges();
};

class VGeometry {
public:
	Program bound;
	Program intersect;
private:
	VGeometry() {}
	VGeometry(string name);

public:
	static VGeometry* Find(string name);
};


class VObject;
class VGeometryFilter {
	friend class VObject;
private:
	Geometry geometry;
	VGeometry* geometry_shader = NULL;
	VObject* object = NULL;
	VGeometryFilter(VGeometry* geometry);
public:
	Handle<VariableObj> Visit(const char* varname);
	inline void SetPrimitiveCount(size_t count) { this->geometry->setPrimitiveCount(count); }
};



class VTransform {
	friend class VObject;
	Group group = NULL;
	Transform transform;
	GeometryGroup hook;

	VTransform* parent = NULL;
	set<VTransform*> childs;

	VTransform();
public:
	void Setparent(VTransform* trans = NULL);

	void AddChild(VTransform* trans);

	static VTransform* Root();

	void MarkDirty();

	Group Group() { return group; }
};

class VObject {
	VTransform* transform = NULL;
	VMaterial* material = NULL;
	VGeometryFilter* geometryFilter = NULL;
	void RebindMaterial();
	~VObject() { }
public:
	GeometryInstance go;
public:
	void MarkDirty() { transform->MarkDirty(); }
	VObject(string geometry_shader_name);
	inline VTransform* const Transform() { return transform; };
	void SetMaterial(VMaterial* mat);
	inline VGeometryFilter* const GeometryFilter() { return geometryFilter; }
};

template<typename T>
void VMaterial::ChangeProperty(string name, T value) {
	*(T*)properties[name].data = value;
	MarkDirty();
}
