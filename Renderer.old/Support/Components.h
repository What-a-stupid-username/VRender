#pragma once

#include "CommonInclude.h"
#include "Basic.h"
#include "OptiXLayer.h"
#include <regex>
#include <sstream> 
#include <set>
#include "Support/objLoader.h"

#define SAFE_RELEASE_OPTIX_OBJ(obj) if (obj != NULL) { obj->destroy(); obj = NULL; }

class OptiXLayer;
class VMaterial;
class VGeometryFilter;
class VObject;

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
	static unordered_map<string, VShader*> GetAllShaders();
};


class VTexture {
	friend class VMaterial;
	Buffer buffer = NULL;
	TextureSampler sampler = NULL;

	VTexture(string path);
	~VTexture();

public:
	static VTexture* Find(string path);
};





class VMaterial {
	friend class VObject;
	friend class VTransform;
private:
	string name;
	string shader_name;
	unordered_map<string, VProperty> properties;
	unordered_map<VObject*, function<void()>> reference;
public:

	Material mat = NULL;
private:
	VMaterial() { name = shader_name = ""; }
	~VMaterial() {}
	VMaterial(string name);
	void ReloadShader();
	void Release(VObject* obj);
	void Reload(string name);
	void SetShaderAsShaderProperties();
public:
	inline string GetName() { return name; }
	void SaveToFile();

	void ApplyPropertiesChange();


	template<typename T>
	void ChangeProperty(string name, T value);

	void MarkDirty();
	inline const unordered_map<string, VProperty>& GetAllProperties() { return properties; }
private:
public:
	static VMaterial* Find(string name);
	static unordered_map<string, VMaterial*> GetAllMaterials();
	static bool ApplyAllChanges();
	static void ReloadMaterial(string name);
};




class VMesh {
	friend class VGeometryFilter;

	int ref_cout = 0;

	string name;

	Buffer vert_buffer = NULL;
	Buffer normal_buffer = NULL;
	Buffer tex_buffer = NULL;
	Buffer v_index_buffer = NULL;
	Buffer n_index_buffer = NULL;
	Buffer t_index_buffer = NULL;

	VMesh(string name);
	~VMesh();
	void Release();
public:
	static VMesh* Find(string name);
};



class VGeometry {
	friend class VGeometryFilter;
private:
	int ref_cout = 0;
	string name;
	Program bound;
	Program intersect;
private:
	VGeometry() {}
	VGeometry(string name);
	void Release();
public:
	static VGeometry* Find(string name);
};

//todo: Move to GeometryTriangles to get higher acceleration rate on RTX boards
class VGeometryFilter {
	friend class VObject;
	friend class VTransform;
private:
	Geometry geometry;
	VGeometry* geometry_shader = NULL;
	VObject* object = NULL;
	VMesh* mesh = NULL;
	VGeometryFilter(VGeometry* geometry = NULL);
	void Release() {
		if (mesh) mesh->Release();
		geometry->destroy();
		geometry_shader->Release();
	}
public:
	Handle<VariableObj> Visit(const char* varname);
	inline void SetPrimitiveCount(size_t count) { this->geometry->setPrimitiveCount(count); }
	void SetMesh(VMesh* mesh);
};

class VTransform {
public:
	friend class VObject;
	friend class OptiXLayer;

	Group group = NULL;
	Transform transform = NULL;

	VTransform* parent = NULL;
	VObject* object = NULL;
	set<VTransform*> childs;

	float3 pos, rotate, scale;

	void ApplyPropertiesChange();

	VTransform();
public:
	void Setparent(VTransform* trans = NULL);

	void AddChild(VTransform* trans);

	inline set<VTransform*> Childs() { return childs; }
	inline VObject* Object() { return object; }

	static VTransform* Root();

	void MarkDirty();
	
	template<typename T>
	T* Position() {
		return (T*)&pos;
	}
	template<typename T>
	T* Rotation() {
		return (T*)&rotate;
	}
	template<typename T>
	T* Scale() {
		return (T*)&scale;
	}

	void Release();

	static bool ApplyAllChanges();
};

class VObject {
	friend class VTransform;
public:
	string name = "";
private:
	GeometryGroup hook;
	GeometryInstance go;
	VTransform* transform = NULL;
	VMaterial* material = NULL;
	VGeometryFilter* geometryFilter = NULL;
	void RebindMaterial();
	~VObject() {
		hook->getAcceleration()->destroy();
		hook->destroy();
		if (material != NULL) material->Release(this);
		geometryFilter->Release();
		go->destroy();
	}
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

