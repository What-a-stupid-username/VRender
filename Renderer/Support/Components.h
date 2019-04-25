#pragma once

#include "CommonInclude.h"
#include "Basic.h"
#include "OptiXLayer.h"
#include <regex>
#include <sstream> 
#include <set>
#include "Support/objLoader.h"

#define SAFE_RELEASE_OPTIX_OBJ(obj) if (obj != NULL) obj->destroy()

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
	static unordered_map<string, VShader*> GetAllShaders();
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

	Material mat = NULL;
private:
	VMaterial() { name = shader_name = ""; }
	~VMaterial() {}
	VMaterial(string name);
	void ReloadShader();
	void Release();
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




class VGeometryFilter;
class VMesh {
	friend class VGeometryFilter;

	Buffer vert_buffer = NULL;
	Buffer normal_buffer = NULL;
	Buffer tex_buffer = NULL;
	Buffer v_index_buffer = NULL;
	Buffer n_index_buffer = NULL;
	Buffer t_index_buffer = NULL;

	VMesh(string name);
	~VMesh();

public:
	static VMesh* Find(string name);
};



class VGeometry {
	friend class VGeometryFilter;
private:
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
	void SetMesh(VMesh* mesh);
};


class OptiXLayer;
class VTransform {
	friend class VObject;
	friend class OptiXLayer;
	Group group = NULL;
	Transform transform;

	VTransform* parent = NULL;
	VObject* object = NULL;
	set<VTransform*> childs;

	float3 pos, rotate, scale;

	void ApplyPropertiesChange() {
		Matrix4x4 mat;
		mat = Matrix4x4::scale(scale);

		mat = Matrix4x4::rotate(rotate.x / 180 * M_PI, make_float3(1, 0, 0)) * mat;
		mat = Matrix4x4::rotate(rotate.y / 180 * M_PI, make_float3(0, 1, 0)) * mat;
		mat = Matrix4x4::rotate(rotate.z / 180 * M_PI, make_float3(0, 0, 1)) * mat;

		mat = Matrix4x4::translate(pos) * mat;
		transform->setMatrix(false, mat.getData(), NULL);
	}

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

	static bool ApplyAllChanges();
};

class VObject {
public:
	string name = "";
private:
	GeometryGroup hook;
	GeometryInstance go;
	VTransform* transform = NULL;
	VMaterial* material = NULL;
	VGeometryFilter* geometryFilter = NULL;
	void RebindMaterial();
	~VObject() { }
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
