#pragma once

#include "CommonInclude.h"
#include <unordered_map>

#include <iostream>
#include <fstream>
#include <string>
#include <functional>

namespace VRender {
	
	#define SAFE_RELEASE_OPTIX_OBJ(obj) if (obj != NULL) { obj->destroy(); obj = NULL; }

	class OptixInstance {
		optix::Context context;
		optix::Program attributeProgram;

		OptixInstance() {
			try
			{
				//#ifdef FORCE_NOT_USE_RTX
				//	int not_use_rtx = 0;
				//	rtGlobalSetAttribute(RT_GLOBAL_ATTRIBUTE_ENABLE_RTX, sizeof(int), (void*)& not_use_rtx);
				//#endif // FOURCE_NOT_USE_RTX

				context = optix::Context::create();

				context->setRayTypeCount(2);
				
				context["scene_epsilon"]->setFloat(1.e-3f);
				context["bad_color"]->setFloat(1000000.0f, 0.0f, 1000000.0f);
				context["bg_color"]->setFloat(make_float3(0.1f));

				const char* ptx = sutil::getPtxString("Geometries", "optixGeometryTriangles.cu");
				attributeProgram = context->createProgramFromPTXString(ptx, "triangle_attributes");

				float3_default = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 0);
				int3_default = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_INT3, 0);
				float2_default = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, 0);
				int_default = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_INT, 0);
			}
			catch (const std::exception& e)
			{
				cout << "Init failed" << endl << e.what() << endl;
				system("PAUSE");
			}
		}

		OptixInstance(OptixInstance& k) abandon;
	public:
		optix::Buffer float3_default, int3_default, float2_default, int_default;

		static OptixInstance& Instance();
		optix::Context Context() {
			return context;
		}

		optix::Program AttributeProgram() {
			return attributeProgram;
		}
	};




	class StringToID {
	public:
		static int ID(string str);
	};

	class PropertyReader;

	struct VProperty {
		friend class PropertyReader;
		friend class PropertyWriter;
	private:
		string type;
		void* data;
	public:
		VProperty(string type = "string", void* data = 0) :type(type), data(data) {};

		template<typename T>
		void SetProperty(T& obj, string name) {
#define TYPE_OP2(x, y) else if (type == #x) obj[name]->y((float*)data)
			if (0);
			TYPE_OP2(float, set1fv);
			TYPE_OP2(float2, set2fv);
			TYPE_OP2(float3, set3fv);
			TYPE_OP2(float4, set4fv);
			else if (type == "int") obj[name]->set1iv((int*)data);
		}
		template<typename T>
		inline T * GetData() { return (T*)data; };

		template<>
		inline string* GetData<string>() { return (string*)data; };

		inline const string& Type() { return type; }

		void Release() {
			delete[] data;
		}
	};

	template<typename T>
	class Shit2 { public: static inline string GetTypeName(); };
	#define TYPE_OP5(x) template <> string Shit2<x>::GetTypeName() { return #x; }
	TYPE_OP5(int);
	TYPE_OP5(float);
	TYPE_OP5(float2);
	TYPE_OP5(float3);
	TYPE_OP5(float4);
	template <> string Shit2<const char*>::GetTypeName() { return "string"; }

	template<typename T>
	string GetType() {
		return Shit2<T>::GetTypeName();
	}


	class PropertyWriter {
		FILE* fp;
		unordered_map<string, string> table;
	public:
		template<typename T>
		static inline string ToString(const T v) { return to_string(v); }
		static inline string ToString(const float v) { return to_string(v); }
		static inline string ToString(const int v) { return to_string(v); }
		static inline string ToString(const float2 v) { return ToString(v.x) + ',' + ToString(v.y); }
		static inline string ToString(const float3 v) { return ToString(v.x) + ',' + ToString(v.y) + ',' + ToString(v.z); }
		static inline string ToString(const float4 v) { return ToString(v.x) + ',' + ToString(v.y) + ',' + ToString(v.z) + ',' + ToString(v.w); }
		static inline string ToString(const char* v) { return string(v); }
		static inline string ToString(const string v) { return v; }

		template<typename T>
		string GetTypeName() {
			return Shit2<T>::GetTypeName();
		}

	public:
		PropertyWriter(string path, string name) {
			fp = fopen((string(sutil::projectDir()) + path + "/" + name).c_str(), "w");
		};
		~PropertyWriter() {
			if (fp && !table.empty()) {
				for each (auto pair in table)
				{
					fprintf(fp, "%s:%s\n", pair.first.c_str(), pair.second.c_str());
					printf("%s:%s\n", pair.first.c_str(), pair.second.c_str());
				}
				fclose(fp);
			}
		}
		template<typename T>
		void AddProperty(string prop, T value) {
			prop = GetTypeName<T>() + ":" + prop;
			table[prop] = ToString(value);
		}

		void AddProperty(string prop, VProperty value) {
			prop = value.type + ":" + prop;
	#define TYPE_OP6(x) else if (value.type == #x) table[prop] = ToString(*value.GetData<x>())
			if (0);
			TYPE_OP6(int);
			TYPE_OP6(float);
			TYPE_OP6(float2);
			TYPE_OP6(float3);
			TYPE_OP6(float4);
			TYPE_OP6(string);
		}
	};

	template <typename T>
	class Shit { public: static inline T ToProperty(string prop); };
	template <>
	float Shit<float>::ToProperty(string str) { float res; sscanf(&str[0], "%f", &res); return res; }
	template <>
	int Shit<int>::ToProperty(string str) { int res; sscanf(&str[0], "%d", &res); return res; }
	template <>
	float2 Shit<float2>::ToProperty(string str) { float2 res; sscanf(&str[0], "%f,%f", &res.x, &res.y); return res; }
	template <>
	float3 Shit<float3>::ToProperty(string str) { float3 res; sscanf(str.c_str(), "%f,%f,%f", &res.x, &res.y, &res.z); return res; }
	template <>
	float4 Shit<float4>::ToProperty(string str) { float4 res; sscanf(&str[0], "%f,%f,%f,%f", &res.x, &res.y, &res.z, &res.w); return res; }
	template <>
	string Shit<string>::ToProperty(string str) { return str; }


	class PropertyReader {
		unordered_map<string, VProperty> table;

		template<typename T>
		T GetPropertyFromString(string value) {
			return Shit <T>::ToProperty(value);
		}
	public:
		PropertyReader(string path, string name) {
			fstream fin;
			fin.open((string(sutil::projectDir()) + path + "/" + name).c_str(), ios::in);
			string tmp;
			while (getline(fin, tmp)) {
				string type, prop, value;
				int k = tmp.find_first_of(":");
				type = tmp.substr(0, k);

				prop = tmp.substr(k + 1, tmp.size() - k);
				tmp = prop;
				k = tmp.find_first_of(":");
				prop = tmp.substr(0, k);
				value = tmp.substr(k + 1, tmp.size() - k);
	#define TYPE_OP3(x) else if (type == #x) table[prop] = VProperty(type, new x)
				if (0);
				TYPE_OP3(float);
				TYPE_OP3(float2);
				TYPE_OP3(float3);
				TYPE_OP3(float4);
				TYPE_OP3(int);
				else table[prop] = VProperty(type, new string());

	#define TYPE_OP4(x) else if (type == #x) {x vv = GetPropertyFromString<x>(value); *(x*)table[prop].data = vv; }
				if (0);
				TYPE_OP4(float)
					TYPE_OP4(float2)
					TYPE_OP4(float3)
					TYPE_OP4(float4)
					TYPE_OP4(int)
					TYPE_OP4(string)
			}
			fin.close();
		};



		VProperty GetProperty(string value) {
			if (table.find(value) != table.end()) {
				return table[value];
			}
			return VProperty{ "",0 };
		}

		template<typename T>
		T GetPropertyValue(string value) {
			if (table.find(value) != table.end()) {
				return *(T*)table[value].data;
			}
			return T();
		}

		unordered_map<string, VProperty> GetAllProperties() {
			return table;
		}
	};
}