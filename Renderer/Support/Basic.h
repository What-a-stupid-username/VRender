#pragma once

#include "CommonInclude.h"
#include <unordered_map>

#include <iostream>
#include <fstream>
#include <string>
#include <functional>

class StringToID {
public:
	static int ID(string str);
};

class PropertyReader;

struct VProperty {
	friend class PropertyReader;
private:
	string type;
	void* data;
public:
	VProperty(string type, void* data) :type(type), data(data) {};

	template<typename T>
	void SetProperty(T obj) {
#define TYPE_OP2(x, y) else if (type == "x") obj[type]->y((x*)data)
		if (0);
		TYPE_OP2(float, set1fv);
		TYPE_OP2(float2, set2fv);
		TYPE_OP2(float3, set3fv);
		TYPE_OP2(float4, set4fv);
		TYPE_OP2(int, set1iv);
	}
	template<typename T>
	inline T* GetData() { return (T*)data; };

	void Release() {
#define TYPE_OP(x) else if (type == "x") delete((x*)data)
		if (0);
		TYPE_OP(float);
		TYPE_OP(float2);
		TYPE_OP(float3);
		TYPE_OP(float4);
		TYPE_OP(int);
	}
};




class PropertyWriter {
	FILE* fp;
	unordered_map<string, string> table;

	template<typename T>
	inline string ToString(const T v) { return to_string(v); }
	inline string ToString(const float2 v) { return ToString(v.x) + ',' + ToString(v.y); }
	inline string ToString(const float3 v) { return ToString(v.x) + ',' + ToString(v.y) + ',' + ToString(v.z); }
	inline string ToString(const float4 v) { return ToString(v.x) + ',' + ToString(v.y) + ',' + ToString(v.z) + ',' + ToString(v.w); }
	inline string ToString(const string v) { return v; }

public:
	PropertyWriter(string path, string name) {
		fp = fopen((path + "/" + name).c_str(), "w");
	};
	~PropertyWriter() {
		if (fp && !table.empty()) {
			for each (auto pair in table)
			{
				fprintf(fp, "%s: %s \n", pair.first.c_str(), pair.second.c_str());
				printf("%s: %s \n", pair.first.c_str(), pair.second.c_str());
			}
			fclose(fp);
		}
	}
	template<typename T>
	void AddProperty(string prop, T value) {
		table[prop] = ToString(value);
	}
};

template <typename T>
class Shit {public: static inline T ToProperty(string prop); };
template <typename T>
T Shit<T>::ToProperty(string str) { istringstream iss(str); T num; iss >> num; return num; }
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

	void Init(string path, string name) {
	}
public:
	PropertyReader(string path, string name) {
		fstream fin;
		fin.open((path + "/" + name).c_str(), ios::in);
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
#define TYPE_OP3(x) else if (type == "x") table[prop] = VProperty(type, new x)
			if (0);
			TYPE_OP3(float);
			TYPE_OP3(float2);
			TYPE_OP3(float3);
			TYPE_OP3(float4);
			TYPE_OP3(int);
			else table[prop] = VProperty(type, new char[value.size() + 1]);

#define TYPE_OP4(x) else if (type == "x") {x vv = GetProperty<x>(value); *(x*)table[prop].data = vv; }
			if (0);
			TYPE_OP4(float)
			TYPE_OP4(float2)
			TYPE_OP4(float3)
			TYPE_OP4(float4)
			TYPE_OP4(int)
			else { string vv = GetProperty<string>(value); memcpy(table[prop].data, vv.c_str(), vv.size()+1); }
		}
		fin.close();
	};

	template<typename T>
	T GetProperty(string prop) {
		string str = table[prop];
		return Shit <T>::ToProperty(str);
	}
};